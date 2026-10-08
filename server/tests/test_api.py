import base64
import copy
import math
import struct
import uuid

import numpy as np
import pytest
from sqlalchemy import func, select
from collector.cli import export_runs
from collector.db import Run, Sample, Token


def post(system, payload, key="headers"):
    return system["client"].post("/v1/ingest", json=payload, headers=system[key])


def close(payload, count=1, end_ns=2000000000):
    result = copy.deepcopy(payload)
    result["run"].update(revision=2, expected_samples=count, end_wall_ms=1790000001000, end_elapsed_ns=end_ns, end_reason="app_switch")
    return result


def test_requires_auth(system, payload):
    assert system["client"].post("/v1/ingest", json=payload).status_code == 401


def test_login_and_revocation(system):
    client = system["client"]
    result = client.post("/v1/auth/login", json={"username": " ALICE ", "password": "correct-password-123"})
    assert result.status_code == 200
    headers = {"Authorization": "Bearer " + result.json()["token"]}
    assert client.get("/v1/models", headers=headers).status_code == 200
    assert client.post("/v1/auth/logout", headers=headers).status_code == 200
    assert client.get("/v1/models", headers=headers).status_code == 401


def test_login_throttle_and_no_password_echo(system):
    for _ in range(3):
        result = system["client"].post("/v1/auth/login", json={"username": "alice", "password": "DO-NOT-ECHO-THIS"})
        assert result.status_code == 401
        assert "DO-NOT-ECHO-THIS" not in result.text
    assert system["client"].post("/v1/auth/login", json={"username": "alice", "password": "wrong"}).status_code == 429


def test_duplicate_network_retry_is_idempotent(system, payload):
    first = post(system, payload)
    assert first.status_code == 200, first.text
    retry = post(system, payload)
    assert retry.json() == first.json()
    with system["app"].state.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Sample)) == 1
        assert session.get(Run, payload["run"]["id"]).received_samples == 1


def test_conflicting_retry_rolls_back_entire_batch(system, payload):
    assert post(system, payload).status_code == 200
    conflict = copy.deepcopy(payload)
    new = copy.deepcopy(conflict["samples"][0])
    new.update(sequence=1, elapsed_ns=1100000000)
    conflict["samples"].append(new)
    conflict["samples"][0]["activity"] = "different.Activity"
    assert post(system, conflict).status_code == 409
    with system["app"].state.sessions() as session:
        assert session.scalar(select(func.count()).select_from(Sample)) == 1


def test_close_then_late_open_retry_never_reopens(system, payload):
    assert post(system, close(payload)).json()["complete"] is True
    retry = post(system, payload)
    assert retry.status_code == 200
    assert retry.json()["complete"] is True
    assert retry.json()["committed_revision"] == 2


def test_out_of_order_chunks_complete_only_when_gap_filled(system, payload):
    second = copy.deepcopy(payload)
    second["samples"][0].update(sequence=1, elapsed_ns=1200000000)
    assert post(system, close(second, 2)).json()["complete"] is False
    result = post(system, payload)
    assert result.status_code == 200 and result.json()["complete"] is True


def test_app_return_gets_new_run_number(system, payload):
    first = post(system, payload)
    second = copy.deepcopy(payload)
    second["run"]["id"] = str(uuid.uuid4())
    assert post(system, second).json()["run_number"] == first.json()["run_number"] + 1


def test_user_isolation(system, payload):
    assert post(system, payload).status_code == 200
    assert post(system, payload, "other_headers").status_code == 409
    assert system["client"].get(f"/v1/runs/{payload['run']['id']}/samples", headers=system["other_headers"]).status_code == 404
    assert system["client"].get("/v1/runs?complete_only=false", headers=system["other_headers"]).json() == []


@pytest.mark.parametrize("field,value", [("model_id", "b" * 64), ("package_name", "other.app"), ("device_model", "changed")])
def test_immutable_run_metadata(system, payload, field, value):
    assert post(system, payload).status_code == 200
    payload["run"][field] = value
    assert post(system, payload).status_code == 409


@pytest.mark.parametrize("kind", ["zero", "nan", "infinity", "oversize", "invalid_base64"])
def test_embedding_validation(system, payload, kind):
    if kind == "invalid_base64":
        value = "!" * 2048
    elif kind == "oversize":
        value = "A" * 4096
    else:
        first = {"zero": 0.0, "nan": float("nan"), "infinity": float("inf")}[kind]
        value = base64.b64encode(struct.pack("<384f", first, *([0.0] * 383))).decode()
    payload["samples"][0]["embedding_b64"] = value
    result = post(system, payload)
    assert result.status_code == 422
    assert value not in result.text


def test_rejects_screenshot_field(system, payload):
    payload["samples"][0]["screenshot"] = "private-screen-data"
    result = post(system, payload)
    assert result.status_code == 422
    assert "private-screen-data" not in result.text


def test_bounded_request_size(system):
    result = system["client"].post("/v1/ingest", content=b"x" * 2100000, headers=system["headers"])
    assert result.status_code == 413


def test_duplicate_sequence_inside_batch_rejected(system, payload):
    payload["samples"].append(copy.deepcopy(payload["samples"][0]))
    assert post(system, payload).status_code == 422


def test_monotonic_time_order_across_batches(system, payload):
    assert post(system, payload).status_code == 200
    payload["samples"][0].update(sequence=1, elapsed_ns=1000000000)
    assert post(system, payload).status_code == 409


def test_closing_cannot_drop_existing_samples(system, payload):
    assert post(system, payload).status_code == 200
    closing = close(payload, 0)
    closing["samples"] = []
    assert post(system, closing).status_code == 409


def test_closing_time_cannot_precede_existing_sample(system, payload):
    assert post(system, payload).status_code == 200
    closing = close(payload, end_ns=1000000000)
    closing["samples"] = []
    assert post(system, closing).status_code == 409


def test_only_complete_runs_are_exported(system, payload):
    assert post(system, payload).status_code == 200
    assert system["client"].get("/v1/runs", headers=system["headers"]).json() == []
    assert post(system, close(payload)).status_code == 200
    with system["app"].state.sessions() as session:
        destination = system["tmp_path"] / "export"
        assert export_runs(session, system["owner"], destination) == 1
    files = list(destination.rglob("embeddings.npy"))
    assert len(files) == 1
    data = np.load(files[0], allow_pickle=False)
    assert data.shape == (1, 384) and data[0, 0] == 1.0


def test_unknown_activity_stays_unknown(system, payload):
    payload["samples"][0].update(activity=None, activity_source="unknown")
    assert post(system, close(payload)).status_code == 200
    result = system["client"].get(f"/v1/runs/{payload['run']['id']}/samples", headers=system["headers"]).json()
    assert result[0]["activity"] is None


def test_activity_provenance_cannot_contradict_value(system, payload):
    payload["samples"][0]["activity_source"] = "unknown"
    assert post(system, payload).status_code == 422


@pytest.mark.parametrize("package", ["..", ".", "com..example", "com.example/../../outside"])
def test_package_cannot_escape_export_directory(system, payload, package):
    payload["run"]["package_name"] = package
    assert post(system, payload).status_code == 422


def test_opencl_mode_metadata_is_accepted_and_legacy_defaults_remain(system, payload):
    payload["samples"][0].update(processing_backend="litert_opencl_full_encoder", preprocessing_backend="opencl", capture_source="accessibility", sampling_mode="BATTERY_SAVER", head_ms=0.0)
    response=system["client"].post("/v1/ingest",json=payload,headers=system["headers"])
    assert response.status_code == 200


def test_invalid_processing_backend_is_rejected(system, payload):
    payload["samples"][0]["processing_backend"]="fabricated_gpu"
    response=system["client"].post("/v1/ingest",json=payload,headers=system["headers"])
    assert response.status_code == 422
