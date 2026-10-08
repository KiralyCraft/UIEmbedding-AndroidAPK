import base64
import json
import struct
import uuid

import pytest
from fastapi.testclient import TestClient
from collector.api import create_app
from collector.config import Settings
from collector.db import Base, Encoder, User
from collector.security import issue_token, password_hash


@pytest.fixture
def system(tmp_path):
    app = create_app(Settings(database_url=f"sqlite:///{tmp_path / 'test.db'}", login_failure_limit=3))
    Base.metadata.create_all(app.state.engine)
    owner = str(uuid.uuid4())
    other = str(uuid.uuid4())
    model_id = "a" * 64
    with app.state.sessions.begin() as session:
        user = User(id=owner, username="alice", password_hash=password_hash("correct-password-123"), active=True)
        second = User(id=other, username="bob", password_hash=password_hash("correct-password-456"), active=True)
        session.add_all([user, second, Encoder(id=model_id, dimension=384, manifest_json="{}", active=True)])
        session.flush()
        token, _ = issue_token(session, user, 90)
        other_token, _ = issue_token(session, second, 90)
    with TestClient(app) as client:
        yield {"app": app, "client": client, "owner": owner, "other": other, "headers": {"Authorization": f"Bearer {token}"}, "other_headers": {"Authorization": f"Bearer {other_token}"}, "model_id": model_id, "tmp_path": tmp_path}
    app.state.engine.dispose()


@pytest.fixture
def payload(system):
    return {
        "run": {
            "id": str(uuid.uuid4()), "device_id": str(uuid.uuid4()), "capture_session_id": str(uuid.uuid4()),
            "model_id": system["model_id"], "package_name": "com.example.app", "device_model": "test-device",
            "android_sdk": 35, "start_wall_ms": 1790000000000, "start_elapsed_ns": 1000000000, "revision": 1,
        },
        "samples": [{
            "sequence": 0, "wall_ms": 1790000000001, "elapsed_ns": 1001000000, "image_timestamp_ns": 1000000000,
            "activity": "com.example.Main", "activity_source": "usage_stats", "window_class": "com.example.Main",
            "label_age_ms": 400, "source_width": 1080, "source_height": 2520, "rotation": 0, "target_fps": 5.0,
            "preprocess_ms": 10.0, "inference_ms": 20.0, "head_ms": 1.0, "thermal_status": 0,
            "embedding_b64": base64.b64encode(struct.pack("<384f", 1.0, *([0.0] * 383))).decode(),
        }],
    }
