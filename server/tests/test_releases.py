import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import delete

from collector.api import create_app
from collector.config import Settings
from collector.db import Administrator, Token, User
from collector.releases import load_release


PREFIX = "/projects/uiembeddings"


def bundle(root: Path, name: str, version: str, payload: bytes, notes: list[str]) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "collector.apk").write_bytes(payload)
    manifest = dict(package_name="ro.ubb.uicollector", version_name=version, version_code=4,
                    minimum_sdk=30, abis=["arm64-v8a"], size_bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(), published_at="2026-10-09T05:00:00Z", changelog=notes)
    (directory / "release.json").write_text(json.dumps(manifest))
    return directory


@pytest.fixture
def release_system(system: dict):
    root = system["tmp_path"] / "releases"
    directory = bundle(root, "one", "1.1.2", b"APK fixture content", ["Keyboard capture."])
    current = root / "current"
    current.symlink_to(directory.name, target_is_directory=True)
    app = create_app(Settings(database_url=system["app"].state.engine.url.render_as_string(hide_password=False),
                              root_path=PREFIX, apk_release_dir=str(current)))
    with TestClient(app, base_url="https://testserver") as client:
        yield {**system, "app": app, "client": client, "root": root, "current": current}
    app.state.engine.dispose()


def sign_in(system: dict) -> TestClient:
    client = system["client"]
    response = client.post(PREFIX + "/admin/login", json={"username": "bob", "password": "correct-password-456"})
    assert response.status_code == 200
    client.headers["X-CSRF-Token"] = response.json()["csrf"]
    return client


def test_release_and_download_require_web_account(release_system: dict) -> None:
    client = release_system["client"]
    for path in ("android-release", "android-apk"):
        assert client.get(PREFIX + "/account/" + path).status_code == 401
        assert client.get(PREFIX + "/account/" + path, headers=release_system["other_headers"]).status_code == 401
    assert client.head(PREFIX + "/account/android-apk").status_code == 401
    assert client.get(PREFIX + "/account/android-apk", headers={"Range": "bytes=0-9"}).status_code == 401
    for path in ("assets/collector.apk", "downloads/collector.apk", "android-release/current/collector.apk"):
        assert client.get(PREFIX + "/" + path).status_code == 404


@pytest.mark.parametrize("administrator", [False, True])
def test_participant_and_admin_can_download_matching_release(release_system: dict, administrator: bool) -> None:
    if administrator:
        with release_system["app"].state.sessions.begin() as session:
            session.add(Administrator(user_id=release_system["other"]))
    client = sign_in(release_system)
    manifest = client.get(PREFIX + "/account/android-release").json()
    assert manifest["version_name"] == "1.1.2"
    assert manifest["changelog"] == ["Keyboard capture."]
    assert manifest["minimum_sdk"] == 30 and manifest["abis"] == ["arm64-v8a"]
    download = PREFIX + "/" + manifest["download_path"]
    response = client.get(download)
    assert response.status_code == 200
    assert response.content == b"APK fixture content"
    assert hashlib.sha256(response.content).hexdigest() == manifest["sha256"]
    assert int(response.headers["content-length"]) == manifest["size_bytes"]
    assert response.headers["content-type"] == "application/vnd.android.package-archive"
    assert response.headers["content-disposition"] == 'attachment; filename="UIEmbedding-Collector-1.1.2.apk"'
    assert response.headers["cache-control"] == "no-store"
    head = client.head(download)
    assert head.status_code == 200 and head.content == b""
    assert head.headers["content-length"] == response.headers["content-length"]
    partial = client.get(download, headers={"Range": "bytes=4-10"})
    assert partial.status_code == 206 and partial.content == response.content[4:11]


@pytest.mark.parametrize("revoke", [False, True])
def test_disabled_or_revoked_account_cannot_download(release_system: dict, revoke: bool) -> None:
    client = sign_in(release_system)
    with release_system["app"].state.sessions.begin() as session:
        if revoke:
            session.execute(delete(Token).where(Token.user_id == release_system["other"]))
        else:
            session.get(User, release_system["other"]).active = False
    assert client.get(PREFIX + "/account/android-release").status_code == 401
    assert client.get(PREFIX + "/account/android-apk").status_code == 401


def test_publication_replaces_apk_version_and_changelog_together(release_system: dict) -> None:
    client = sign_in(release_system)
    old = client.get(PREFIX + "/account/android-release").json()
    snapshot = load_release(str(release_system["current"]))
    directory = bundle(release_system["root"], "two", "1.1.3", b"replacement APK", ["Only the new changes."])
    temporary = release_system["root"] / ".next"
    temporary.symlink_to(directory.name, target_is_directory=True)
    temporary.replace(release_system["current"])
    new = client.get(PREFIX + "/account/android-release").json()
    assert new["version_name"] == "1.1.3" and new["changelog"] == ["Only the new changes."]
    assert client.get(PREFIX + "/" + old["download_path"]).status_code == 409
    assert client.get(PREFIX + "/" + new["download_path"]).content == b"replacement APK"
    assert snapshot.apk_path.read_bytes() == b"APK fixture content"


@pytest.mark.parametrize("failure", ["missing", "truncated", "invalid_metadata"])
def test_incomplete_release_is_unavailable_without_path_leak(release_system: dict, failure: str) -> None:
    client = sign_in(release_system)
    directory = release_system["current"].resolve()
    if failure == "missing":
        release_system["current"].unlink()
    elif failure == "truncated":
        (directory / "collector.apk").write_bytes(b"short")
    else:
        (directory / "release.json").write_text("{}")
    for path in ("android-release", "android-apk"):
        response = client.get(PREFIX + "/account/" + path)
        assert response.status_code == 503 and response.headers["retry-after"] == "60"
        assert str(release_system["root"]) not in response.text


def test_publisher_reads_actual_version_and_preserves_previous_release_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).resolve().parents[2] / "deploy" / "publish-apk.py"
    spec = importlib.util.spec_from_file_location("apk_publisher", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    apk, notes, root = tmp_path / "app.apk", tmp_path / "notes.json", tmp_path / "releases"
    apk.write_bytes(b"first APK")
    notes.write_text(json.dumps(["First changes."]))
    badging = "package: name='ro.ubb.uicollector' versionCode='4' versionName='1.1.2'\nminSdkVersion:'30'\nnative-code: 'arm64-v8a'\n"
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: badging)
    first = module.publish_apk(apk, root, notes, "aapt2")
    assert (root / "current").resolve() == first
    release = load_release(str(root / "current"))
    assert release.manifest.version_name == "1.1.2"
    assert release.manifest.sha256 == hashlib.sha256(apk.read_bytes()).hexdigest()
    apk.write_bytes(b"second APK")
    notes.write_text(json.dumps(["Replacement changes."]))
    badging = badging.replace("versionCode='4'", "versionCode='5'").replace("versionName='1.1.2'", "versionName='1.1.3'")
    second = module.publish_apk(apk, root, notes, "aapt2")
    release = load_release(str(root / "current"))
    assert (root / "current").resolve() == second and second != first
    assert release.manifest.version_name == "1.1.3" and release.manifest.changelog == ["Replacement changes."]
    assert release.apk_path.read_bytes() == b"second APK"
    badging = badging.replace("ro.ubb.uicollector", "org.wrong.app")
    with pytest.raises(ValueError, match="ro.ubb.uicollector"):
        module.publish_apk(apk, root, notes, "aapt2")
    assert (root / "current").resolve() == second
