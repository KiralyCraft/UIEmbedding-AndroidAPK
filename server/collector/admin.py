"""Small, same-origin administration UI with revocable server-side sessions."""
from __future__ import annotations

import json
import secrets
import uuid
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import Field
from sqlalchemy import Integer, delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from .config import Settings
from .db import Administrator, Benchmark, Device, Run, Sample, Token, User
from .schemas import Login, StrictModel
from .security import authenticate, digest, now_ms, password_hash

COOKIE = "uiembeddings_session"
STATIC = Path(__file__).with_name("static")


class CreateUser(StrictModel):
    username: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,128}$")
    password: str = Field(min_length=12, max_length=1024)


class UpdateUser(StrictModel):
    active: bool | None = None
    password: str | None = Field(default=None, min_length=12, max_length=1024)


class UpdateDevice(StrictModel):
    label: str = Field(max_length=128)
    active: bool


def install_admin(app: FastAPI, config: Settings, sessions: sessionmaker, login: Callable, login_guard: Callable) -> None:
    cookie_path = (config.root_path or "") + "/"

    @app.middleware("http")
    async def browser_headers(request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    def administrator(request: Request) -> str:
        token = request.cookies.get(COOKIE, "")
        with sessions() as session:
            user = authenticate(session, token)
            if user is None:
                raise HTTPException(401, "Sign in to continue")
            if session.get(Administrator, user.id) is None:
                raise HTTPException(403, "Administrator access required")
            if request.method not in ("GET", "HEAD") and not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""), digest("csrf:" + token)):
                raise HTTPException(403, "Reload the page and try again")
            return user.id

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/assets/{name}", include_in_schema=False)
    def asset(name: str) -> FileResponse:
        if name not in ("admin.js", "admin.css"):
            raise HTTPException(404)
        return FileResponse(STATIC / name)

    @app.post("/admin/login", dependencies=[Depends(login_guard)])
    def web_login(body: Login, request: Request, response: Response) -> dict:
        # JSON-only input and Fetch Metadata prevent cross-site login submission.
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, "Use this site's sign-in page")
        result = login(body, request)
        with sessions.begin() as session:
            token = session.get(Token, digest(result["token"]))
            allowed = session.get(Administrator, result["user_id"]) is not None
            if not allowed:
                session.delete(token)
            else:
                token.expires_ms = now_ms() + 8 * 3600 * 1000
        if not allowed:
            raise HTTPException(403, "Administrator access required")
        response.set_cookie(COOKIE, result["token"], max_age=8 * 3600, httponly=True, secure=config.secure_cookies, samesite="strict", path=cookie_path)
        return {"username": result["username"], "csrf": digest("csrf:" + result["token"])}

    @app.get("/admin/session")
    def web_session(request: Request, owner: str = Depends(administrator)) -> dict:
        with sessions() as session:
            return {"username": session.get(User, owner).username, "csrf": digest("csrf:" + request.cookies[COOKIE])}

    @app.post("/admin/logout")
    def web_logout(request: Request, response: Response, owner: str = Depends(administrator)) -> dict:
        with sessions.begin() as session:
            session.execute(delete(Token).where(Token.token_hash == digest(request.cookies[COOKIE]), Token.user_id == owner))
        response.delete_cookie(COOKIE, path=cookie_path, secure=config.secure_cookies, httponly=True, samesite="strict")
        return {"revoked": True}

    @app.get("/admin/users")
    def users(owner: str = Depends(administrator)) -> list[dict]:
        with sessions() as session:
            aggregate = {row.user_id: row for row in session.execute(select(Run.user_id, func.count().label("runs"), func.sum(Run.received_samples).label("samples"), func.max(Run.updated_ms).label("last_upload_ms")).group_by(Run.user_id))}
            counts = dict(session.execute(select(Device.user_id, func.count()).group_by(Device.user_id)).all())
            admins = set(session.scalars(select(Administrator.user_id)))
            return [{"id": u.id, "username": u.username, "active": u.active, "administrator": u.id in admins, "self": u.id == owner, "devices": counts.get(u.id, 0), "runs": aggregate[u.id].runs if u.id in aggregate else 0, "samples": int(aggregate[u.id].samples) if u.id in aggregate else 0, "last_upload_ms": aggregate[u.id].last_upload_ms if u.id in aggregate else None} for u in session.scalars(select(User).order_by(User.username))]

    @app.post("/admin/users", status_code=201)
    def create_user(body: CreateUser, owner: str = Depends(administrator)) -> dict:
        with sessions.begin() as session:
            username = body.username.lower()
            if session.scalar(select(User.id).where(User.username == username)):
                raise HTTPException(409, "Username already exists")
            user = User(id=str(uuid.uuid4()), username=username, password_hash=password_hash(body.password), active=True)
            session.add(user)
            try:
                session.flush()
            except IntegrityError as exc:
                raise HTTPException(409, "Username already exists") from exc
            return {"id": user.id, "username": username}

    @app.patch("/admin/users/{user_id}")
    def update_user(user_id: uuid.UUID, body: UpdateUser, owner: str = Depends(administrator)) -> dict:
        with sessions.begin() as session:
            user = session.get(User, str(user_id))
            if user is None:
                raise HTTPException(404, "User not found")
            if body.active is False and session.get(Administrator, user.id) is not None:
                raise HTTPException(409, "Administrator accounts cannot be disabled through this page")
            if body.active is not None:
                user.active = body.active
            if body.password is not None:
                user.password_hash = password_hash(body.password)
            if body.active is False or body.password is not None:
                session.execute(delete(Token).where(Token.user_id == user.id))
            return {"updated": True}

    @app.post("/admin/users/{user_id}/revoke")
    def revoke(user_id: uuid.UUID, owner: str = Depends(administrator)) -> dict:
        with sessions.begin() as session:
            if session.get(User, str(user_id)) is None:
                raise HTTPException(404, "User not found")
            session.execute(delete(Token).where(Token.user_id == str(user_id)))
        return {"revoked": True}

    @app.get("/admin/users/{user_id}/devices")
    def devices(user_id: uuid.UUID, owner: str = Depends(administrator)) -> list[dict]:
        with sessions() as session:
            if session.get(User, str(user_id)) is None:
                raise HTTPException(404, "User not found")
            rows = []
            for device in session.scalars(select(Device).where(Device.user_id == str(user_id)).order_by(Device.first_seen_ms)):
                stats = session.execute(select(func.count(), func.coalesce(func.sum(Run.received_samples), 0), func.count(func.distinct(Run.package_name)), func.coalesce(func.sum(Run.complete.cast(Integer)), 0)).where(Run.user_id == str(user_id), Run.device_id == device.id)).one()
                benchmark = session.scalar(select(Benchmark).where(Benchmark.user_id == str(user_id), Benchmark.device_id == device.id).order_by(Benchmark.received_ms.desc()).limit(1))
                latest = session.scalar(select(Run).where(Run.user_id == str(user_id), Run.device_id == device.id).order_by(Run.updated_ms.desc()).limit(1))
                sample = session.scalar(select(Sample).where(Sample.run_id == latest.id).order_by(Sample.sequence.desc()).limit(1)) if latest else None
                metrics = json.loads(sample.metadata_json) if sample else {}
                rows.append({"id": device.id, "label": device.label, "model": device.model, "android_sdk": device.android_sdk, "active": device.active, "first_seen_ms": device.first_seen_ms, "last_seen_ms": device.last_seen_ms, "runs": stats[0], "samples": int(stats[1]), "packages": stats[2], "complete_runs": int(stats[3]), "calibration": json.loads(benchmark.report_json) if benchmark else None, "latest_metrics": {k: metrics.get(k) for k in ("preprocess_ms", "inference_ms", "head_ms", "target_fps", "processing_backend", "preprocessing_backend", "capture_source", "sampling_mode", "thermal_status")}})
            return rows

    @app.patch("/admin/users/{user_id}/devices/{device_id}")
    def update_device(user_id: uuid.UUID, device_id: uuid.UUID, body: UpdateDevice, owner: str = Depends(administrator)) -> dict:
        with sessions.begin() as session:
            device = session.get(Device, (str(user_id), str(device_id)))
            if device is None:
                raise HTTPException(404, "Device not found")
            device.label = body.label.strip()
            device.active = body.active
        return {"updated": True}
