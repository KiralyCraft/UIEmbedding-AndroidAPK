from __future__ import annotations

import base64
import hashlib
import json
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError, OperationalError

from .config import Settings
from .db import Application, Benchmark, Encoder, LoginFailure, Run, Sample, Token, User, database
from .schemas import BenchmarkReport, Ingest, Login
from .security import authenticate, digest, issue_token, now_ms, password_hash, verify_password


def canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


class BodyLimit:
    """Bound request memory, including chunked requests, before JSON parsing."""
    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.limit:
                await JSONResponse({"detail": "Request too large"}, status_code=413)(scope, receive, send)
                return
            if message.get("more_body", False) is False:
                break
        delivered = False
        async def replacement():
            nonlocal delivered
            if delivered is False:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, replacement, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings.from_environment()
    engine, sessions = database(config.database_url)
    app = FastAPI(title="UI Embedding Collector", version="1.0.0")
    app.state.engine = engine
    app.state.sessions = sessions
    app.add_middleware(BodyLimit, limit=config.max_body_bytes)
    dummy_hash = password_hash("a-dummy-password-never-used-to-login")

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        # Never echo passwords, embeddings, or arbitrary rejected fields in API errors.
        errors = [{"loc": e["loc"], "type": e["type"], "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    def current_user(authorization: Annotated[str | None, Header()] = None) -> str:
        if authorization is None or authorization.startswith("Bearer ") is False:
            raise HTTPException(401, "Bearer authentication required")
        with sessions() as session:
            user = authenticate(session, authorization[7:])
            if user is None:
                raise HTTPException(401, "Session expired or revoked; sign in again")
            return user.id

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.post("/v1/auth/login")
    def login(body: Login, request: Request):
        username = body.username.strip().lower()
        key = digest(username)
        cutoff = now_ms() - config.login_window_seconds * 1000
        with sessions.begin() as session:
            session.execute(delete(LoginFailure).where(LoginFailure.time_ms < cutoff))
            failures = session.scalar(select(func.count()).select_from(LoginFailure).where(LoginFailure.key == key))
            if failures >= config.login_failure_limit:
                raise HTTPException(429, "Too many failed sign-ins; retry later", headers={"Retry-After": str(config.login_window_seconds)})
            user = session.scalar(select(User).where(User.username == username))
            valid = verify_password(user.password_hash if user is not None else dummy_hash, body.password)
            if user is None or valid is False or user.active is False:
                session.add(LoginFailure(key=key, time_ms=now_ms()))
                rejected = True
            else:
                session.execute(delete(LoginFailure).where(LoginFailure.key == key))
                token, expiry = issue_token(session, user, config.token_days)
                response = {"token": token, "user_id": user.id, "username": user.username, "expires_ms": expiry}
                rejected = False
        if rejected:
            raise HTTPException(401, "Invalid credentials")
        return response

    @app.post("/v1/auth/logout")
    def logout(authorization: Annotated[str, Header()], owner: str = Depends(current_user)):
        with sessions.begin() as session:
            session.execute(delete(Token).where(Token.token_hash == digest(authorization[7:]), Token.user_id == owner))
        return {"revoked": True}

    @app.get("/v1/models")
    def models(owner: str = Depends(current_user)):
        with sessions() as session:
            return [{"model_id": e.id, "dimension": e.dimension} for e in session.scalars(select(Encoder).where(Encoder.active.is_(True)))]

    @app.post("/v1/ingest")
    def ingest(body: Ingest, owner: str = Depends(current_user)):
        # Transactions serialize per run and allocate display ordinals under a row lock.
        # Retry a race creating the same application/run, or a MySQL deadlock.
        for attempt in range(4):
            try:
                with sessions.begin() as session:
                    result = store_batch(session, body, owner)
                return result  # Acknowledgment is emitted only AFTER COMMIT succeeds.
            except IntegrityError:
                if attempt == 3:
                    raise HTTPException(503, "Concurrent write conflict; retry unchanged request")
            except OperationalError as exc:
                if attempt == 3:
                    raise HTTPException(503, "Database temporarily unavailable") from exc
        raise HTTPException(503, "Retry ingestion")

    @app.post("/v1/benchmarks")
    def benchmarks(body: BenchmarkReport, owner: str = Depends(current_user)):
        document = body.model_dump(mode="json")
        # Preserve the pre-adaptive canonical payload for retries of old reports.
        for result, row in zip(body.results, document["results"]):
            for field in ("attempts", "rate_limited", "confirmation"):
                if field not in result.model_fields_set:
                    row.pop(field, None)
        text = canonical(document)
        signature = digest(text)
        with sessions.begin() as session:
            encoder = session.get(Encoder, body.model_id)
            if encoder is None or encoder.active is False:
                raise HTTPException(409, "Model must be registered by the server administrator")
            old = session.get(Benchmark, str(body.id))
            if old is not None:
                if old.user_id != owner or old.payload_sha256 != signature:
                    raise HTTPException(409, "Benchmark ID collision")
            else:
                session.add(Benchmark(id=str(body.id), user_id=owner, device_id=str(body.device_id), model_id=body.model_id, payload_sha256=signature, report_json=text, received_ms=now_ms()))
        return {"id": str(body.id), "committed": True}

    @app.get("/v1/runs")
    def runs(owner: str = Depends(current_user), package: str | None = Query(default=None, max_length=255), complete_only: bool = True, offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=1000)):
        with sessions() as session:
            query = select(Run).where(Run.user_id == owner)
            if package is not None:
                query = query.where(Run.package_name == package)
            if complete_only:
                query = query.where(Run.complete.is_(True))
            found = session.scalars(query.order_by(Run.start_wall_ms, Run.id).offset(offset).limit(limit))
            return [{"run_id": r.id, "run_number": r.run_number, "package_name": r.package_name, "device_id": r.device_id, "model_id": r.model_id, "received_samples": r.received_samples, "complete": r.complete, "metadata": json.loads(r.metadata_json)} for r in found]

    @app.get("/v1/runs/{run_id}/samples")
    def samples(run_id: uuid.UUID, owner: str = Depends(current_user), after: int = Query(default=-1, ge=-1), limit: int = Query(default=1000, ge=1, le=5000)):
        with sessions() as session:
            run = session.get(Run, str(run_id))
            if run is None or run.user_id != owner:
                raise HTTPException(404, "Run not found")
            rows = session.scalars(select(Sample).where(Sample.run_id == str(run_id), Sample.sequence > after).order_by(Sample.sequence).limit(limit))
            return [{**json.loads(row.metadata_json), "embedding_b64": base64.b64encode(row.embedding).decode("ascii")} for row in rows]

    return app


def store_batch(session, body: Ingest, owner: str) -> dict:
    metadata = body.run.model_dump(mode="json")
    model = session.get(Encoder, body.run.model_id)
    if model is None or model.active is False or model.dimension != 384:
        raise HTTPException(409, "Register this 384-dimensional model before uploading")
    run = session.scalar(select(Run).where(Run.id == str(body.run.id)).with_for_update())
    if run is None:
        application = session.scalar(select(Application).where(Application.user_id == owner, Application.device_id == str(body.run.device_id), Application.package_name == body.run.package_name).with_for_update())
        if application is None:
            application = Application(id=str(uuid.uuid4()), user_id=owner, device_id=str(body.run.device_id), package_name=body.run.package_name, next_run_number=1)
            session.add(application)
            session.flush()
        ordinal = application.next_run_number
        application.next_run_number += 1
        run = Run(id=str(body.run.id), user_id=owner, application_id=application.id, device_id=str(body.run.device_id), capture_session_id=str(body.run.capture_session_id), model_id=body.run.model_id, package_name=body.run.package_name, run_number=ordinal, start_wall_ms=body.run.start_wall_ms, start_elapsed_ns=body.run.start_elapsed_ns, revision=body.run.revision, expected_samples=body.run.expected_samples, received_samples=0, complete=False, metadata_json=canonical(metadata), updated_ms=now_ms())
        session.add(run)
        session.flush()
    if run.user_id != owner:
        raise HTTPException(409, "Run ID is unavailable")
    old_metadata = json.loads(run.metadata_json)
    mutable = {"revision", "end_wall_ms", "end_elapsed_ns", "end_reason", "expected_samples"}
    if any(metadata[k] != old_metadata[k] for k in metadata.keys() - mutable):
        raise HTTPException(409, "Immutable run metadata changed")
    if body.run.revision == run.revision and canonical(metadata) != run.metadata_json:
        raise HTTPException(409, "Conflicting metadata revision")
    if body.run.revision > run.revision:
        if run.expected_samples is not None:
            raise HTTPException(409, "Closed run cannot be revised or reopened")
        run.metadata_json = canonical(metadata)
        run.revision = body.run.revision
        run.expected_samples = body.run.expected_samples
    effective = json.loads(run.metadata_json)
    for sample in sorted(body.samples, key=lambda x: x.sequence):
        if run.expected_samples is not None and sample.sequence >= run.expected_samples:
            raise HTTPException(409, "Sample exceeds closed run length")
        if effective["end_elapsed_ns"] is not None and sample.elapsed_ns > effective["end_elapsed_ns"]:
            raise HTTPException(409, "Sample follows closed run")
        payload = sample.model_dump(mode="json")
        raw = base64.b64decode(payload.pop("embedding_b64"), validate=True)
        metadata_text = canonical(payload)
        signature = hashlib.sha256(metadata_text.encode() + raw).hexdigest()
        previous = session.get(Sample, (run.id, sample.sequence))
        if previous is not None:
            if previous.payload_sha256 != signature:
                raise HTTPException(409, "Conflicting sample at existing sequence")
            continue
        before = session.scalar(select(Sample).where(Sample.run_id == run.id, Sample.sequence < sample.sequence).order_by(Sample.sequence.desc()).limit(1))
        after = session.scalar(select(Sample).where(Sample.run_id == run.id, Sample.sequence > sample.sequence).order_by(Sample.sequence).limit(1))
        if (before is not None and before.elapsed_ns >= sample.elapsed_ns) or (after is not None and after.elapsed_ns <= sample.elapsed_ns):
            raise HTTPException(409, "Sequence/time ordering conflict")
        session.add(Sample(run_id=run.id, sequence=sample.sequence, elapsed_ns=sample.elapsed_ns, wall_ms=sample.wall_ms, activity=sample.activity, embedding=raw, payload_sha256=signature, metadata_json=metadata_text))
        run.received_samples += 1
        session.flush()
    if run.expected_samples is not None:
        last = session.scalar(select(func.max(Sample.sequence)).where(Sample.run_id == run.id))
        if last is not None and last >= run.expected_samples:
            raise HTTPException(409, "Closed length would discard existing samples")
        last_time = session.scalar(select(func.max(Sample.elapsed_ns)).where(Sample.run_id == run.id))
        if last_time is not None and last_time > effective["end_elapsed_ns"]:
            raise HTTPException(409, "Closed end time precedes an existing sample")
        # Indices are unique nonnegative integers and all < expected_samples.
        run.complete = run.received_samples == run.expected_samples
    run.updated_ms = now_ms()
    return {"run_id": run.id, "run_number": run.run_number, "committed_sequences": [s.sequence for s in body.samples], "committed_revision": run.revision, "complete": run.complete}
