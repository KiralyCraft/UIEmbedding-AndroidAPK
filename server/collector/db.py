from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, ForeignKey, Index, Integer, LargeBinary, String, Text, UniqueConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Token(Base):
    __tablename__ = "auth_tokens"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    expires_ms: Mapped[int] = mapped_column(BigInteger)


class Administrator(Base):
    __tablename__ = "administrators"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)


class Device(Base):
    __tablename__ = "devices"
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), primary_key=True)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    label: Mapped[str] = mapped_column(String(128), default="")
    model: Mapped[str] = mapped_column(String(128), default="")
    android_sdk: Mapped[int | None] = mapped_column(Integer, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    first_seen_ms: Mapped[int] = mapped_column(BigInteger)
    last_seen_ms: Mapped[int] = mapped_column(BigInteger)


class LoginFailure(Base):
    __tablename__ = "login_failures"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(64), index=True)
    time_ms: Mapped[int] = mapped_column(BigInteger, index=True)


class LoginRateLimit(Base):
    __tablename__ = "login_rate_limits"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    attempts: Mapped[int] = mapped_column(Integer)
    expires_ms: Mapped[int] = mapped_column(BigInteger, index=True)


class Encoder(Base):
    __tablename__ = "encoders"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dimension: Mapped[int] = mapped_column(Integer)
    manifest_json: Mapped[str] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Application(Base):
    __tablename__ = "applications"
    __table_args__ = (UniqueConstraint("user_id", "device_id", "package_name", name="uq_owner_device_package"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    device_id: Mapped[str] = mapped_column(String(36))
    package_name: Mapped[str] = mapped_column(String(255))
    next_run_number: Mapped[int] = mapped_column(Integer, default=1)


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (UniqueConstraint("application_id", "run_number", name="uq_application_run_number"), Index("ix_runs_owner_device_updated", "user_id", "device_id", "updated_ms"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    application_id: Mapped[str] = mapped_column(ForeignKey("applications.id"), index=True)
    device_id: Mapped[str] = mapped_column(String(36))
    capture_session_id: Mapped[str] = mapped_column(String(36))
    model_id: Mapped[str] = mapped_column(ForeignKey("encoders.id"))
    package_name: Mapped[str] = mapped_column(String(255), index=True)
    run_number: Mapped[int] = mapped_column(Integer)
    start_wall_ms: Mapped[int] = mapped_column(BigInteger, index=True)
    start_elapsed_ns: Mapped[int] = mapped_column(BigInteger)
    revision: Mapped[int] = mapped_column(Integer)
    expected_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    received_samples: Mapped[int] = mapped_column(Integer, default=0)
    complete: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    metadata_json: Mapped[str] = mapped_column(Text)
    updated_ms: Mapped[int] = mapped_column(BigInteger, index=True)


class Sample(Base):
    __tablename__ = "samples"
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    sequence: Mapped[int] = mapped_column(Integer, primary_key=True)
    elapsed_ns: Mapped[int] = mapped_column(BigInteger)
    wall_ms: Mapped[int] = mapped_column(BigInteger)
    activity: Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary(1536))
    payload_sha256: Mapped[str] = mapped_column(String(64))
    metadata_json: Mapped[str] = mapped_column(Text)


class Benchmark(Base):
    __tablename__ = "benchmarks"
    __table_args__ = (Index("ix_benchmarks_owner_device_received", "user_id", "device_id", "received_ms"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    device_id: Mapped[str] = mapped_column(String(36))
    model_id: Mapped[str] = mapped_column(ForeignKey("encoders.id"))
    payload_sha256: Mapped[str] = mapped_column(String(64))
    report_json: Mapped[str] = mapped_column(Text)
    received_ms: Mapped[int] = mapped_column(BigInteger)


def database(url: str):
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # READ COMMITTED avoids stale snapshot reads when an ingestion transaction retries.
        kwargs["isolation_level"] = "READ COMMITTED"
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def sqlite_options(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=10000")
    return engine, sessionmaker(engine, expire_on_commit=False)
