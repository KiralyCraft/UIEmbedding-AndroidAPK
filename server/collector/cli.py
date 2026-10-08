from __future__ import annotations

import argparse
import getpass
import json
import uuid
from pathlib import Path

import numpy as np
from sqlalchemy import delete, select
from .api import canonical
from .config import Settings
from .db import AccountCreation, Administrator, Base, Encoder, LoginRateLimit, Run, Sample, Token, User, database
from .security import password_hash


def export_runs(session, owner: str, destination: Path, package: str | None = None) -> int:
    query = select(Run).where(Run.user_id == owner, Run.complete.is_(True), Run.received_samples > 0).order_by(Run.package_name, Run.start_wall_ms)
    if package is not None:
        query = query.where(Run.package_name == package)
    count = 0
    for run in session.scalars(query):
        folder = destination / run.package_name / run.device_id / f"run_{run.run_number:06d}_{run.id}"
        if folder.exists():
            raise FileExistsError(f"Refusing to overwrite an existing export: {folder}")
        folder.mkdir(parents=True)
        vector_path = folder / "embeddings.npy"
        # No object arrays or pickle; consumers can memory-map arbitrarily long runs.
        vectors = np.lib.format.open_memmap(vector_path, mode="w+", dtype="<f4", shape=(run.received_samples, 384))
        with (folder / "samples.jsonl").open("w", encoding="utf-8") as labels:
            rows = session.scalars(select(Sample).where(Sample.run_id == run.id).order_by(Sample.sequence).execution_options(yield_per=1024))
            for expected, sample in enumerate(rows):
                if sample.sequence != expected:
                    raise RuntimeError("Complete-run invariant violated; export stopped")
                vectors[expected] = np.frombuffer(sample.embedding, dtype="<f4", count=384)
                labels.write(sample.metadata_json + "\n")
        vectors.flush()
        del vectors
        (folder / "run.json").write_text(json.dumps({"schema_version": 1, "run_number": run.run_number, "sample_count": run.received_samples, "model_id": run.model_id, "metadata": json.loads(run.metadata_json)}, indent=2) + "\n")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="UI Embedding Collector administration")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Create initial schema; do not use for future schema upgrades")
    commands.add_parser("upgrade-rate-limits", help="Add persistent shared login rate limits without changing existing records")
    commands.add_parser("upgrade-account-creations", help="Add account creator history without inventing provenance for existing users")
    add = commands.add_parser("create-user")
    add.add_argument("username")
    add.add_argument("--admin", action="store_true")
    reset = commands.add_parser("reset-password")
    reset.add_argument("username")
    revoke = commands.add_parser("revoke-sessions")
    revoke.add_argument("username")
    register = commands.add_parser("register-model")
    register.add_argument("manifest", type=Path)
    export = commands.add_parser("export")
    export.add_argument("--username", required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--package")
    args = parser.parse_args()
    engine, sessions = database(Settings.from_environment().database_url)
    if args.command == "upgrade-account-creations":
        AccountCreation.__table__.create(engine, checkfirst=True)
        print("Account creation history ready")
        return
    if args.command == "upgrade-rate-limits":
        LoginRateLimit.__table__.create(engine, checkfirst=True)
        print("Login rate-limit table ready")
        return
    if args.command == "init-db":
        Base.metadata.create_all(engine)
        print("Schema v1 created")
        return
    with sessions.begin() as session:
        if args.command == "register-model":
            manifest = json.loads(args.manifest.read_text())
            if manifest["embedding_dim"] != 384 or manifest["export_parity_passed"] is not True:
                raise ValueError("Register only a parity-validated F6 deployment manifest")
            model_id = manifest["model_id"]
            if len(model_id) != 64 or any(c not in "0123456789abcdef" for c in model_id):
                raise ValueError("Invalid model_id")
            old = session.get(Encoder, model_id)
            text = canonical(manifest)
            if old is not None and old.manifest_json != text:
                raise ValueError("Different manifest already registered for this model ID")
            if old is None:
                session.add(Encoder(id=model_id, dimension=384, manifest_json=text, active=True))
            print(model_id)
            return
        username = args.username.strip().lower()
        user = session.scalar(select(User).where(User.username == username))
        if args.command == "create-user":
            if user is not None:
                raise ValueError("Username already exists")
            password = getpass.getpass("Password (12+ characters): ")
            if password != getpass.getpass("Repeat password: "):
                raise ValueError("Passwords differ")
            user = User(id=str(uuid.uuid4()), username=username, password_hash=password_hash(password), active=True)
            session.add(user)
            session.flush()
            if args.admin:
                session.add(Administrator(user_id=user.id))
            print(f"Created {username}")
        elif args.command in ("reset-password", "revoke-sessions"):
            if user is None:
                raise ValueError("User not found")
            if args.command == "reset-password":
                user.password_hash = password_hash(getpass.getpass("New password: "))
            session.execute(delete(Token).where(Token.user_id == user.id))
            print("All user sessions revoked")
        elif args.command == "export":
            if user is None:
                raise ValueError("User not found")
            print(f"Exported {export_runs(session, user.id, args.output, args.package)} complete runs")
    engine.dispose()


if __name__ == "__main__":
    main()
