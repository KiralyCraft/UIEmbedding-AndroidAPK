from __future__ import annotations

import hashlib
import secrets
import time
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, InvalidHashError
from sqlalchemy import delete, func, select
from .db import LoginFailure, Token, User


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def password_hash(password: str) -> str:
    if len(password) < 12:
        raise ValueError("Use a password containing at least 12 characters")
    return PasswordHasher().hash(password)


def verify_password(stored: str, supplied: str) -> bool:
    try:
        return bool(PasswordHasher().verify(stored, supplied))
    except (VerificationError, InvalidHashError):
        return False


def issue_token(session, user: User, days: int) -> tuple[str, int]:
    value = secrets.token_urlsafe(48)
    expiry = now_ms() + days * 86_400_000
    session.add(Token(token_hash=digest(value), user_id=user.id, expires_ms=expiry))
    return value, expiry


def authenticate(session, token: str) -> User | None:
    if len(token) > 512:
        return None
    found = session.get(Token, digest(token))
    if found is None or found.expires_ms <= now_ms():
        return None
    user = session.get(User, found.user_id)
    return user if user is not None and user.active else None
