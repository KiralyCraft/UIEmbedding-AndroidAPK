"""Atomic, database-backed login budgets shared by API/web and every worker."""
from __future__ import annotations

import ipaddress
import math

from fastapi import HTTPException, Request
from sqlalchemy import delete
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import sessionmaker

from .config import Settings
from .db import LoginRateLimit
from .security import digest, now_ms


def client_key(request: Request) -> str:
    # Uvicorn resolves trusted proxy chains. Never consume a raw forwarding header here.
    host = request.client.host if request.client else "unknown"
    try:
        address = ipaddress.ip_address(host)
        host = str(address.ipv4_mapped or address) if isinstance(address, ipaddress.IPv6Address) else str(address)
    except ValueError:
        pass  # Non-IP ASGI transports share their transport identifier.
    return host


def enforce_login_limit(request: Request, config: Settings, sessions: sessionmaker) -> None:
    host = client_key(request)
    budgets = sorted([
        (digest("login-ip-minute:" + host), config.login_ip_burst_limit, config.login_ip_burst_seconds),
        (digest("login-ip-long:" + host), config.login_ip_long_limit, config.login_ip_long_seconds),
    ])
    for attempt in range(4):
        try:
            retry_ms = 0
            current = now_ms()
            with sessions.begin() as session:
                session.execute(delete(LoginRateLimit).where(LoginRateLimit.expires_ms <= current))
                dialect = session.bind.dialect.name
                for key, limit, seconds in budgets:
                    values = dict(key=key, attempts=0, expires_ms=current + seconds * 1000)
                    if dialect == "sqlite":
                        statement = sqlite_insert(LoginRateLimit).values(**values).on_conflict_do_nothing(index_elements=["key"])
                    elif dialect in ("mysql", "mariadb"):
                        statement = mysql_insert(LoginRateLimit).values(**values).on_duplicate_key_update(key=key)
                    else:
                        raise RuntimeError("Login rate limits require SQLite or MySQL/MariaDB")
                    # The upsert and row lock serialize admission across processes.
                    session.execute(statement)
                    row = session.get(LoginRateLimit, key, with_for_update=True)
                    if row.attempts >= limit:
                        retry_ms = max(retry_ms, row.expires_ms - current)
                    else:
                        row.attempts += 1
                    session.flush()
            # Reject only after commit, so rejected credentials still consume the budget.
            if retry_ms:
                raise HTTPException(429, "Too many sign-in attempts from this connection; retry later", headers={"Retry-After": str(max(1, math.ceil(retry_ms / 1000)))})
            return
        except (IntegrityError, OperationalError) as exc:
            if attempt == 3:
                raise HTTPException(503, "Sign-in temporarily unavailable; retry later", headers={"Retry-After": "5"}) from exc
