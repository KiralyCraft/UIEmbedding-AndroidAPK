from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///collector-development.db"
    token_days: int = 90
    max_body_bytes: int = 2_000_000
    login_window_seconds: int = 900
    login_failure_limit: int = 10

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_url=os.environ.get("DATABASE_URL", "sqlite:///collector-development.db"),
            token_days=int(os.environ.get("TOKEN_DAYS", "90")),
        )
