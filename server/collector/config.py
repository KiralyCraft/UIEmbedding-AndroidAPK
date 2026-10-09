from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///collector-development.db"
    token_days: int = 90
    max_body_bytes: int = 2_000_000
    login_window_seconds: int = 900
    login_failure_limit: int = 10
    login_ip_burst_limit: int = 30
    login_ip_burst_seconds: int = 60
    login_ip_long_limit: int = 100
    login_ip_long_seconds: int = 900
    root_path: str = ""
    secure_cookies: bool = True
    apk_release_dir: str = ""

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            database_url=os.environ.get("DATABASE_URL", "sqlite:///collector-development.db"),
            token_days=int(os.environ.get("TOKEN_DAYS", "90")),
            root_path=os.environ.get("ROOT_PATH", "").rstrip("/"),
            apk_release_dir=os.environ.get("APK_RELEASE_DIR", ""),
        )
