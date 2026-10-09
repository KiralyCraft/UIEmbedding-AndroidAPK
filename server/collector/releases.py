"""Read one immutable APK release selected by the deployment's current symlink."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import Field

from .schemas import StrictModel


class ReleaseManifest(StrictModel):
    package_name: Literal["ro.ubb.uicollector"]
    version_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
    version_code: int = Field(ge=1)
    minimum_sdk: int = Field(ge=1)
    abis: list[Annotated[str, Field(min_length=1, max_length=32)]] = Field(min_length=1, max_length=8)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    published_at: datetime
    changelog: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(min_length=1, max_length=12)


@dataclass(frozen=True)
class ApkRelease:
    manifest: ReleaseManifest
    apk_path: Path


def load_release(directory: str) -> ApkRelease:
    """Resolve current once so metadata and downloads use the same release."""
    try:
        if not directory:
            raise ValueError("No release configured")
        bundle = Path(directory).resolve(strict=True)
        manifest = ReleaseManifest.model_validate_json((bundle / "release.json").read_bytes())
        apk = bundle / "collector.apk"
        if not apk.is_file() or apk.stat().st_size != manifest.size_bytes:
            raise ValueError("Incomplete release")
        return ApkRelease(manifest=manifest, apk_path=apk)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(503, "The Android app download is temporarily unavailable. Please try again later.", headers={"Retry-After": "60"}) from exc
