#!/usr/bin/env python3
"""Publish a verified APK and changelog together by atomically switching current."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid


@dataclass(frozen=True)
class ApkDetails:
    package_name: str
    version_name: str
    version_code: int
    minimum_sdk: int
    abis: list[str]


def inspect_apk(apk: Path, aapt: str) -> ApkDetails:
    output = subprocess.check_output([aapt, "dump", "badging", str(apk)], text=True)
    package = re.search(r"^package: name='([^']+)' versionCode='(\d+)' versionName='([^']+)'", output, re.MULTILINE)
    sdk = re.search(r"^(?:minSdkVersion|sdkVersion):'(\d+)'", output, re.MULTILINE)
    native = re.search(r"^native-code: (.+)$", output, re.MULTILINE)
    if package is None or sdk is None or native is None:
        raise ValueError("APK must have package, version, minimum SDK and native ABI metadata")
    name, code, version = package.groups()
    if name != "ro.ubb.uicollector" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", version) or int(code) < 1:
        raise ValueError("Expected the versioned ro.ubb.uicollector APK")
    abis = re.findall(r"'([^']+)'", native.group(1))
    if not abis:
        raise ValueError("APK has no native ABI")
    return ApkDetails(name, version, int(code), int(sdk.group(1)), abis)


def publish_apk(apk: Path, release_root: Path, changelog: Path, aapt: str) -> Path:
    notes = json.loads(changelog.read_text())
    if not isinstance(notes, list) or not 1 <= len(notes) <= 12 or any(not isinstance(n, str) or not 1 <= len(n.strip()) <= 300 for n in notes):
        raise ValueError("Changelog must be a JSON array of 1–12 short, nonempty notes")
    release_root.mkdir(parents=True, exist_ok=True)
    current = release_root / "current"
    if current.exists() and not current.is_symlink():
        raise ValueError("current must be a release symlink, not an existing directory")
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=release_root) as staging:
        bundle = Path(staging) / "release"
        bundle.mkdir(mode=0o755)
        copied = bundle / "collector.apk"
        shutil.copyfile(apk, copied)
        details = inspect_apk(copied, aapt)
        with copied.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = {
            **asdict(details), "sha256": checksum, "size_bytes": copied.stat().st_size,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "changelog": [n.strip() for n in notes],
        }
        metadata = bundle / "release.json"
        metadata.write_text(json.dumps(manifest, indent=2) + "\n")
        copied.chmod(0o644)
        metadata.chmod(0o644)
        target = release_root / (details.version_name + "-" + uuid.uuid4().hex)
        bundle.rename(target)
    temporary_link = release_root / (".current-" + uuid.uuid4().hex)
    try:
        temporary_link.symlink_to(target.name, target_is_directory=True)
        temporary_link.replace(current)
    finally:
        temporary_link.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--changelog", type=Path, required=True)
    parser.add_argument("--aapt", default="aapt2", help="Android SDK aapt2 executable")
    args = parser.parse_args()
    target = publish_apk(args.apk, args.release_root, args.changelog, args.aapt)
    print("Published " + str(target))


if __name__ == "__main__":
    main()
