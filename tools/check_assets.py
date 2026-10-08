#!/usr/bin/env python3
"""Refuse to build/deploy missing, changed or unvalidated model assets."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
assets = root / "android/app/src/main/assets"
path = assets / "f6_manifest.json"
if not path.exists():
    raise SystemExit("No validated deployment manifest. Run ./00_export_model.sh first; no placeholder model is supplied.")
manifest = json.loads(path.read_text())
if manifest.get("export_parity_passed") is not True or manifest.get("embedding_dim") != 384:
    raise SystemExit("Manifest is not a parity-validated F6 export")
required = {"f6_backbone.tflite", "f6_head.bin", "golden_input.bin", "golden_embedding.bin"}
if manifest.get("deployment_schema") == 2:
    required.add("f6_encoder.tflite")
if set(manifest["asset_sha256"]) != required:
    raise SystemExit("Unexpected deployment asset set")
for name, expected in manifest["asset_sha256"].items():
    if hashlib.sha256((assets / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Changed asset: {name}")
identity = {key: manifest[key] for key in ("deployment_schema", "preprocessing", "asset_sha256")}
expected = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if expected != manifest["model_id"]:
    raise SystemExit("Deployment identity hash mismatch")
print(f"Deployment assets verified: {manifest['model_id']}")
