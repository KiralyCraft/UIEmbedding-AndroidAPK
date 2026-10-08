#!/usr/bin/env python3
"""Compare actual production Kotlin inference head against trained PyTorch head."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch
from export_head import head_bytes


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    compiler = shutil.which("kotlinc")
    if compiler is None:
        raise SystemExit("Install Kotlin CLI to run this independent JVM parity test")
    sys.path.insert(0, str(root / "vendor"))
    from ui_foundation.models.heads import MobileEmbeddingHead
    state = torch.load(root / "models/f6_weights.pt", map_location="cpu", weights_only=True)["student_model_state_dict"]
    head = MobileEmbeddingHead(960, 384).eval()
    head.load_state_dict({key.removeprefix("embedding_head."): value for key, value in state.items() if key.startswith("embedding_head.")}, strict=True)
    rng = np.random.default_rng(6001)
    features = np.concatenate([rng.normal(0, scale, (16, 960)).astype(np.float32) for scale in (.001, .1, 1, 10, 100)] + [np.zeros((1, 960), np.float32), np.ones((1, 960), np.float32)])
    with torch.inference_mode():
        expected = head(torch.from_numpy(features)).numpy().astype(np.float64)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "head.bin").write_bytes(head_bytes(state))
        features.astype("<f4").tofile(path / "features.bin")
        subprocess.run([compiler, str(root / "android/app/src/main/java/ro/ubb/uicollector/Core.kt"), str(root / "tools/CoreHarness.kt"), "-include-runtime", "-d", str(path / "head.jar")], check=True)
        subprocess.run(["java", "-jar", str(path / "head.jar"), str(path / "head.bin"), str(path / "features.bin"), str(path / "actual.bin")], check=True)
        actual = np.fromfile(path / "actual.bin", dtype="<f4").reshape(-1, 384).astype(np.float64)
    cosine = np.sum(expected * actual, axis=1) / (np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1))
    maximum = float(np.max(np.abs(expected - actual)))
    assert cosine.min() > .999999, cosine.min()
    assert maximum < 2e-5, maximum
    report = {"passed": True, "vectors": len(features), "minimum_cosine": float(cosine.min()), "maximum_absolute_error": maximum, "torch_version": torch.__version__, "scope": "Exact shipped Kotlin head using actual trained F6 tensors; not full backbone conversion or Android GPU validation"}
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/kotlin_head_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
