#!/usr/bin/env python3
"""Compare export-time preprocessing to the exact supplied deterministic transform."""
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "vendor"))
from ui_foundation.config import DataConfig, ResolutionBucketConfig
from ui_foundation.data.transforms import FullScreenTransform
from export_f6 import preprocess

config = DataConfig(color_jitter_strength=0, grayscale_probability=0, blur_probability=0, jpeg_probability=0)
transform = FullScreenTransform(config, strong=False)
rng = np.random.default_rng(6002)
shapes = [(800, 384), (2520, 1080), (1080, 2520), (701, 333), (512, 256), (120, 50), (1000, 1000)]
errors = []
for h, w in shapes:
    image = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    expected = transform(image, ResolutionBucketConfig(384, 800)).tensor.numpy().transpose(1, 2, 0)[None]
    actual = preprocess(image)
    error = float(np.abs(expected - actual).max())
    assert error <= 1e-6, (h, w, error)
    errors.append({"height": h, "width": w, "maximum_absolute_error": error})
report = {"passed": True, "cases": errors, "scope": "Python export preprocessing versus original supplied transform with augmentation disabled; not Android OpenCV validation"}
(root / "reports/preprocessing-parity.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
