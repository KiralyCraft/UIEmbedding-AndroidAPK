#!/usr/bin/env python3
"""Extract the trained F6 global embedding head without importing TIMM/TensorFlow."""
from __future__ import annotations
import argparse
import struct
from pathlib import Path
import numpy as np
import torch


def head_bytes(state: dict[str, torch.Tensor]) -> bytes:
    prefix = "embedding_head.network."
    fields = [("0.weight", (384, 960)), ("0.bias", (384,)), ("1.weight", (384,)), ("1.bias", (384,)), ("4.weight", (384, 384))]
    result = bytearray(struct.pack("<4siif", b"UIH1", 960, 384, 1e-5))
    for name, shape in fields:
        value = state[prefix + name].detach().cpu().numpy()
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Unexpected or non-finite trained tensor: {prefix + name}")
        result.extend(value.astype("<f4").tobytes(order="C"))
    return bytes(result)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "models/f6_weights.pt")
    parser.add_argument("--output", type=Path, default=root / "android/app/src/main/assets/f6_head.bin")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(head_bytes(payload["student_model_state_dict"]))
    print(args.output)


if __name__ == "__main__":
    main()
