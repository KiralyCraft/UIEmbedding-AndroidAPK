from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from PIL import Image


class ImageMetadataReader:
    def read_size(self, path: str | Path) -> tuple[int, int]:
        image_path = Path(path)
        with image_path.open("rb") as handle:
            header = handle.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
        with Image.open(image_path) as image:
            return int(image.width), int(image.height)

    def sha256(self, path: str | Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if chunk == b"":
                    break
                digest.update(chunk)
        return digest.hexdigest()
