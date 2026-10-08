from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(output_dir: str, verbose: bool = False, rank: int = 0) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.FileHandler(directory / "training.log", encoding="utf-8")]
    if rank == 0:
        handlers.insert(0, logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
