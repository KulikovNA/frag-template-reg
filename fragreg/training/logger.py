"""Text, JSONL, and TensorBoard logging helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fragreg.utils.io import append_jsonl, ensure_dir


def setup_logger(work_dir: str | Path, name: str = "fragreg") -> logging.Logger:
    work_dir = ensure_dir(work_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(work_dir / "train.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class JSONLLogger:
    def __init__(self, path: str | Path, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)

    def log(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        append_jsonl(self.path, payload)


class TensorBoardLogger:
    """Optional TensorBoard scalar logger.

    The tensorboard package is imported lazily so training still works in
    environments where TensorBoard is not installed.
    """

    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        self.writer = None
        self.enabled = bool(enabled)
        self.log_dir = Path(log_dir)
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            self.enabled = False
            return
        ensure_dir(self.log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    @property
    def available(self) -> bool:
        return self.enabled and self.writer is not None

    def log_scalars(self, scalars: dict[str, float], step: int, prefix: str = "") -> None:
        if self.writer is None:
            return
        for key, value in scalars.items():
            if isinstance(value, (int, float)):
                tag = f"{prefix}/{key}" if prefix else key
                self.writer.add_scalar(tag, float(value), step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
