"""Python config loading helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


class ConfigDict(dict):
    """Dict with attribute-style access and recursive conversion."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.update(dict(*args, **kwargs))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._convert(value)

    def update(self, other: dict[str, Any], **kwargs: Any) -> None:  # type: ignore[override]
        for key, value in {**other, **kwargs}.items():
            self[key] = self._convert(value)

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            return ConfigDict(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._convert(item) for item in value)
        return value

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, ConfigDict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            if isinstance(value, tuple):
                return tuple(convert(v) for v in value)
            return value

        return {key: convert(value) for key, value in self.items()}


def load_config(path: str | Path) -> ConfigDict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = {
        key: value
        for key, value in vars(module).items()
        if not key.startswith("_") and key not in {"Path"}
    }
    cfg["_config_path"] = str(path)
    cfg["_config_name"] = path.stem
    return ConfigDict(cfg)

