"""Small OpenMMLab-style registries used across the project."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable


class Registry:
    """Map string names to classes/functions and build them from config dicts."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._modules: dict[str, Callable[..., Any]] = {}

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def __repr__(self) -> str:
        return f"Registry(name={self.name}, items={list(self._modules)})"

    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._modules:
            raise KeyError(f"{key!r} is not registered in {self.name}.")
        return self._modules[key]

    def register_module(
        self,
        module: Callable[..., Any] | None = None,
        name: str | None = None,
    ) -> Callable[..., Any]:
        def _register(obj: Callable[..., Any]) -> Callable[..., Any]:
            module_name = name or obj.__name__
            if module_name in self._modules:
                raise KeyError(f"{module_name!r} is already registered in {self.name}.")
            self._modules[module_name] = obj
            return obj

        if module is not None:
            return _register(module)
        return _register

    def build(self, cfg: dict[str, Any], **default_kwargs: Any) -> Any:
        return build_from_cfg(cfg, self, **default_kwargs)


DATASETS = Registry("dataset")
MODELS = Registry("model")
LOSSES = Registry("loss")


def build_from_cfg(cfg: dict[str, Any], registry: Registry, **default_kwargs: Any) -> Any:
    if cfg is None:
        raise ValueError("cfg must be a dict, got None.")
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg)!r}.")
    if "type" not in cfg:
        raise KeyError(f"cfg for registry {registry.name} must contain key 'type'.")

    cfg = deepcopy(dict(cfg))
    obj_type = cfg.pop("type")
    kwargs = {**default_kwargs, **cfg}

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
    elif callable(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(f"type must be a string or callable, got {type(obj_type)!r}.")
    return obj_cls(**kwargs)


def build_dataset(cfg: dict[str, Any]) -> Any:
    from fragreg.data import datasets as _datasets  # noqa: F401

    return DATASETS.build(cfg)


def build_model(cfg: dict[str, Any]) -> Any:
    from fragreg import models as _models  # noqa: F401

    return MODELS.build(cfg)


def build_loss(cfg: dict[str, Any]) -> Any:
    from fragreg import losses as _losses  # noqa: F401

    return LOSSES.build(cfg)

