"""Coordinate unit configuration helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def get_units_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return normalized unit settings from a loaded config."""

    eval_cfg = dict(cfg.get("eval_cfg", {}))
    loss_cfg = dict(cfg.get("loss", {}))
    units = dict(cfg.get("units", {}))
    return {
        "dataset_coord_unit": units.get("dataset_coord_unit", "m"),
        "coord_unit": units.get("coord_unit", units.get("target_coord_unit", eval_cfg.get("coord_unit", "dataset"))),
        "coord_scale": float(units.get("coord_scale", eval_cfg.get("coord_scale", loss_cfg.get("coord_scale", 1.0)))),
    }


def apply_units_to_loss_cfg(loss_cfg: dict[str, Any], units_cfg: dict[str, Any]) -> dict[str, Any]:
    """Inject the global coordinate scale into a loss config."""

    loss_cfg = deepcopy(dict(loss_cfg))
    loss_cfg["coord_scale"] = float(units_cfg.get("coord_scale", 1.0))
    return loss_cfg
