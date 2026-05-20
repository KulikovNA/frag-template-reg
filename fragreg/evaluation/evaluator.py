"""Model evaluator for pose recovery through Kabsch."""

from __future__ import annotations

from itertools import islice
from typing import Any

import torch
from tqdm import tqdm

from fragreg.evaluation.metrics import (
    add_axial_span_groups,
    axis_vector_error_deg,
    compute_batch_metrics,
    compute_profile_batch_metrics,
    summarize_metric_dicts,
    summarize_metric_groups,
)
from fragreg.geometry.axisymmetric_solver import estimate_axisymmetric_pose
from fragreg.geometry.kabsch import batch_kabsch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def _as_list(value: Any, count: int) -> list[Any]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif isinstance(value, tuple):
        value = list(value)
    elif not isinstance(value, list):
        value = [value] * count
    return list(value)


def _attach_batch_metadata(metrics: list[dict[str, Any]], batch: dict[str, Any]) -> None:
    if not metrics:
        return
    batch_size = int(batch["points_C"].shape[0])
    scene_ids = _as_list(batch.get("scene_id", ""), batch_size)
    frame_ids = _as_list(batch.get("frame_id", -1), batch_size)
    fragment_ids = _as_list(batch.get("fragment_id", -1), batch_size)
    for idx, item in enumerate(metrics):
        batch_idx = idx % batch_size
        item["scene_id"] = str(scene_ids[batch_idx])
        item["frame_id"] = int(frame_ids[batch_idx])
        item["fragment_id"] = int(fragment_ids[batch_idx])


class Evaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device | str,
        loss_fn: torch.nn.Module | None = None,
        axis: str = "z",
        max_batches: int | None = None,
        coord_scale: float = 1.0,
        coord_unit: str = "dataset",
        solver_cfg: dict[str, Any] | None = None,
        profile_eval_modes: list[str] | tuple[str, ...] | None = None,
        return_per_sample: bool = False,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.loss_fn = loss_fn
        self.axis = axis
        self.max_batches = max_batches
        self.coord_scale = float(coord_scale)
        self.coord_unit = coord_unit
        self.solver_cfg = dict(solver_cfg or {})
        self.profile_eval_modes = list(profile_eval_modes or ["pred_profile_auto_init"])
        self.return_per_sample = bool(return_per_sample)

    @torch.no_grad()
    def evaluate(self, desc: str = "eval") -> dict[str, Any]:
        self.model.eval()
        all_metrics: list[dict[str, Any]] = []
        loss_values: list[float] = []
        default_profile_mode = "pred_profile_auto_init"
        iterable = self.dataloader if self.max_batches is None else islice(self.dataloader, self.max_batches)
        total = len(self.dataloader) if self.max_batches is None else self.max_batches
        for batch in tqdm(iterable, total=total, desc=desc, leave=False):
            batch = move_batch_to_device(batch, self.device)
            outputs = self.model(batch["points_C"])
            pred_axis_errors: list[float] | None = None
            if "pred_axis_C" in outputs and "gt_axis_C" in batch:
                axis_errors = axis_vector_error_deg(outputs["pred_axis_C"], batch["gt_axis_C"])
                pred_axis_errors = [float(value.detach().cpu()) for value in axis_errors]
            pred_axial_stats_metrics: list[dict[str, float]] | None = None
            pred_axial_stats = outputs.get("pred_axial_stats")
            if isinstance(pred_axial_stats, dict):
                suffix = f"_{self.coord_unit}" if self.coord_unit and self.coord_unit != "dataset" else ""
                pred_axial_stats_metrics = []
                for batch_idx in range(batch["points_C"].shape[0]):
                    pred_axial_stats_metrics.append(
                        {
                            f"pred_axial_stats_mean{suffix}": float(
                                (pred_axial_stats["mean"][batch_idx] * self.coord_scale).detach().cpu()
                            ),
                            f"pred_axial_stats_std{suffix}": float(
                                (pred_axial_stats["std"][batch_idx] * self.coord_scale).detach().cpu()
                            ),
                            f"pred_axial_stats_span{suffix}": float(
                                (pred_axial_stats["span"][batch_idx] * self.coord_scale).detach().cpu()
                            ),
                        }
                    )

            if self.loss_fn is not None:
                loss_dict = self.loss_fn(outputs, batch)
                loss_values.append(float(loss_dict["loss"].detach().cpu()))

            if "pred_points_O" in outputs:
                pred_points_O = outputs["pred_points_O"]
                weights = batch["valid_mask"].to(dtype=pred_points_O.dtype)
                T_pred = batch_kabsch(pred_points_O, batch["points_C_orig"], weights=weights)
                batch_metrics = compute_batch_metrics(
                    pred_points_O=pred_points_O,
                    target_points_O=batch["points_O"],
                    points_C_orig=batch["points_C_orig"],
                    T_pred=T_pred,
                    T_gt=batch["T_C_from_O"],
                    valid_mask=batch["valid_mask"],
                    axis=self.axis,
                    coord_scale=self.coord_scale,
                )
                if pred_axis_errors is not None:
                    for item, axis_error in zip(batch_metrics, pred_axis_errors, strict=False):
                        item["pred_axis_error_deg"] = axis_error
                if pred_axial_stats_metrics is not None:
                    for item, stats in zip(batch_metrics, pred_axial_stats_metrics, strict=False):
                        item.update(stats)
            elif "pred_profile_O" in outputs:
                pred_profile_O = outputs["pred_profile_O"]
                weights = batch["valid_mask"].to(dtype=pred_profile_O.dtype)
                solver_kwargs = {
                    "num_iters": int(self.solver_cfg.get("num_iters", 200)),
                    "lr": float(self.solver_cfg.get("lr", 1e-2)),
                    "huber_beta": float(self.solver_cfg.get("huber_beta", 0.01)),
                }
                batch_metrics = []
                for mode in self.profile_eval_modes:
                    if mode == "gt_profile_auto_init":
                        solver_profile = batch["profile_O"]
                        init = None
                        init_axis_C = None
                    elif mode == "pred_profile_gt_init":
                        solver_profile = pred_profile_O
                        init = batch["T_C_from_O"]
                        init_axis_C = None
                    elif mode == "pred_profile_auto_init":
                        solver_profile = pred_profile_O
                        init = None
                        init_axis_C = (
                            outputs["pred_axis_C"]
                            if self.solver_cfg.get("use_pred_axis_init", False) and "pred_axis_C" in outputs
                            else None
                        )
                    else:
                        raise ValueError(
                            f"Unsupported profile eval mode {mode!r}; expected "
                            "'gt_profile_auto_init', 'pred_profile_gt_init' or 'pred_profile_auto_init'."
                        )
                    T_pred, _ = estimate_axisymmetric_pose(
                        batch["points_C_orig"],
                        solver_profile,
                        weights=weights,
                        axis=self.axis,
                        init=init,
                        init_axis_C=init_axis_C,
                        **solver_kwargs,
                    )
                    mode_metrics = compute_profile_batch_metrics(
                        pred_profile_O=pred_profile_O,
                        target_profile_O=batch["profile_O"],
                        points_C_orig=batch["points_C_orig"],
                        T_pred=T_pred,
                        T_gt=batch["T_C_from_O"],
                        valid_mask=batch["valid_mask"],
                        axis=self.axis,
                        coord_scale=self.coord_scale,
                        coord_unit=self.coord_unit,
                        residual_profile_O=solver_profile,
                    )
                    if pred_axis_errors is not None:
                        for item, axis_error in zip(mode_metrics, pred_axis_errors, strict=False):
                            item["pred_axis_error_deg"] = axis_error
                    if pred_axial_stats_metrics is not None:
                        for item, stats in zip(mode_metrics, pred_axial_stats_metrics, strict=False):
                            item.update(stats)
                    for item in mode_metrics:
                        item["eval_mode"] = mode
                    if mode == default_profile_mode:
                        batch_metrics.extend(mode_metrics)
                    else:
                        for item in mode_metrics:
                            prefixed = {"eval_mode": mode}
                            for key, value in item.items():
                                if key == "eval_mode":
                                    continue
                                prefixed[f"{mode}/{key}"] = value
                            batch_metrics.append(prefixed)
            else:
                raise KeyError("Model outputs must contain either 'pred_points_O' or 'pred_profile_O'.")
            if self.return_per_sample:
                _attach_batch_metadata(batch_metrics, batch)
            all_metrics.extend(batch_metrics)

        default_metrics = [
            item for item in all_metrics if item.get("eval_mode", default_profile_mode) == default_profile_mode
        ]
        if default_metrics and any("gt_axial_span_mm" in item for item in default_metrics):
            add_axial_span_groups(default_metrics)
        summary = summarize_metric_dicts(all_metrics)
        summary.update(summarize_metric_groups(default_metrics))
        if loss_values:
            summary["loss"] = float(torch.tensor(loss_values, dtype=torch.float64).mean().item())
        summary["coord_scale"] = self.coord_scale
        summary["coord_unit"] = self.coord_unit
        if self.return_per_sample:
            summary["per_sample"] = all_metrics
        return summary
