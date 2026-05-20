#!/usr/bin/env python
"""Run profile-model inference and export visual diagnostics."""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fragreg.evaluation.metrics import axis_vector_error_deg, compute_profile_batch_metrics
from fragreg.geometry.axisymmetric_solver import estimate_axisymmetric_pose
from fragreg.geometry.transforms import apply_transform, invert_transform
from fragreg.registry import build_dataset, build_model
from fragreg.training import load_checkpoint
from fragreg.utils import get_units_cfg, load_config
from fragreg.utils.io import ensure_dir, write_json
from fragreg.visualization import (
    draw_profile_scatter_png,
    reconstruct_points_from_profile,
    write_axis_ply,
    write_colored_points_ply,
    write_error_points_ply,
    write_overlay_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/dgcnn_profile_global_axisym_y_one_scene.py",
        help="Path to config. Defaults to the one-scene overfit config.",
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to checkpoint. Defaults to <train_cfg.work_dir>/best.pth.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu.")
    parser.add_argument("--out-dir", default=None, help="Output directory for inference visualizations.")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--indices", default=None, help="Comma-separated dataset indices, e.g. 0,3,7.")
    parser.add_argument("--scene-id", default=None, help="Optional scene_id override.")
    parser.add_argument("--frame-ids", default=None, help="Optional comma-separated frame_id filter.")
    parser.add_argument("--fragment-ids", default=None, help="Optional comma-separated fragment_id filter.")
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--solver-iters", type=int, default=None)
    parser.add_argument("--max-edges", type=int, default=512, help="Max residual line segments in overlay PLY.")
    return parser.parse_args()


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def dataloader_cfg_for_split(cfg, split: str) -> dict:
    key = f"{split}_dataloader"
    if key in cfg:
        return copy.deepcopy(cfg[key])
    loader_cfg = copy.deepcopy(cfg.val_dataloader if "val_dataloader" in cfg else cfg.train_dataloader)
    loader_cfg["dataset"]["split"] = split
    loader_cfg["dataset"]["random_sample"] = False
    return loader_cfg


def resolve_checkpoint(cfg, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg is not None:
        return Path(checkpoint_arg)
    return Path(cfg.train_cfg.get("work_dir", f"work_dirs/{cfg._config_name}")) / "best.pth"


def resolve_out_dir(cfg, out_dir_arg: str | None) -> Path:
    if out_dir_arg is not None:
        return ensure_dir(out_dir_arg)
    work_dir = Path(cfg.train_cfg.get("work_dir", f"work_dirs/{cfg._config_name}"))
    return ensure_dir(work_dir / "inference_vis")


def selected_indices(dataset_len: int, args: argparse.Namespace) -> list[int]:
    if args.indices:
        indices = parse_int_list(args.indices) or []
        for index in indices:
            if index < 0 or index >= dataset_len:
                raise IndexError(f"Dataset index {index} is out of range for length {dataset_len}.")
        return indices
    return list(range(min(int(args.num_samples), dataset_len)))


def make_batch(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0).to(device)
        else:
            batch[key] = [value]
    return batch


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def sample_name(sample: dict[str, Any], index: int) -> str:
    return (
        f"idx_{index:04d}_"
        f"{sample['scene_id']}_"
        f"frame_{int(sample['frame_id']):06d}_"
        f"fragment_{int(sample['fragment_id']):04d}"
    )


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return value


def save_point_csv(path: Path, arrays: dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    headers: list[str] = []
    columns: list[np.ndarray] = []
    for name, values in arrays.items():
        values = np.asarray(values)
        if values.ndim == 1:
            headers.append(name)
            columns.append(values)
        elif values.ndim == 2:
            for dim in range(values.shape[1]):
                headers.append(f"{name}_{dim}")
                columns.append(values[:, dim])
        else:
            raise ValueError(f"Unsupported CSV array shape for {name}: {values.shape}")
    stacked = np.stack(columns, axis=1)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(stacked.tolist())


def add_prefixed_metrics(target: dict[str, Any], prefix: str, metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        target[f"{prefix}/{key}"] = value


def run_sample(
    index: int,
    sample: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    axis: str,
    units_cfg: dict[str, Any],
    solver_cfg: dict[str, Any],
    out_dir: Path,
    max_edges: int,
) -> dict[str, Any]:
    batch = make_batch(sample, device)
    with torch.no_grad():
        outputs = model(batch["points_C"])

    pred_profile = outputs["pred_profile_O"]
    target_profile = batch["profile_O"]
    points_C_orig = batch["points_C_orig"]
    points_O = batch["points_O"]
    valid_mask = batch["valid_mask"]
    T_gt = batch["T_C_from_O"]
    weights = valid_mask.to(dtype=pred_profile.dtype)
    solver_kwargs = {
        "num_iters": int(solver_cfg.get("num_iters", 200)),
        "lr": float(solver_cfg.get("lr", 1e-2)),
        "huber_beta": float(solver_cfg.get("huber_beta", 0.01)),
    }

    init_axis_C = (
        outputs["pred_axis_C"]
        if solver_cfg.get("use_pred_axis_init", False) and "pred_axis_C" in outputs
        else None
    )
    T_pred, solver_diag = estimate_axisymmetric_pose(
        points_C_orig,
        pred_profile,
        weights=weights,
        axis=axis,
        init_axis_C=init_axis_C,
        **solver_kwargs,
    )
    metrics = compute_profile_batch_metrics(
        pred_profile_O=pred_profile,
        target_profile_O=target_profile,
        points_C_orig=points_C_orig,
        T_pred=T_pred,
        T_gt=T_gt,
        valid_mask=valid_mask,
        axis=axis,
        coord_scale=units_cfg["coord_scale"],
        coord_unit=units_cfg["coord_unit"],
        residual_profile_O=pred_profile,
    )[0]
    if "pred_axis_C" in outputs:
        pred_axis_error = axis_vector_error_deg(outputs["pred_axis_C"], batch["gt_axis_C"])
        metrics["pred_axis_error_deg"] = float(pred_axis_error[0].detach().cpu())

    T_oracle, oracle_diag = estimate_axisymmetric_pose(
        points_C_orig,
        target_profile,
        weights=weights,
        axis=axis,
        **solver_kwargs,
    )
    oracle_metrics = compute_profile_batch_metrics(
        pred_profile_O=pred_profile,
        target_profile_O=target_profile,
        points_C_orig=points_C_orig,
        T_pred=T_oracle,
        T_gt=T_gt,
        valid_mask=valid_mask,
        axis=axis,
        coord_scale=units_cfg["coord_scale"],
        coord_unit=units_cfg["coord_unit"],
        residual_profile_O=target_profile,
    )[0]
    pred_gt_init_T, pred_gt_init_diag = estimate_axisymmetric_pose(
        points_C_orig,
        pred_profile,
        weights=weights,
        axis=axis,
        init=T_gt,
        **solver_kwargs,
    )
    pred_gt_init_metrics = compute_profile_batch_metrics(
        pred_profile_O=pred_profile,
        target_profile_O=target_profile,
        points_C_orig=points_C_orig,
        T_pred=pred_gt_init_T,
        T_gt=T_gt,
        valid_mask=valid_mask,
        axis=axis,
        coord_scale=units_cfg["coord_scale"],
        coord_unit=units_cfg["coord_unit"],
        residual_profile_O=pred_profile,
    )[0]

    valid = tensor_to_numpy(valid_mask[0]).astype(bool)
    points_C_np = tensor_to_numpy(points_C_orig[0])[valid]
    points_O_np = tensor_to_numpy(points_O[0])[valid]
    pred_profile_np = tensor_to_numpy(pred_profile[0])[valid]
    target_profile_np = tensor_to_numpy(target_profile[0])[valid]
    T_pred_np = tensor_to_numpy(T_pred[0])
    T_gt_np = tensor_to_numpy(T_gt[0])
    T_O_from_C = invert_transform(T_pred)[0]
    q_pose_O = tensor_to_numpy(apply_transform(points_C_orig, T_O_from_C.unsqueeze(0))[0])[valid]
    pred_points_O_lifted = reconstruct_points_from_profile(pred_profile_np, q_pose_O, axis=axis)
    pred_points_C_fit = tensor_to_numpy(
        apply_transform(
            torch.as_tensor(pred_points_O_lifted, dtype=points_C_orig.dtype, device=device).unsqueeze(0),
            T_pred,
        )[0]
    )

    name = sample_name(sample, index)
    sample_dir = ensure_dir(out_dir / name)
    coord_scale = float(units_cfg["coord_scale"])
    profile_error_mm = np.linalg.norm((pred_profile_np - target_profile_np) * coord_scale, axis=1)
    radial_error_mm = (pred_profile_np[:, 0] - target_profile_np[:, 0]) * coord_scale
    axial_error_mm = (pred_profile_np[:, 1] - target_profile_np[:, 1]) * coord_scale

    draw_profile_scatter_png(
        sample_dir / "profile_scatter.svg",
        target_profile=target_profile_np,
        pred_profile=pred_profile_np,
        coord_scale=coord_scale,
        metrics=metrics,
    )
    write_overlay_ply(
        sample_dir / "camera_observed_vs_pred_fit.ply",
        observed_points_C=points_C_np,
        fitted_points_C=pred_points_C_fit,
        max_edges=max_edges,
    )
    write_error_points_ply(sample_dir / "camera_profile_error.ply", points_C_np, profile_error_mm)
    object_points = np.concatenate([points_O_np, pred_points_O_lifted], axis=0)
    object_colors = np.concatenate(
        [
            np.repeat(np.array([[60, 210, 80]], dtype=np.uint8), points_O_np.shape[0], axis=0),
            np.repeat(np.array([[255, 120, 20]], dtype=np.uint8), pred_points_O_lifted.shape[0], axis=0),
        ],
        axis=0,
    )
    write_colored_points_ply(sample_dir / "object_gt_vs_pred_lifted.ply", object_points, object_colors)
    pred_axis_C_np = tensor_to_numpy(outputs["pred_axis_C"][0]) if "pred_axis_C" in outputs else None
    gt_axis_C_np = tensor_to_numpy(batch["gt_axis_C"][0])
    axis_length = max(float(np.linalg.norm(points_C_np - points_C_np.mean(axis=0), axis=1).max()) * 1.2, 0.02)
    write_axis_ply(
        sample_dir / "camera_axes_gt_green_pred_red.ply",
        center_C=points_C_np.mean(axis=0),
        gt_axis_C=gt_axis_C_np,
        pred_axis_C=pred_axis_C_np,
        length=axis_length,
    )
    save_point_csv(
        sample_dir / "points_profile.csv",
        {
            "points_C": points_C_np,
            "points_O_gt": points_O_np,
            "profile_gt": target_profile_np,
            "profile_pred": pred_profile_np,
            "profile_error_mm": profile_error_mm,
            "radial_error_mm": radial_error_mm,
            "axial_error_mm": axial_error_mm,
            "q_pose_O": q_pose_O,
            "pred_points_O_lifted": pred_points_O_lifted,
            "pred_points_C_fit": pred_points_C_fit,
        },
    )

    summary: dict[str, Any] = {
        "index": index,
        "scene_id": sample["scene_id"],
        "frame_id": int(sample["frame_id"]),
        "fragment_id": int(sample["fragment_id"]),
        "num_points": int(valid.sum()),
        "sample_dir": str(sample_dir),
        "metrics": metrics,
        "solver_diagnostics": solver_diag,
        "T_pred": T_pred_np,
        "T_gt": T_gt_np,
        "gt_axis_C": gt_axis_C_np,
        "pred_axis_C": pred_axis_C_np,
        "artifacts": {
            "profile_scatter": str(sample_dir / "profile_scatter.svg"),
            "camera_overlay_ply": str(sample_dir / "camera_observed_vs_pred_fit.ply"),
            "camera_profile_error_ply": str(sample_dir / "camera_profile_error.ply"),
            "object_overlay_ply": str(sample_dir / "object_gt_vs_pred_lifted.ply"),
            "axis_ply": str(sample_dir / "camera_axes_gt_green_pred_red.ply"),
            "point_csv": str(sample_dir / "points_profile.csv"),
        },
    }
    add_prefixed_metrics(summary["metrics"], "gt_profile_auto_init", oracle_metrics)
    add_prefixed_metrics(summary["metrics"], "pred_profile_gt_init", pred_gt_init_metrics)
    summary["oracle_solver_diagnostics"] = oracle_diag
    summary["pred_profile_gt_init_solver_diagnostics"] = pred_gt_init_diag
    write_json(sample_dir / "summary.json", jsonable(summary))
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    units_cfg = get_units_cfg(cfg)
    eval_cfg = cfg.get("eval_cfg", {})
    axis = eval_cfg.get("axis", cfg.loss.get("axis", "y"))
    solver_cfg = dict(eval_cfg.get("solver", {}))
    if args.solver_iters is not None:
        solver_cfg["num_iters"] = args.solver_iters

    checkpoint = resolve_checkpoint(cfg, args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    out_dir = resolve_out_dir(cfg, args.out_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    loader_cfg = dataloader_cfg_for_split(cfg, args.split)
    dataset_cfg = copy.deepcopy(loader_cfg["dataset"])
    dataset_cfg["return_profile"] = True
    dataset_cfg["random_sample"] = args.split == "train" and bool(dataset_cfg.get("random_sample", False))
    if args.split != "train":
        dataset_cfg["random_sample"] = False
    if args.scene_id is not None:
        dataset_cfg["scene_ids"] = [args.scene_id]
    frame_ids = parse_int_list(args.frame_ids)
    fragment_ids = parse_int_list(args.fragment_ids)
    if frame_ids is not None:
        dataset_cfg["frame_ids"] = frame_ids
    if fragment_ids is not None:
        dataset_cfg["fragment_ids"] = fragment_ids
    if args.num_points is not None:
        dataset_cfg["num_points"] = args.num_points
    dataset = build_dataset(dataset_cfg)
    indices = selected_indices(len(dataset), args)

    model = build_model(cfg.model).to(device)
    checkpoint_payload = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    print(f"config: {args.config}")
    print(f"checkpoint: {checkpoint}")
    print(f"checkpoint epoch: {checkpoint_payload.get('epoch', 'unknown')}")
    print(f"device: {device}")
    print(f"dataset samples: {len(dataset)}")
    if hasattr(dataset, "get_summary"):
        print(f"dataset summary: {dataset.get_summary()}")
    print(f"selected indices: {indices}")
    print(f"output: {out_dir}")

    summaries = []
    for index in indices:
        summary = run_sample(
            index=index,
            sample=dataset[index],
            model=model,
            device=device,
            axis=axis,
            units_cfg=units_cfg,
            solver_cfg=solver_cfg,
            out_dir=out_dir,
            max_edges=int(args.max_edges),
        )
        summaries.append(summary)
        metrics = summary["metrics"]
        print(
            f"[{summary['sample_dir']}] "
            f"profile_axial_l1_mm={metrics.get('profile_axial_l1_mm', float('nan')):.4g}, "
            f"axial_corr={metrics.get('axial_corr', float('nan')):.4g}, "
            f"axis_error_deg={metrics.get('axis_error_deg', float('nan')):.4g}, "
            f"pred_axis_error_deg={metrics.get('pred_axis_error_deg', float('nan')):.4g}, "
            f"translation_error_mm={metrics.get('translation_error_mm', float('nan')):.4g}"
        )

    manifest = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "split": args.split,
        "axis": axis,
        "coord_scale": units_cfg["coord_scale"],
        "coord_unit": units_cfg["coord_unit"],
        "dataset_summary": dataset.get_summary() if hasattr(dataset, "get_summary") else {},
        "samples": summaries,
    }
    write_json(out_dir / "manifest.json", jsonable(manifest))
    print(f"saved manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
