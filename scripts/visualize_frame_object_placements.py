#!/usr/bin/env python
"""Visualize all visible fragments from each frame in object coordinates."""

from __future__ import annotations

import argparse
import copy
import sys
from collections import defaultdict
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
from fragreg.visualization import write_colored_points_ply, write_error_points_ply


PALETTE = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/dgcnn_profile_global_axisym_y_one_scene.py",
        help="Path to config.",
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to checkpoint. Defaults to latest <train_cfg.work_dir>/<date>/best.pth when date_subdir is enabled.",
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--scene-id", default=None, help="Optional scene_id override.")
    parser.add_argument("--frame-ids", default=None, help="Comma-separated frame ids. Defaults to first frames.")
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--solver-iters", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--use-pred-axis-init", action="store_true")
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
    work_dir = Path(cfg.train_cfg.get("work_dir", f"work_dirs/{cfg._config_name}"))
    candidates = [work_dir / "best.pth"]
    if bool(cfg.train_cfg.get("date_subdir", True)) and work_dir.is_dir():
        candidates.extend(work_dir.glob("*/best.pth"))
    existing = [path for path in candidates if path.is_file()]
    if existing:
        return max(existing, key=lambda path: path.stat().st_mtime)
    return candidates[0]


def resolve_out_dir(checkpoint: Path, out_dir_arg: str | None) -> Path:
    if out_dir_arg is not None:
        return ensure_dir(out_dir_arg)
    return ensure_dir(checkpoint.resolve().parent / "frame_object_placements")


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


def rotation_about_axis(theta: float, axis: str) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    axis = axis.lower()
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
    raise ValueError(f"Unsupported axis {axis!r}; expected 'x', 'y' or 'z'.")


def yaw_align_matrix(R_pred: np.ndarray, R_gt: np.ndarray, axis: str) -> np.ndarray:
    """Return object-axis rotation S so R_pred @ S is closest to R_gt."""

    M = R_pred.T @ R_gt
    axis = axis.lower()
    if axis == "y":
        theta = np.arctan2(M[0, 2] - M[2, 0], M[0, 0] + M[2, 2])
    elif axis == "z":
        theta = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    elif axis == "x":
        theta = np.arctan2(M[2, 1] - M[1, 2], M[1, 1] + M[2, 2])
    else:
        raise ValueError(f"Unsupported axis {axis!r}; expected 'x', 'y' or 'z'.")
    return rotation_about_axis(float(theta), axis)


def color_for_fragment(fragment_id: int) -> np.ndarray:
    return PALETTE[int(fragment_id) % len(PALETTE)]


def write_object_axes_ply(path: Path, radius: float = 0.1) -> None:
    vertices = np.array(
        [
            [-radius, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [0.0, -radius, 0.0],
            [0.0, radius, 0.0],
            [0.0, 0.0, -radius],
            [0.0, 0.0, radius],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [[255, 60, 60], [255, 60, 60], [60, 220, 80], [60, 220, 80], [60, 100, 255], [60, 100, 255]],
        dtype=np.uint8,
    )
    edges = [(0, 1, colors[0]), (2, 3, colors[2]), (4, 5, colors[4])]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(vertices, colors, strict=True):
            f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {int(color[0])} {int(color[1])} {int(color[2])}\n")
        for start, end, color in edges:
            f.write(f"{start} {end} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def grouped_frame_indices(dataset) -> dict[tuple[str, int], list[int]]:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, info in enumerate(dataset.samples):
        groups[(info.scene_id, int(info.frame_id))].append(idx)
    return dict(groups)


def run_fragment(
    sample: dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    axis: str,
    units_cfg: dict[str, Any],
    solver_cfg: dict[str, Any],
) -> dict[str, Any]:
    batch = make_batch(sample, device)
    with torch.no_grad():
        outputs = model(batch["points_C"])

    pred_profile = outputs["pred_profile_O"]
    weights = batch["valid_mask"].to(dtype=pred_profile.dtype)
    init_axis_C = (
        outputs["pred_axis_C"]
        if solver_cfg.get("use_pred_axis_init", False) and "pred_axis_C" in outputs
        else None
    )
    T_pred, solver_diag = estimate_axisymmetric_pose(
        batch["points_C_orig"],
        pred_profile,
        weights=weights,
        axis=axis,
        init_axis_C=init_axis_C,
        num_iters=int(solver_cfg.get("num_iters", 200)),
        lr=float(solver_cfg.get("lr", 1e-2)),
        huber_beta=float(solver_cfg.get("huber_beta", 0.01)),
    )
    metrics = compute_profile_batch_metrics(
        pred_profile_O=pred_profile,
        target_profile_O=batch["profile_O"],
        points_C_orig=batch["points_C_orig"],
        T_pred=T_pred,
        T_gt=batch["T_C_from_O"],
        valid_mask=batch["valid_mask"],
        axis=axis,
        coord_scale=units_cfg["coord_scale"],
        coord_unit=units_cfg["coord_unit"],
        residual_profile_O=pred_profile,
    )[0]
    if "pred_axis_C" in outputs:
        metrics["pred_axis_error_deg"] = float(axis_vector_error_deg(outputs["pred_axis_C"], batch["gt_axis_C"])[0].detach().cpu())

    valid = tensor_to_numpy(batch["valid_mask"][0]).astype(bool)
    points_C = tensor_to_numpy(batch["points_C_orig"][0])[valid]
    points_O_gt = tensor_to_numpy(batch["points_O"][0])[valid]
    T_pred_i = T_pred[0]
    T_gt_i = batch["T_C_from_O"][0]
    points_O_pred = tensor_to_numpy(apply_transform(batch["points_C_orig"], invert_transform(T_pred))[0])[valid]
    S_yaw = yaw_align_matrix(
        tensor_to_numpy(T_pred_i[:3, :3]),
        tensor_to_numpy(T_gt_i[:3, :3]),
        axis=axis,
    )
    points_O_pred_yaw_aligned = points_O_pred @ S_yaw
    profile_error = np.linalg.norm(
        (tensor_to_numpy(pred_profile[0])[valid] - tensor_to_numpy(batch["profile_O"][0])[valid])
        * float(units_cfg["coord_scale"]),
        axis=1,
    )
    return {
        "scene_id": sample["scene_id"],
        "frame_id": int(sample["frame_id"]),
        "fragment_id": int(sample["fragment_id"]),
        "points_C": points_C,
        "points_O_gt": points_O_gt,
        "points_O_pred": points_O_pred,
        "points_O_pred_yaw_aligned": points_O_pred_yaw_aligned,
        "profile_error_mm": profile_error,
        "metrics": metrics,
        "solver_diagnostics": solver_diag,
        "T_pred": tensor_to_numpy(T_pred_i),
        "T_gt": tensor_to_numpy(T_gt_i),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    return value


def write_frame_outputs(frame_dir: Path, scene_id: str, frame_id: int, fragments: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    frame_dir = ensure_dir(frame_dir)
    pred_points = []
    pred_aligned_points = []
    gt_points = []
    error_points = []
    error_values = []
    colors = []
    fragment_summaries = []
    for item in fragments:
        fragment_id = int(item["fragment_id"])
        color = color_for_fragment(fragment_id)
        count = item["points_O_pred"].shape[0]
        pred_points.append(item["points_O_pred"])
        pred_aligned_points.append(item["points_O_pred_yaw_aligned"])
        gt_points.append(item["points_O_gt"])
        error_points.append(item["points_O_pred"])
        error_values.append(item["profile_error_mm"])
        colors.append(np.repeat(color[None, :], count, axis=0))
        fragment_summaries.append(
            {
                "fragment_id": fragment_id,
                "num_points": count,
                "metrics": item["metrics"],
                "solver_diagnostics": item["solver_diagnostics"],
                "T_pred": item["T_pred"],
                "T_gt": item["T_gt"],
            }
        )

    pred_points_np = np.concatenate(pred_points, axis=0)
    pred_aligned_points_np = np.concatenate(pred_aligned_points, axis=0)
    gt_points_np = np.concatenate(gt_points, axis=0)
    colors_np = np.concatenate(colors, axis=0)
    error_points_np = np.concatenate(error_points, axis=0)
    error_values_np = np.concatenate(error_values, axis=0)

    write_colored_points_ply(frame_dir / "object_fragments_pred.ply", pred_points_np, colors_np)
    write_colored_points_ply(frame_dir / "object_fragments_pred_yaw_aligned_to_gt.ply", pred_aligned_points_np, colors_np)
    write_colored_points_ply(frame_dir / "object_fragments_gt.ply", gt_points_np, colors_np)
    write_error_points_ply(frame_dir / "object_fragments_pred_profile_error.ply", error_points_np, error_values_np)
    radius = max(float(np.linalg.norm(gt_points_np, axis=1).max()) * 1.15, 0.05)
    write_object_axes_ply(frame_dir / "object_axes_xyz_rgb.ply", radius=radius)

    summary = {
        "scene_id": scene_id,
        "frame_id": frame_id,
        "axis": axis,
        "num_fragments": len(fragments),
        "num_points": int(pred_points_np.shape[0]),
        "artifacts": {
            "pred": str(frame_dir / "object_fragments_pred.ply"),
            "pred_yaw_aligned_to_gt": str(frame_dir / "object_fragments_pred_yaw_aligned_to_gt.ply"),
            "gt": str(frame_dir / "object_fragments_gt.ply"),
            "pred_profile_error": str(frame_dir / "object_fragments_pred_profile_error.ply"),
            "object_axes": str(frame_dir / "object_axes_xyz_rgb.ply"),
        },
        "fragments": fragment_summaries,
    }
    write_json(frame_dir / "frame_summary.json", jsonable(summary))
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
    if args.use_pred_axis_init:
        solver_cfg["use_pred_axis_init"] = True

    checkpoint = resolve_checkpoint(cfg, args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    out_dir = resolve_out_dir(checkpoint, args.out_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    loader_cfg = dataloader_cfg_for_split(cfg, args.split)
    dataset_cfg = copy.deepcopy(loader_cfg["dataset"])
    dataset_cfg["return_profile"] = True
    dataset_cfg["random_sample"] = False
    if args.scene_id is not None:
        dataset_cfg["scene_ids"] = [args.scene_id]
    frame_ids = parse_int_list(args.frame_ids)
    if frame_ids is not None:
        dataset_cfg["frame_ids"] = frame_ids
    if args.num_points is not None:
        dataset_cfg["num_points"] = args.num_points

    dataset = build_dataset(dataset_cfg)
    frame_groups = grouped_frame_indices(dataset)
    available_frames = sorted(frame_groups)
    if frame_ids is None:
        selected_frames = available_frames[: int(args.num_frames)]
    else:
        selected_frames = [key for key in available_frames if key[1] in set(frame_ids)]
    if not selected_frames:
        raise RuntimeError("No frames selected for visualization.")

    model = build_model(cfg.model).to(device)
    checkpoint_payload = load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    print(f"config: {args.config}")
    print(f"checkpoint: {checkpoint}")
    print(f"checkpoint epoch: {checkpoint_payload.get('epoch', 'unknown')}")
    print(f"device: {device}")
    print(f"output: {out_dir}")
    print(f"dataset summary: {dataset.get_summary() if hasattr(dataset, 'get_summary') else {}}")
    print(f"selected frames: {selected_frames}")
    print("Note: pred_yaw_aligned_to_gt uses GT yaw only for visualization, not inference.")

    frame_summaries = []
    for scene_id, frame_id in selected_frames:
        fragments = []
        for index in frame_groups[(scene_id, frame_id)]:
            fragments.append(
                run_fragment(
                    sample=dataset[index],
                    model=model,
                    device=device,
                    axis=axis,
                    units_cfg=units_cfg,
                    solver_cfg=solver_cfg,
                )
            )
        frame_dir = out_dir / f"{scene_id}_frame_{frame_id:06d}"
        summary = write_frame_outputs(frame_dir, scene_id=scene_id, frame_id=frame_id, fragments=fragments, axis=axis)
        frame_summaries.append(summary)
        axis_errors = [item["metrics"].get("axis_error_deg", float("nan")) for item in fragments]
        trans_errors = [item["metrics"].get("translation_error_mm", float("nan")) for item in fragments]
        print(
            f"[{frame_dir}] fragments={len(fragments)}, "
            f"axis_error_deg_mean={float(np.nanmean(axis_errors)):.4g}, "
            f"translation_error_mm_mean={float(np.nanmean(trans_errors)):.4g}"
        )

    manifest = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "split": args.split,
        "axis": axis,
        "coord_unit": units_cfg["coord_unit"],
        "coord_scale": units_cfg["coord_scale"],
        "yaw_alignment_note": "pred_yaw_aligned_to_gt uses GT yaw only for visualization; object yaw is unidentifiable for the axisymmetric profile model.",
        "frames": frame_summaries,
    }
    write_json(out_dir / "manifest.json", jsonable(manifest))
    print(f"saved manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
