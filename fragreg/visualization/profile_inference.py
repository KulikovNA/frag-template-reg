"""Small dependency-light visualization helpers for profile inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fragreg.utils.io import ensure_dir

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment extras.
    cv2 = None


def _as_uint8_colors(colors: np.ndarray, count: int) -> np.ndarray:
    colors = np.asarray(colors, dtype=np.uint8)
    if colors.ndim == 1:
        colors = np.repeat(colors[None, :], count, axis=0)
    if colors.shape != (count, 3):
        raise ValueError(f"Expected colors with shape ({count}, 3), got {colors.shape}.")
    return colors


def _error_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    vmax = float(np.percentile(values, 95)) if values.size else 1.0
    vmax = max(vmax, 1e-6)
    t = np.clip(values / vmax, 0.0, 1.0)
    red = (255.0 * t).astype(np.uint8)
    green = (180.0 * (1.0 - np.abs(t - 0.5) * 2.0)).astype(np.uint8)
    blue = (255.0 * (1.0 - t)).astype(np.uint8)
    return np.stack([red, green, blue], axis=1)


def write_colored_points_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write an ASCII PLY point cloud with RGB colors."""

    path = Path(path)
    ensure_dir(path.parent)
    points = np.asarray(points, dtype=np.float32)
    colors = _as_uint8_colors(colors, points.shape[0])
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            f.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def write_overlay_ply(
    path: str | Path,
    observed_points_C: np.ndarray,
    fitted_points_C: np.ndarray,
    max_edges: int = 512,
) -> None:
    """Write observed/fitted points plus sparse residual line segments."""

    path = Path(path)
    ensure_dir(path.parent)
    observed_points_C = np.asarray(observed_points_C, dtype=np.float32)
    fitted_points_C = np.asarray(fitted_points_C, dtype=np.float32)
    if observed_points_C.shape != fitted_points_C.shape:
        raise ValueError("observed_points_C and fitted_points_C must have the same shape.")

    observed_colors = np.repeat(np.array([[220, 220, 220]], dtype=np.uint8), observed_points_C.shape[0], axis=0)
    fitted_colors = np.repeat(np.array([[255, 120, 20]], dtype=np.uint8), fitted_points_C.shape[0], axis=0)
    points = np.concatenate([observed_points_C, fitted_points_C], axis=0)
    colors = np.concatenate([observed_colors, fitted_colors], axis=0)
    if observed_points_C.shape[0] > max_edges:
        edge_indices = np.linspace(0, observed_points_C.shape[0] - 1, max_edges, dtype=np.int64)
    else:
        edge_indices = np.arange(observed_points_C.shape[0], dtype=np.int64)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {edge_indices.shape[0]}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            f.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        offset = observed_points_C.shape[0]
        for idx in edge_indices:
            f.write(f"{int(idx)} {int(offset + idx)} 255 40 40\n")


def write_axis_ply(
    path: str | Path,
    center_C: np.ndarray,
    gt_axis_C: np.ndarray,
    pred_axis_C: np.ndarray | None = None,
    length: float = 0.1,
) -> None:
    """Write GT and optional predicted symmetry axes as colored line segments."""

    path = Path(path)
    ensure_dir(path.parent)
    center_C = np.asarray(center_C, dtype=np.float32)
    gt_axis_C = np.asarray(gt_axis_C, dtype=np.float32)
    gt_axis_C = gt_axis_C / max(float(np.linalg.norm(gt_axis_C)), 1e-8)
    vertices = [center_C - gt_axis_C * length, center_C + gt_axis_C * length]
    colors = [np.array([60, 220, 80], dtype=np.uint8), np.array([60, 220, 80], dtype=np.uint8)]
    edges = [(0, 1, np.array([60, 220, 80], dtype=np.uint8))]
    if pred_axis_C is not None:
        pred_axis_C = np.asarray(pred_axis_C, dtype=np.float32)
        pred_axis_C = pred_axis_C / max(float(np.linalg.norm(pred_axis_C)), 1e-8)
        start = len(vertices)
        vertices.extend([center_C - pred_axis_C * length, center_C + pred_axis_C * length])
        colors.extend([np.array([255, 70, 40], dtype=np.uint8), np.array([255, 70, 40], dtype=np.uint8)])
        edges.append((start, start + 1, np.array([255, 70, 40], dtype=np.uint8)))

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for vertex, color in zip(vertices, colors, strict=True):
            f.write(
                f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for start, end, color in edges:
            f.write(f"{start} {end} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def reconstruct_points_from_profile(profile: np.ndarray, reference_points_O: np.ndarray, axis: str = "y") -> np.ndarray:
    """Lift profile coordinates back to 3D using azimuth from reference object-frame points."""

    profile = np.asarray(profile, dtype=np.float32)
    reference_points_O = np.asarray(reference_points_O, dtype=np.float32)
    radial = profile[:, 0]
    axial = profile[:, 1]
    points = np.zeros((profile.shape[0], 3), dtype=np.float32)
    axis = axis.lower()
    if axis == "y":
        radial_ref = reference_points_O[:, [0, 2]]
        norms = np.linalg.norm(radial_ref, axis=1, keepdims=True)
        directions = radial_ref / np.maximum(norms, 1e-8)
        directions[norms[:, 0] < 1e-8] = np.array([1.0, 0.0], dtype=np.float32)
        points[:, 0] = radial * directions[:, 0]
        points[:, 1] = axial
        points[:, 2] = radial * directions[:, 1]
    elif axis == "z":
        radial_ref = reference_points_O[:, [0, 1]]
        norms = np.linalg.norm(radial_ref, axis=1, keepdims=True)
        directions = radial_ref / np.maximum(norms, 1e-8)
        directions[norms[:, 0] < 1e-8] = np.array([1.0, 0.0], dtype=np.float32)
        points[:, 0] = radial * directions[:, 0]
        points[:, 1] = radial * directions[:, 1]
        points[:, 2] = axial
    elif axis == "x":
        radial_ref = reference_points_O[:, [1, 2]]
        norms = np.linalg.norm(radial_ref, axis=1, keepdims=True)
        directions = radial_ref / np.maximum(norms, 1e-8)
        directions[norms[:, 0] < 1e-8] = np.array([1.0, 0.0], dtype=np.float32)
        points[:, 0] = axial
        points[:, 1] = radial * directions[:, 0]
        points[:, 2] = radial * directions[:, 1]
    else:
        raise ValueError(f"Unsupported axis {axis!r}; expected 'x', 'y' or 'z'.")
    return points


def _map_to_canvas(values: np.ndarray, low: float, high: float, size: int, margin: int) -> np.ndarray:
    denom = max(high - low, 1e-6)
    return margin + (values - low) / denom * (size - 2 * margin)


def draw_profile_scatter_png(
    path: str | Path,
    target_profile: np.ndarray,
    pred_profile: np.ndarray,
    coord_scale: float = 1000.0,
    metrics: dict[str, Any] | None = None,
    max_points: int = 2048,
) -> None:
    """Draw a profile scatter plot: x=axial, y=radius.

    If OpenCV is unavailable, write an SVG file. Callers can pass a ``.svg``
    path to force the dependency-free branch.
    """

    path = Path(path)
    ensure_dir(path.parent)
    target = np.asarray(target_profile, dtype=np.float32) * float(coord_scale)
    pred = np.asarray(pred_profile, dtype=np.float32) * float(coord_scale)
    if target.shape[0] > max_points:
        indices = np.linspace(0, target.shape[0] - 1, max_points, dtype=np.int64)
        target = target[indices]
        pred = pred[indices]

    width, height = 1200, 900
    margin = 90
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    all_axial = np.concatenate([target[:, 1], pred[:, 1]])
    all_radial = np.concatenate([target[:, 0], pred[:, 0]])
    axial_pad = max(float(np.ptp(all_axial)) * 0.08, 5.0)
    radial_pad = max(float(np.ptp(all_radial)) * 0.08, 5.0)
    x_low, x_high = float(all_axial.min() - axial_pad), float(all_axial.max() + axial_pad)
    y_low, y_high = float(all_radial.min() - radial_pad), float(all_radial.max() + radial_pad)

    x_gt = _map_to_canvas(target[:, 1], x_low, x_high, width, margin)
    y_gt = height - _map_to_canvas(target[:, 0], y_low, y_high, height, margin)
    x_pred = _map_to_canvas(pred[:, 1], x_low, x_high, width, margin)
    y_pred = height - _map_to_canvas(pred[:, 0], y_low, y_high, height, margin)
    text_parts = []
    if metrics:
        for key in ("profile_axial_l1_mm", "profile_r_l1_mm", "axial_corr", "axis_error_deg", "translation_error_mm"):
            if key in metrics:
                value = metrics[key]
                text_parts.append(f"{key}={value:.3g}" if isinstance(value, (int, float)) else f"{key}={value}")

    if cv2 is None or path.suffix.lower() == ".svg":
        if path.suffix.lower() != ".svg":
            path = path.with_suffix(".svg")
        line_step = max(1, target.shape[0] // 256)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<rect x="{margin}" y="{margin}" width="{width - 2 * margin}" height="{height - 2 * margin}" fill="none" stroke="#282828" stroke-width="2"/>',
            f'<text x="{margin}" y="42" font-family="monospace" font-size="22" fill="#141414">profile scatter: x=axial, y=radius (mm)</text>',
            f'<text x="{width - margin - 170}" y="42" font-family="monospace" font-size="20" fill="#46b446">GT</text>',
            f'<text x="{width - margin - 100}" y="42" font-family="monospace" font-size="20" fill="#ff641e">pred</text>',
        ]
        for tick in np.linspace(x_low, x_high, 5):
            x = float(_map_to_canvas(np.array([tick]), x_low, x_high, width, margin)[0])
            parts.append(f'<line x1="{x:.2f}" y1="{margin}" x2="{x:.2f}" y2="{height - margin}" stroke="#e6e6e6" stroke-width="1"/>')
            parts.append(f'<text x="{x - 25:.2f}" y="{height - margin + 35}" font-family="monospace" font-size="15" fill="#282828">{tick:.0f}</text>')
        for tick in np.linspace(y_low, y_high, 5):
            y = float(height - _map_to_canvas(np.array([tick]), y_low, y_high, height, margin)[0])
            parts.append(f'<line x1="{margin}" y1="{y:.2f}" x2="{width - margin}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
            parts.append(f'<text x="20" y="{y + 5:.2f}" font-family="monospace" font-size="15" fill="#282828">{tick:.0f}</text>')
        for idx in range(0, target.shape[0], line_step):
            parts.append(
                f'<line x1="{x_gt[idx]:.2f}" y1="{y_gt[idx]:.2f}" '
                f'x2="{x_pred[idx]:.2f}" y2="{y_pred[idx]:.2f}" stroke="#d2d2d2" stroke-width="1"/>'
            )
        for x, y in zip(x_gt, y_gt, strict=True):
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="#46b446"/>')
        for x, y in zip(x_pred, y_pred, strict=True):
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="#ff641e"/>')
        if text_parts:
            parts.append(f'<text x="{margin}" y="{height - 38}" font-family="monospace" font-size="15" fill="#141414">{" | ".join(text_parts[:3])}</text>')
            if len(text_parts) > 3:
                parts.append(f'<text x="{margin}" y="{height - 15}" font-family="monospace" font-size="15" fill="#141414">{" | ".join(text_parts[3:])}</text>')
        parts.append("</svg>\n")
        path.write_text("\n".join(parts), encoding="utf-8")
        return

    image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (margin, margin), (width - margin, height - margin), (40, 40, 40), 2)
    for tick in np.linspace(x_low, x_high, 5):
        x = int(_map_to_canvas(np.array([tick]), x_low, x_high, width, margin)[0])
        cv2.line(image, (x, margin), (x, height - margin), (230, 230, 230), 1)
        cv2.putText(image, f"{tick:.0f}", (x - 25, height - margin + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
    for tick in np.linspace(y_low, y_high, 5):
        y_raw = _map_to_canvas(np.array([tick]), y_low, y_high, height, margin)[0]
        y = int(height - y_raw)
        cv2.line(image, (margin, y), (width - margin, y), (230, 230, 230), 1)
        cv2.putText(image, f"{tick:.0f}", (20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)

    x_gt = x_gt.astype(np.int32)
    y_gt = y_gt.astype(np.int32)
    x_pred = x_pred.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    line_step = max(1, target.shape[0] // 256)
    for idx in range(0, target.shape[0], line_step):
        cv2.line(image, (x_gt[idx], y_gt[idx]), (x_pred[idx], y_pred[idx]), (210, 210, 210), 1)
    for x, y in zip(x_gt, y_gt, strict=True):
        cv2.circle(image, (int(x), int(y)), 2, (70, 180, 70), -1)
    for x, y in zip(x_pred, y_pred, strict=True):
        cv2.circle(image, (int(x), int(y)), 2, (30, 100, 255), -1)

    cv2.putText(image, "profile scatter: x=axial, y=radius (mm)", (margin, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2)
    cv2.putText(image, "GT", (width - margin - 170, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 180, 70), 2)
    cv2.putText(image, "pred", (width - margin - 100, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 100, 255), 2)
    if text_parts:
        cv2.putText(image, " | ".join(text_parts[:3]), (margin, height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1)
        if len(text_parts) > 3:
            cv2.putText(image, " | ".join(text_parts[3:]), (margin, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1)
    cv2.imwrite(str(path), image)


def write_error_points_ply(path: str | Path, points: np.ndarray, errors: np.ndarray) -> None:
    colors = _error_colors(np.asarray(errors, dtype=np.float32))
    write_colored_points_ply(path, points, colors)
