"""Dataset for visible fragment-shell to object canonical-coordinate regression."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from fragreg.data.transforms import normalize_points, sample_or_pad_indices
from fragreg.geometry.symmetry import axis_to_vector, points_to_profile
from fragreg.registry import DATASETS


@dataclass(frozen=True)
class FragmentSampleInfo:
    scene_id: str
    scene_dir: Path
    frame_id: int
    fragment_id: int
    visible_points_path: Path
    T_C_from_O: np.ndarray
    num_shell_points: int


def _parse_frame_id(path: str | Path) -> int:
    match = re.search(r"frame_(\d+)", Path(path).stem)
    if match is None:
        raise ValueError(f"Cannot parse frame id from {path}")
    return int(match.group(1))


def _as_int(value: Any) -> int:
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def _normalize_filter(values: Any, cast: type) -> set[Any] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return {cast(values)}
    try:
        return {cast(value) for value in values}
    except TypeError:
        return {cast(values)}


@DATASETS.register_module()
class FragmentRegistrationDataset(Dataset):
    """One sample is one visible fragment in one rendered frame."""

    def __init__(
        self,
        dataset_root: str,
        split: str,
        num_points: int = 1024,
        min_shell_points: int = 32,
        normalize_input: bool = True,
        use_normals: bool = False,
        random_sample: bool = True,
        repeat_if_less: bool = True,
        axis: str = "y",
        return_profile: bool = True,
        scene_ids: str | list[str] | tuple[str, ...] | None = None,
        frame_ids: int | list[int] | tuple[int, ...] | None = None,
        fragment_ids: int | list[int] | tuple[int, ...] | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser()
        self.split = split
        self.split_dir = self.dataset_root / split
        self.num_points = int(num_points)
        self.min_shell_points = int(min_shell_points)
        self.normalize_input = bool(normalize_input)
        self.use_normals = bool(use_normals)
        self.random_sample = bool(random_sample)
        self.repeat_if_less = bool(repeat_if_less)
        self.axis = axis
        self.return_profile = bool(return_profile)
        self.scene_id_filter = _normalize_filter(scene_ids, str)
        self.frame_id_filter = _normalize_filter(frame_ids, int)
        self.fragment_id_filter = _normalize_filter(fragment_ids, int)
        self.max_samples = int(max_samples) if max_samples is not None else None

        if self.use_normals:
            raise NotImplementedError("Normals are not required in the first version.")
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        self.scene_dirs_all = sorted(p for p in self.split_dir.glob("scene_*") if p.is_dir())
        self.scene_dirs = [
            path for path in self.scene_dirs_all if self.scene_id_filter is None or path.name in self.scene_id_filter
        ]
        self.samples: list[FragmentSampleInfo] = []
        self._build_index()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        info = self.samples[index]
        with np.load(info.visible_points_path) as data:
            selected = self._select_shell_indices(data, info.fragment_id)
            if selected.size < self.min_shell_points:
                raise RuntimeError(
                    f"Indexed sample became invalid: {info.visible_points_path}, "
                    f"fragment={info.fragment_id}, shell_points={selected.size}"
                )

            points_C_all = np.asarray(data["points_C"], dtype=np.float32)[selected]
            points_O_all = np.asarray(data["points_O"], dtype=np.float32)[selected]

        local_indices, valid_mask = sample_or_pad_indices(
            len(points_C_all),
            self.num_points,
            random_sample=self.random_sample,
            repeat_if_less=self.repeat_if_less,
        )
        points_C_orig = points_C_all[local_indices].astype(np.float32)
        points_O = points_O_all[local_indices].astype(np.float32)

        if not valid_mask.all():
            points_C_orig = points_C_orig.copy()
            points_O = points_O.copy()
            points_C_orig[~valid_mask] = 0.0
            points_O[~valid_mask] = 0.0

        if self.normalize_input:
            points_C, centroid_C, scale_C = normalize_points(points_C_orig[valid_mask])
            full_points_C = np.zeros_like(points_C_orig, dtype=np.float32)
            full_points_C[valid_mask] = points_C
            points_C = full_points_C
        else:
            points_C = points_C_orig.copy()
            centroid_C = np.zeros(3, dtype=np.float32)
            scale_C = 1.0

        T_C_from_O = torch.from_numpy(info.T_C_from_O.astype(np.float32))
        axis_O = axis_to_vector(self.axis, dtype=T_C_from_O.dtype)
        gt_axis_C = F.normalize(T_C_from_O[:3, :3] @ axis_O, dim=0)

        sample = {
            "points_C": torch.from_numpy(points_C.astype(np.float32)),
            "points_C_orig": torch.from_numpy(points_C_orig.astype(np.float32)),
            "points_O": torch.from_numpy(points_O.astype(np.float32)),
            "fragment_id": info.fragment_id,
            "scene_id": info.scene_id,
            "frame_id": info.frame_id,
            "T_C_from_O": T_C_from_O,
            "gt_axis_C": gt_axis_C,
            "valid_mask": torch.from_numpy(valid_mask.astype(bool)),
            "centroid_C": torch.from_numpy(centroid_C.astype(np.float32)),
            "scale_C": torch.tensor(float(scale_C), dtype=torch.float32),
            "num_shell_points": info.num_shell_points,
        }
        if self.return_profile:
            sample["profile_O"] = points_to_profile(sample["points_O"], axis=self.axis)
        return sample

    def get_summary(self) -> dict[str, Any]:
        shell_counts = np.asarray([sample.num_shell_points for sample in self.samples], dtype=np.int64)
        stats: dict[str, Any] = {
            "dataset_root": str(self.dataset_root),
            "split": self.split,
            "num_scenes": len(self.scene_dirs),
            "num_scenes_total": len(self.scene_dirs_all),
            "num_scenes_used": len(self.scene_dirs),
            "scene_ids": [path.name for path in self.scene_dirs],
            "scene_filter": sorted(self.scene_id_filter) if self.scene_id_filter is not None else None,
            "frame_filter": sorted(self.frame_id_filter) if self.frame_id_filter is not None else None,
            "fragment_filter": sorted(self.fragment_id_filter) if self.fragment_id_filter is not None else None,
            "max_samples": self.max_samples,
            "num_samples": len(self.samples),
        }
        if shell_counts.size:
            stats.update(
                {
                    "shell_points_min": int(shell_counts.min()),
                    "shell_points_mean": float(shell_counts.mean()),
                    "shell_points_median": float(np.median(shell_counts)),
                    "shell_points_max": int(shell_counts.max()),
                }
            )
        return stats

    def _build_index(self) -> None:
        for scene_dir in self.scene_dirs:
            gt_path = scene_dir / "gt_annotations.json"
            visible_dir = scene_dir / "visible_points"
            if not gt_path.is_file() or not visible_dir.is_dir():
                continue

            pose_lookup = self._load_pose_lookup(gt_path)
            for npz_path in sorted(visible_dir.glob("frame_*.npz")):
                frame_id = _parse_frame_id(npz_path)
                if self.frame_id_filter is not None and frame_id not in self.frame_id_filter:
                    continue
                with np.load(npz_path) as data:
                    if "fragment_id" not in data or "points_C" not in data or "points_O" not in data:
                        continue
                    fragment_ids = np.unique(np.asarray(data["fragment_id"]))
                    for fragment_id_value in fragment_ids:
                        fragment_id = _as_int(fragment_id_value)
                        if self.fragment_id_filter is not None and fragment_id not in self.fragment_id_filter:
                            continue
                        selected = self._select_shell_indices(data, fragment_id)
                        if selected.size < self.min_shell_points:
                            continue
                        T_C_from_O = pose_lookup.get((frame_id, fragment_id))
                        if T_C_from_O is None:
                            continue
                        self.samples.append(
                            FragmentSampleInfo(
                                scene_id=scene_dir.name,
                                scene_dir=scene_dir,
                                frame_id=frame_id,
                                fragment_id=fragment_id,
                                visible_points_path=npz_path,
                                T_C_from_O=T_C_from_O,
                                num_shell_points=int(selected.size),
                            )
                        )
        if self.max_samples is not None:
            self.samples = self.samples[: self.max_samples]

    @staticmethod
    def _select_shell_indices(data: np.lib.npyio.NpzFile, fragment_id: int) -> np.ndarray:
        fragment_ids = np.asarray(data["fragment_id"])
        fragment_mask = fragment_ids == fragment_id

        shell_mask: np.ndarray | None = None
        if "shell_indices" in data:
            shell_indices = np.asarray(data["shell_indices"])
            if shell_indices.dtype == np.bool_ and shell_indices.shape[0] == fragment_ids.shape[0]:
                shell_mask = shell_indices.astype(bool)
            elif shell_indices.size > 0:
                shell_mask = np.zeros(fragment_ids.shape[0], dtype=bool)
                valid = shell_indices[(shell_indices >= 0) & (shell_indices < fragment_ids.shape[0])]
                shell_mask[valid.astype(np.int64)] = True

        if shell_mask is None:
            if "surface_label" not in data:
                raise KeyError("Neither shell_indices nor surface_label is available in visible_points npz.")
            shell_mask = np.asarray(data["surface_label"]) == 0

        return np.nonzero(fragment_mask & shell_mask)[0].astype(np.int64)

    @staticmethod
    def _load_pose_lookup(gt_path: Path) -> dict[tuple[int, int], np.ndarray]:
        with gt_path.open("r", encoding="utf-8") as f:
            gt = json.load(f)

        lookup: dict[tuple[int, int], np.ndarray] = {}
        frames = gt.get("frames", [])
        for frame in frames:
            frame_id = frame.get("frame_id")
            if frame_id is None and "visible_points" in frame:
                frame_id = _parse_frame_id(frame["visible_points"])
            if frame_id is None:
                continue
            for fragment in frame.get("fragments", []):
                if "fragment_id" not in fragment or "T_C_from_O" not in fragment:
                    continue
                fragment_id = int(fragment["fragment_id"])
                matrix = np.asarray(fragment["T_C_from_O"], dtype=np.float32)
                if matrix.shape != (4, 4):
                    continue
                lookup[(int(frame_id), fragment_id)] = matrix
        return lookup
