# SPDX-License-Identifier: Apache-2.0
"""torchvision-compatible Dataset for the extracted labeled CARLA cells.

Reads frames + KITTI-format .txt labels emitted by tools/extract_carla_frames.py.
Returns (image_tensor, target_dict) in fasterrcnn_resnet50_fpn format. Class
indices match KITTIDetection (Car=1, Pedestrian=2, Cyclist=3) so we can train
a single head from either source.

Optionally co-yields the precomputed Alpamayo latent for each frame so the
auxiliary CORAL-aligned-features head has its target. Pass an .npz with
`z` (N, D) and `paths` (N,) keys; we match frames by `<cell>/<frame.jpg>`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

from s2rgap.data.kitti import KITTI_CLASSES

# Re-export for convenience.
CARLA_CLASSES = KITTI_CLASSES


def _parse_carla_label(path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = parts[0]
        x1, y1, x2, y2 = (float(p) for p in parts[4:8])
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((cls, (x1, y1, x2, y2)))
    return out


class CARLADetection(Dataset):
    """Iterates the multi-cell labeled CARLA extraction.

    Args:
        root:           top-level extraction dir with one subdir per (town,weather) cell.
        cells:          optional cell allow-list. None = all subdirs.
        latents_npz:    optional path to per-frame Alpamayo latents (`z`, `paths`).
                        When set, `target["alpamayo_latent"]` is populated with the
                        4096-D vector for the frame.
        latent_transform: optional callable (np.ndarray) -> np.ndarray applied to each
                          latent before stuffing it into the target. Use to pre-apply
                          the CORAL transform so the model sees aligned targets directly.
        min_box_area:   drop boxes with pixel area below this.
        drop_empty:     drop frames with zero detections (after filtering).
    """

    def __init__(
        self,
        root: str | Path,
        cells: list[str] | None = None,
        latents_npz: str | Path | None = None,
        latent_transform: Any = None,
        min_box_area: float = 25.0,
        drop_empty: bool = False,
    ) -> None:
        root = Path(root).expanduser()
        self.root = root
        if cells is None:
            cell_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        else:
            cell_dirs = [root / c for c in cells]

        records: list[tuple[Path, Path, str]] = []  # (img, lbl, rel_path)
        for cell_dir in cell_dirs:
            for img_path in sorted(cell_dir.glob("frame_*.jpg")):
                lbl_path = img_path.with_suffix(".txt")
                rel = f"{cell_dir.name}/{img_path.name}"
                records.append((img_path, lbl_path, rel))

        self.records = records
        self.min_box_area = min_box_area
        self.drop_empty = drop_empty
        self.latent_transform = latent_transform

        # Latent index.
        self.latent_by_path: dict[str, np.ndarray] = {}
        if latents_npz is not None:
            blob = np.load(latents_npz, allow_pickle=True)
            z = blob["z"]
            paths = list(blob["paths"])
            for rel_path, vec in zip(paths, z):
                self.latent_by_path[str(rel_path)] = vec

        # Optional empty-frame filter (post-class-mapping).
        if drop_empty:
            self.records = [
                rec
                for rec in records
                if any(self._record_objects(rec[1]))
            ]

    def _record_objects(self, lbl_path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
        return _parse_carla_label(lbl_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_path, lbl_path, rel = self.records[idx]
        img = Image.open(img_path).convert("RGB")

        boxes: list[tuple[float, float, float, float]] = []
        labels: list[int] = []
        areas: list[float] = []
        for cls, (x1, y1, x2, y2) in self._record_objects(lbl_path):
            if cls not in CARLA_CLASSES:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_box_area:
                continue
            boxes.append((x1, y1, x2, y2))
            labels.append(CARLA_CLASSES[cls])
            areas.append(area)

        target: dict[str, torch.Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "frame_rel": rel,
        }

        if rel in self.latent_by_path:
            latent = self.latent_by_path[rel]
            if self.latent_transform is not None:
                latent = self.latent_transform(latent)
            target["alpamayo_latent"] = torch.from_numpy(np.ascontiguousarray(latent)).float()

        img_tensor = F.to_tensor(img)
        return img_tensor, target
