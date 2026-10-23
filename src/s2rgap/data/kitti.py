# SPDX-License-Identifier: Apache-2.0
"""torchvision-compatible KITTI 2D detection Dataset.

Returns (image_tensor, target_dict) where target_dict has the keys
fasterrcnn_resnet50_fpn expects: boxes (N,4 xyxy), labels (N,), image_id, area,
iscrowd. Classes follow KITTI conventions, mapped to a contiguous label index
that matches CARLA's extractor output (Car=1, Pedestrian=2, Cyclist=3).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

# Class label IDs match the CARLA-side extractor in tools/carla_labels.py so
# the same fasterrcnn head can be trained from either source.
KITTI_CLASSES: dict[str, int] = {
    "__background__": 0,
    "Car": 1,
    "Pedestrian": 2,
    "Cyclist": 3,
}

# KITTI's other types ("Van", "Truck", "Person_sitting", "Tram", "Misc", "DontCare")
# are folded into the relevant supercategory or dropped.
KITTI_TYPE_REMAP: dict[str, str | None] = {
    "Car": "Car",
    "Van": "Car",
    "Truck": "Car",
    "Pedestrian": "Pedestrian",
    "Person_sitting": "Pedestrian",
    "Cyclist": "Cyclist",
    "Tram": None,
    "Misc": None,
    "DontCare": None,
}


def parse_kitti_label_file(path: Path) -> list[dict[str, Any]]:
    objs: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        raw_type = parts[0]
        remap = KITTI_TYPE_REMAP.get(raw_type)
        if remap is None:
            continue
        truncation = float(parts[1])
        occlusion = int(parts[2])
        x1, y1, x2, y2 = (float(p) for p in parts[4:8])
        if x2 <= x1 or y2 <= y1:
            continue
        objs.append(
            {
                "cls": remap,
                "bbox": (x1, y1, x2, y2),
                "truncation": truncation,
                "occlusion": occlusion,
            }
        )
    return objs


class KITTIDetection(Dataset):
    """KITTI training-split detection dataset, restricted by a split JSON.

    Args:
        kitti_root:   path to .../kitti/extracted/training  (with image_2/, label_2/).
        split_json:   JSON with `train` and `val` filename lists (from build_kitti_splits.py).
        split:        "train" or "val".
        transforms:   optional callable applied to (image, target).
        min_box_area: drop boxes with pixel area below this.
        max_occlusion: drop boxes with occlusion >= this. 0 = fully visible, 3 = unknown.
    """

    def __init__(
        self,
        kitti_root: str | Path,
        split_json: str | Path,
        split: str = "train",
        transforms: Any = None,
        min_box_area: float = 25.0,
        max_occlusion: int = 2,
    ) -> None:
        kitti_root = Path(kitti_root).expanduser()
        split_json = Path(split_json).expanduser()
        with split_json.open() as f:
            cfg = json.load(f)
        if split not in cfg:
            raise KeyError(f"split `{split}` not in {sorted(cfg.keys())}")
        self.kitti_root = kitti_root
        self.image_dir = kitti_root / "image_2"
        self.label_dir = kitti_root / "label_2"
        self.frame_names: list[str] = list(cfg[split])
        self.transforms = transforms
        self.min_box_area = min_box_area
        self.max_occlusion = max_occlusion

    def __len__(self) -> int:
        return len(self.frame_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fname = self.frame_names[idx]
        img_path = self.image_dir / fname
        lbl_path = self.label_dir / (Path(fname).stem + ".txt")
        img = Image.open(img_path).convert("RGB")
        objs = parse_kitti_label_file(lbl_path)

        boxes: list[Sequence[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        for o in objs:
            if o["occlusion"] > self.max_occlusion:
                continue
            x1, y1, x2, y2 = o["bbox"]
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_box_area:
                continue
            boxes.append((x1, y1, x2, y2))
            labels.append(KITTI_CLASSES[o["cls"]])
            areas.append(area)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "frame_rel": fname,
        }
        img_tensor = F.to_tensor(img)
        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)
        return img_tensor, target


def collate_fn(batch: list) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Batch collator that matches torchvision detection conventions (lists, not tensors)."""
    images, targets = list(zip(*batch))
    return list(images), list(targets)
