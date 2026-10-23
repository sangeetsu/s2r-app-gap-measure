# SPDX-License-Identifier: Apache-2.0
"""Cityscapes detection from ALDI's STANDARDIZED COCO annotations.

Replaces the polygon-derived boxes in cityscapes_det.py, which were measured on 2026-08-26
to be systematically ~26% larger in area than the community standard (mean best-match IoU
0.761 vs ALDI; ours larger in 63.6% of cases, smaller in 1.5%; outward bias on all four
edges). Root cause: gtFine POLYGONS include occluded extent, whereas Detectron2 and ALDI
derive boxes from ``instanceIds.png`` — visible pixels only.

ALDI (Kay et al., TMLR 2025, github.com/justinkay/aldi) publishes standardized COCO
annotations precisely because ground-truth boxes have historically differed between DAOD
codebases; using theirs is what makes our numbers comparable to published ones.

Fetch (no login required):
    mkdir -p nvme/datasets/aldi_labels && cd nvme/datasets/aldi_labels
    for f in cityscapes_train_instances.json cityscapes_val_instances.json; do
      curl -sLO https://github.com/justinkay/aldi/releases/download/v0.0.1/$f
    done

Class mapping to our 3-class protocol (ALDI ids: 0 person, 1 rider, 2 car, 3 truck,
4 bus, 5 train, 6 motorcycle, 7 bicycle):
    Car        <- car, truck        (mirrors the KITTI Van/Truck fold)
    Pedestrian <- person
    Cyclist    <- rider             (the rider, not the bicycle; matches KITTI's Cyclist)
Dropped: bus, train (tram-like, dropped in KITTI too), motorcycle, bicycle (the vehicle
itself — KITTI annotates rider+bike as one Cyclist box, so 'rider' is the analogue).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

CITYSCAPES_CLASSES: dict[str, int] = {
    "__background__": 0,
    "Car": 1,
    "Pedestrian": 2,
    "Cyclist": 3,
}

# ALDI category id -> our contiguous label (None = drop)
ALDI_ID_TO_OURS: dict[int, int | None] = {
    0: 2,   # person     -> Pedestrian
    1: 3,   # rider      -> Cyclist
    2: 1,   # car        -> Car
    3: 1,   # truck      -> Car
    4: None,  # bus
    5: None,  # train
    6: None,  # motorcycle
    7: None,  # bicycle
}


class CityscapesCOCO(Dataset):
    """Cityscapes detection with ALDI-standard boxes.

    Args:
        images_root: .../cityscapes/extracted/leftImg8bit  (the split dir is appended)
        ann_json:    ALDI cityscapes_{train,val}_instances.json
        split:       "train" | "val" — used to locate images under images_root
        min_box_area: drop degenerate boxes below this pixel area
    """

    def __init__(
        self,
        images_root: str | Path,
        ann_json: str | Path,
        split: str = "val",
        transforms: Any = None,
        min_box_area: float = 25.0,
    ) -> None:
        self.images_root = Path(images_root).expanduser() / split
        self.transforms = transforms
        self.min_box_area = min_box_area
        with Path(ann_json).expanduser().open() as f:
            blob = json.load(f)

        self.by_img: dict[int, list[dict]] = {}
        for a in blob["annotations"]:
            self.by_img.setdefault(a["image_id"], []).append(a)

        # keep only images that exist on disk; store (image_id, relative path)
        self.entries: list[tuple[int, str]] = []
        for im in blob["images"]:
            fn = im["file_name"]
            # ALDI file_name may be "frankfurt/frankfurt_000000_000294_leftImg8bit.png"
            # or a bare basename; resolve both.
            cand = self.images_root / fn
            if not cand.exists():
                base = fn.split("/")[-1]
                city = base.split("_")[0]
                cand = self.images_root / city / base
                fn = f"{city}/{base}"
            if cand.exists():
                self.entries.append((im["id"], fn))
        if not self.entries:
            raise FileNotFoundError(
                f"no Cityscapes images resolved under {self.images_root} for {ann_json}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_id, rel = self.entries[idx]
        img = Image.open(self.images_root / rel).convert("RGB")
        boxes, labels, areas = [], [], []
        for a in self.by_img.get(img_id, []):
            lab = ALDI_ID_TO_OURS.get(int(a["category_id"]))
            if lab is None:
                continue
            x, y, w, h = a["bbox"]           # COCO xywh
            if w <= 0 or h <= 0 or w * h < self.min_box_area:
                continue
            boxes.append((x, y, x + w, y + h))
            labels.append(lab)
            areas.append(w * h)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "frame_rel": rel,
        }
        img_t = F.to_tensor(img)
        if self.transforms is not None:
            img_t, target = self.transforms(img_t, target)
        return img_t, target
