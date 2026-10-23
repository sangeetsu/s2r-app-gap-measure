# SPDX-License-Identifier: Apache-2.0
"""torchvision-compatible Cityscapes 2D detection Dataset (boxes from gtFine polygons).

Second sim-to-real pair for the WACV pivot (added 2026-08-19). Returns
(image_tensor, target_dict) with the same contract as
``s2rgap.data.kitti.KITTIDetection`` so every downstream tool
(train_detector eval loop, probe transfer, verifier operating point) can
consume it unchanged.

Class mapping — chosen for protocol consistency with the KITTI remap
(KITTI folds Van/Truck into Car; Person_sitting into Pedestrian):

    Car        <- car, truck
    Pedestrian <- person
    Cyclist    <- rider          (CAVEAT: Cityscapes 'rider' includes
                                  motorcyclists; disclose in the paper)

Dropped: bus, train (tram-like, dropped in KITTI too), motorcycle, bicycle
(the vehicle itself — KITTI Cyclist annotates rider+bike as one box, so the
closest Cityscapes analogue is 'rider'), caravan, trailer, and every stuff
class. Boxes are the axis-aligned extents of the gtFine polygon, clipped to
the image. 'group' variants (persongroup, cargroup, ridergroup, truckgroup)
are dropped — they are crowd regions without instance separation, analogous
to KITTI DontCare.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

# Same contiguous ids as CARLA extractor / KITTIDetection.
CITYSCAPES_CLASSES: dict[str, int] = {
    "__background__": 0,
    "Car": 1,
    "Pedestrian": 2,
    "Cyclist": 3,
}

CITYSCAPES_LABEL_REMAP: dict[str, str | None] = {
    "car": "Car",
    "truck": "Car",
    "person": "Pedestrian",
    "rider": "Cyclist",
}


def polygon_to_bbox(polygon: list[list[float]], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(img_w), max(xs))
    y2 = min(float(img_h), max(ys))
    return x1, y1, x2, y2


def parse_polygons_file(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        blob = json.load(f)
    img_w, img_h = int(blob["imgWidth"]), int(blob["imgHeight"])
    objs: list[dict[str, Any]] = []
    for o in blob["objects"]:
        # Skip deleted objects (gtFine convention) and group/crowd regions.
        if o.get("deleted", 0):
            continue
        label = o["label"]
        if label.endswith("group"):
            continue
        remap = CITYSCAPES_LABEL_REMAP.get(label)
        if remap is None:
            continue
        x1, y1, x2, y2 = polygon_to_bbox(o["polygon"], img_w, img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        objs.append({"cls": remap, "bbox": (x1, y1, x2, y2)})
    return objs


class CityscapesDetection(Dataset):
    """Cityscapes detection dataset over a gtFine split.

    Args:
        cityscapes_root: path to .../cityscapes/extracted (with leftImg8bit/, gtFine/).
        split:           "train" | "val".
        transforms:      optional callable applied to (image, target).
        min_box_area:    drop boxes below this pixel area (KITTI-protocol default 25;
                         note Cityscapes images are 2048x1024 vs KITTI ~1242x375, so
                         consider the resolution difference when comparing).
        resize_hw:       optional (H, W); when set, image AND boxes are resized. Use
                         to keep inference memory bounded (e.g. (512, 1024)).
    """

    def __init__(
        self,
        cityscapes_root: str | Path,
        split: str = "val",
        transforms: Any = None,
        min_box_area: float = 25.0,
        resize_hw: tuple[int, int] | None = None,
    ) -> None:
        root = Path(cityscapes_root).expanduser()
        self.image_dir = root / "leftImg8bit" / split
        self.label_dir = root / "gtFine" / split
        if not self.image_dir.is_dir():
            raise FileNotFoundError(self.image_dir)
        self.frame_paths: list[Path] = sorted(self.image_dir.rglob("*_leftImg8bit.png"))
        self.transforms = transforms
        self.min_box_area = min_box_area
        self.resize_hw = resize_hw
        self.split = split

    def __len__(self) -> int:
        return len(self.frame_paths)

    def _label_path(self, img_path: Path) -> Path:
        city = img_path.parent.name
        stem = img_path.name.replace("_leftImg8bit.png", "_gtFine_polygons.json")
        return self.label_dir / city / stem

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_path = self.frame_paths[idx]
        img = Image.open(img_path).convert("RGB")
        W0, H0 = img.size
        objs = parse_polygons_file(self._label_path(img_path))

        sx = sy = 1.0
        if self.resize_hw is not None:
            H1, W1 = self.resize_hw
            sx, sy = W1 / W0, H1 / H0
            img = img.resize((W1, H1), Image.BILINEAR)

        boxes, labels, areas = [], [], []
        for o in objs:
            x1, y1, x2, y2 = o["bbox"]
            x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_box_area:
                continue
            boxes.append((x1, y1, x2, y2))
            labels.append(CITYSCAPES_CLASSES[o["cls"]])
            areas.append(area)

        # frame_rel matches the rel format of gather_cityscapes() in
        # tools/extract_alpamayo_feature_maps.py: path relative to leftImg8bit/<split>.
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "frame_rel": str(img_path.relative_to(self.image_dir)),
        }
        img_tensor = F.to_tensor(img)
        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)
        return img_tensor, target
