# SPDX-License-Identifier: Apache-2.0
"""KITTI dataset variant that yields *pseudo-labels* instead of real labels.

Used by the Alpamayo Teacher student (Stage 2c). At training time the
student treats KITTI train images as "labeled" using the predictions of the
frozen Alpamayo-backbone labeler from Stage 2a. The labeler's predictions
are loaded from the JSON file produced by ``tools/label_kitti_with_teacher.py``;
this wrapper filters them by a confidence threshold τ and produces standard
torchvision detection target dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

from s2rgap.data.kitti import KITTI_CLASSES


__all__ = ["KITTIPseudoLabels"]


class KITTIPseudoLabels(Dataset):
    """KITTI training images with labeler-generated pseudo-bounding-boxes.

    Args:
        kitti_root: ``.../kitti/extracted/training``.
        pseudo_json: JSON produced by ``label_kitti_with_teacher.py``.
            Must contain ``predictions`` keyed by KITTI frame filename
            (e.g. ``"000000.png"``).
        score_thresh: confidence floor; predictions with score < τ are
            dropped. 0.7 follows DINO Teacher's default; do an ablation
            sweep over {0.5, 0.7, 0.8} if time allows.
        min_box_area: drop predictions with pixel area below this (matches
            ``KITTIDetection``'s default).
        max_dets_per_image: cap on retained detections per frame after
            filtering. Useful to drop runaway long-tail predictions.
        require_predictions: if True, raises if a frame is not in the JSON;
            else falls back to an empty target.
    """

    def __init__(
        self,
        kitti_root: str | Path,
        pseudo_json: str | Path,
        score_thresh: float = 0.7,
        min_box_area: float = 25.0,
        max_dets_per_image: int | None = 100,
        require_predictions: bool = True,
    ) -> None:
        kitti_root = Path(kitti_root).expanduser()
        pseudo_json = Path(pseudo_json).expanduser()
        with pseudo_json.open() as f:
            blob = json.load(f)
        if "predictions" not in blob:
            raise ValueError(
                f"{pseudo_json} missing 'predictions' key (got: {list(blob.keys())})"
            )
        self.preds: dict[str, dict[str, list]] = blob["predictions"]
        self.kitti_root = kitti_root
        self.image_dir = kitti_root / "image_2"
        self.frame_names: list[str] = sorted(self.preds.keys())
        self.score_thresh = float(score_thresh)
        self.min_box_area = float(min_box_area)
        self.max_dets_per_image = max_dets_per_image
        self.require_predictions = require_predictions

    def __len__(self) -> int:
        return len(self.frame_names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        fname = self.frame_names[idx]
        img_path = self.image_dir / fname
        img = Image.open(img_path).convert("RGB")

        pred = self.preds.get(fname)
        if pred is None:
            if self.require_predictions:
                raise KeyError(f"no pseudo-label entry for {fname}")
            boxes_keep: list[list[float]] = []
            labels_keep: list[int] = []
            scores_keep: list[float] = []
        else:
            boxes = pred["boxes"]
            labels = pred["labels"]
            scores = pred["scores"]
            boxes_keep, labels_keep, scores_keep = [], [], []
            for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
                if sc < self.score_thresh:
                    continue
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area < self.min_box_area:
                    continue
                # Drop background class predictions (label 0); shouldn't appear
                # but be defensive.
                if lab == 0:
                    continue
                boxes_keep.append([x1, y1, x2, y2])
                labels_keep.append(int(lab))
                scores_keep.append(float(sc))
            # Top-k by score if too many.
            if (
                self.max_dets_per_image is not None
                and len(boxes_keep) > self.max_dets_per_image
            ):
                order = sorted(
                    range(len(scores_keep)), key=lambda i: -scores_keep[i]
                )[: self.max_dets_per_image]
                boxes_keep = [boxes_keep[i] for i in order]
                labels_keep = [labels_keep[i] for i in order]
                scores_keep = [scores_keep[i] for i in order]

        if boxes_keep:
            boxes_t = torch.tensor(boxes_keep, dtype=torch.float32)
            labels_t = torch.tensor(labels_keep, dtype=torch.int64)
            areas_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t = torch.zeros((0,), dtype=torch.float32)

        target: dict[str, Any] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
            "frame_rel": fname,
            # Per-detection pseudo-label score (handy for diagnostics / weighted loss).
            "pseudo_scores": torch.tensor(scores_keep, dtype=torch.float32) if scores_keep else torch.zeros((0,), dtype=torch.float32),
        }
        img_tensor = F.to_tensor(img)
        return img_tensor, target
