# SPDX-License-Identifier: Apache-2.0
"""Per-class and overall mAP@IoU=0.5 evaluation for our 3-class detection setup.

We do this in-process rather than via pycocotools to keep the dependency
surface small and the protocol explicit. Computes precision-recall curves
from predictions vs ground-truth boxes per class, averaged across the val
split. mAP = mean of per-class APs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def _box_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """IoU between two xyxy boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _ap_for_class(
    preds_by_image: dict[int, list[tuple[np.ndarray, float]]],
    gts_by_image: dict[int, list[np.ndarray]],
    iou_threshold: float = 0.5,
) -> tuple[float, int]:
    """Return AP and total ground-truth count for one class."""
    all_preds: list[tuple[int, np.ndarray, float]] = []
    for img_id, items in preds_by_image.items():
        for box, score in items:
            all_preds.append((img_id, box, score))
    all_preds.sort(key=lambda x: -x[2])

    n_gt = sum(len(b) for b in gts_by_image.values())
    if n_gt == 0:
        return 0.0, 0

    matched = {img_id: [False] * len(boxes) for img_id, boxes in gts_by_image.items()}
    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    for k, (img_id, pred_box, _score) in enumerate(all_preds):
        gts = gts_by_image.get(img_id, [])
        best_iou = 0.0
        best_idx = -1
        for j, gt_box in enumerate(gts):
            if matched[img_id][j]:
                continue
            iou = _box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0 and best_iou >= iou_threshold:
            tp[k] = 1.0
            matched[img_id][best_idx] = True
        else:
            fp[k] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall = tp_cum / max(n_gt, 1)

    # 11-point interpolation (KITTI / VOC style).
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        mask = recall >= t
        p_at_t = float(np.max(precision[mask])) if mask.any() else 0.0
        ap += p_at_t / 11.0
    return float(ap), int(n_gt)


def evaluate_map(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    class_names: dict[str, int],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute per-class AP and overall mAP at a single IoU threshold.

    `predictions` and `targets` follow the torchvision detection convention.
    `class_names` maps human-readable name -> contiguous class index.
    """
    classes = {name: idx for name, idx in class_names.items() if idx != 0}
    preds_by_class: dict[int, dict[int, list[tuple[np.ndarray, float]]]] = defaultdict(lambda: defaultdict(list))
    gts_by_class: dict[int, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    for img_idx, (pred, tgt) in enumerate(zip(predictions, targets)):
        p_boxes = pred["boxes"].detach().cpu().numpy()
        p_labels = pred["labels"].detach().cpu().numpy()
        p_scores = pred["scores"].detach().cpu().numpy()
        for box, lbl, score in zip(p_boxes, p_labels, p_scores):
            preds_by_class[int(lbl)][img_idx].append((box, float(score)))

        t_boxes = tgt["boxes"].detach().cpu().numpy()
        t_labels = tgt["labels"].detach().cpu().numpy()
        for box, lbl in zip(t_boxes, t_labels):
            gts_by_class[int(lbl)][img_idx].append(box)

    out: dict[str, float] = {}
    aps: list[float] = []
    for name, idx in classes.items():
        ap, n_gt = _ap_for_class(
            preds_by_class.get(idx, {}),
            gts_by_class.get(idx, {}),
            iou_threshold,
        )
        out[f"AP_{name}"] = ap
        out[f"n_gt_{name}"] = n_gt
        aps.append(ap)
    out[f"mAP@IoU={iou_threshold}"] = float(np.mean(aps)) if aps else 0.0
    return out


def save_eval_json(result: dict[str, float], path: Path, extra: dict | None = None) -> None:
    payload = {"metrics": result}
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
