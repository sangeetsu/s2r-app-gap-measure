#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Confidence calibration analysis (Tier-1.5 new idea per publishability_plan.md).

For each model's captured KITTI val predictions, computes:
  - **Reliability diagram per class**: bin predictions by confidence score,
    compute the empirical precision-at-IoU>=0.5 vs the model's confidence.
    Well-calibrated detectors have empirical precision ≈ confidence (the y=x line).
  - **Expected Calibration Error (ECE)** per class: weighted L1 distance from y=x.

Why this matters: if the per-class specialists are well-calibrated, their confidence
scores can be directly compared across models — the per-class hard-rule ensemble's
implicit assumption is that "highest-confidence specialist wins per class," which
only works if confidence scales meaningfully across models. If models are
poorly calibrated, a learned ensemble (T3.2) could exploit the miscalibration.

Usage:
    PY=python
    $PY tools/calibration_analysis.py \\
        --preds results/preds_vanilla_carla.json:vanilla \\
        --preds results/preds_dinov2_mtmsrd.json:DINOv2_M12 \\
        ...
        --iou 0.5 --n-bins 10 --out-dir figures/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from s2rgap.data.kitti import KITTI_CLASSES
from torchvision.ops import box_iou as tv_box_iou


def log(msg: str) -> None:
    print(f"[calib] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--preds", action="append", required=True, help="<json_path>:<label>")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--out-json", default="results/calibration.json")
    return p.parse_args()


def _deserialize_pred(d: dict) -> dict:
    return {
        "boxes": torch.tensor(d["boxes"], dtype=torch.float32) if len(d["boxes"]) else torch.zeros((0, 4)),
        "labels": torch.tensor(d["labels"], dtype=torch.int64) if len(d["labels"]) else torch.zeros((0,), dtype=torch.int64),
        "scores": torch.tensor(d["scores"], dtype=torch.float32) if len(d["scores"]) else torch.zeros((0,)),
    }


def _deserialize_target(d: dict) -> dict:
    return {
        "boxes": torch.tensor(d["boxes"], dtype=torch.float32) if len(d["boxes"]) else torch.zeros((0, 4)),
        "labels": torch.tensor(d["labels"], dtype=torch.int64) if len(d["labels"]) else torch.zeros((0,), dtype=torch.int64),
    }


def collect_tp_scores(preds: list, tgts: list, class_idx: int, iou_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    """For each prediction of `class_idx`, compute (score, is_tp) via greedy IoU matching to same-class GT."""
    scores_l: list[float] = []
    tp_l: list[bool] = []
    for pred, tgt in zip(preds, tgts):
        p_mask = pred["labels"] == class_idx
        t_mask = tgt["labels"] == class_idx
        p_boxes = pred["boxes"][p_mask]
        p_scores = pred["scores"][p_mask].numpy()
        t_boxes = tgt["boxes"][t_mask]
        if len(p_boxes) == 0:
            continue
        order = np.argsort(-p_scores)
        p_boxes = p_boxes[order]
        p_scores = p_scores[order]
        matched = np.zeros(len(t_boxes), dtype=bool)
        for i, pb in enumerate(p_boxes):
            if len(t_boxes) == 0:
                scores_l.append(float(p_scores[i]))
                tp_l.append(False)
                continue
            ious = tv_box_iou(pb.unsqueeze(0), t_boxes).squeeze(0).numpy()
            ious_valid = np.where(matched, -1.0, ious)
            best = int(np.argmax(ious_valid))
            if ious_valid[best] >= iou_thresh:
                matched[best] = True
                scores_l.append(float(p_scores[i]))
                tp_l.append(True)
            else:
                scores_l.append(float(p_scores[i]))
                tp_l.append(False)
    return np.array(scores_l), np.array(tp_l)


def reliability(scores: np.ndarray, tps: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (bin_centers, empirical_precision, weights) for `n_bins` equal-width score bins."""
    if len(scores) == 0:
        return np.array([]), np.array([]), np.array([])
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    precision = np.zeros(n_bins)
    weights = np.zeros(n_bins)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (scores >= lo) & (scores < hi if b < n_bins - 1 else scores <= hi)
        n_in = in_bin.sum()
        weights[b] = n_in
        precision[b] = (tps[in_bin].sum() / max(1, n_in)) if n_in > 0 else 0.0
    return centers, precision, weights


def ece(centers: np.ndarray, precision: np.ndarray, weights: np.ndarray) -> float:
    """Expected calibration error: weighted L1 from y=x."""
    total = weights.sum()
    if total == 0:
        return 0.0
    return float((np.abs(precision - centers) * weights).sum() / total)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = {name: idx for name, idx in KITTI_CLASSES.items() if idx != 0}

    configs: list[tuple[str, list, list]] = []
    for spec in args.preds:
        path, _, label = spec.rpartition(":")
        if not path:
            path, label = label, label
        log(f"loading {path}  ->  '{label}'")
        blob = json.loads(Path(path).read_text())
        preds = [_deserialize_pred(p) for p in blob["predictions"]]
        tgts = [_deserialize_target(t) for t in blob["targets"]]
        configs.append((label, preds, tgts))

    all_results: dict[str, dict] = {}
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4.2), sharey=True)
    if len(classes) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    for ax, (cls_name, cls_idx) in zip(axes, classes.items()):
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="perfect calibration")
        for color, (label, preds, tgts) in zip(colors, configs):
            scores, tps = collect_tp_scores(preds, tgts, cls_idx, args.iou)
            centers, precision, weights = reliability(scores, tps, args.n_bins)
            if len(centers) == 0:
                continue
            cal_err = ece(centers, precision, weights)
            label_full = f"{label} (ECE={cal_err:.3f})"
            # Plot bars at non-zero bins.
            mask = weights > 0
            ax.plot(centers[mask], precision[mask], "-o", label=label_full, color=color, markersize=4, linewidth=1.2)
            all_results.setdefault(label, {})[cls_name] = {
                "centers": centers.tolist(), "precision": precision.tolist(),
                "weights": weights.tolist(), "ece": cal_err,
            }
        ax.set_xlabel("Confidence")
        ax.set_title(f"{cls_name}")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    axes[0].set_ylabel(f"Empirical precision @ IoU={args.iou}")
    fig.suptitle(f"Reliability diagram (KITTI val, IoU={args.iou}, {args.n_bins} bins)", y=1.02)
    fig.tight_layout()
    out_pdf = out_dir / f"calibration_iou{args.iou:.2f}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    log(f"saved {out_pdf}")
    plt.close(fig)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"results": all_results, "n_bins": args.n_bins, "iou": args.iou}, indent=2))
    log(f"saved {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
