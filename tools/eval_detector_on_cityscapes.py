#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Zero-shot evaluation of a torchvision Faster R-CNN on Cityscapes val.

Pair-2 (CARLA→Cityscapes) Tier-A baseline for the WACV pivot (2026-08-19).
Evaluates with the SAME protocol as the KITTI numbers (s2rgap.eval_detection
.evaluate_map, AP@IoU=0.5, 3 classes) and dumps per-frame predictions + targets
in the preds_vanilla_carla.json schema so tools/verifier_operating_point.py and
tools/fusion_ranking_analysis.py can replicate the redundancy gates on pair 2.

Class mapping caveat lives in src/s2rgap/data/cityscapes_det.py
(Car<-{car,truck}, Pedestrian<-{person}, Cyclist<-{rider}).

Usage:
    PY=python
    $PY tools/eval_detector_on_cityscapes.py \
        --ckpt checkpoints/vanilla_carla/last.pt \
        --result results/vanilla_carla_cityscapes_val.json \
        --preds-out results/preds_vanilla_carla_cityscapes.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from s2rgap.data.cityscapes_coco import CityscapesCOCO, CITYSCAPES_CLASSES
from s2rgap.eval_detection import evaluate_map


def log(msg: str) -> None:
    print(f"[cs-eval] {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cityscapes-root",
                   default=str(Path.home() / "datasets" / "cityscapes" / "extracted"))
    p.add_argument("--split", default="val")
    p.add_argument("--result", required=True)
    p.add_argument("--preds-out", default=None)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--min-box-area", type=float, default=25.0)
    p.add_argument("--ann-json", default="nvme/datasets/aldi_labels/cityscapes_val_instances.json",
                   help="ALDI standardized COCO annotations (see cityscapes_coco.py).")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        log("FAIL: no CUDA")
        return 1
    device = torch.device("cuda")

    log(f"loading torchvision FasterRCNN from {args.ckpt}")
    from s2rgap.detector import load_detector_from_ckpt
    model = load_detector_from_ckpt(args.ckpt, device)   # rebuilds fasterrcnn / fcos / retinanet from the checkpoint's saved args

    ds = CityscapesCOCO(Path(args.cityscapes_root).expanduser()/"leftImg8bit",
                        args.ann_json, split=args.split, min_box_area=args.min_box_area)
    n = min(args.limit, len(ds)) if args.limit else len(ds)
    log(f"evaluating on {n} Cityscapes {args.split} frames (native 2048x1024)")

    all_preds, all_targets, dump_preds, dump_targets = [], [], [], []
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            idxs = range(start, min(start + args.batch_size, n))
            imgs, tgts = [], []
            for i in idxs:
                im, tg = ds[i]
                imgs.append(im.to(device))
                tgts.append(tg)
            preds = model(imgs)
            for pr, tg in zip(preds, tgts):
                pr_cpu = {k: v.detach().cpu() for k, v in pr.items()}
                all_preds.append(pr_cpu)
                all_targets.append(tg)
                dump_preds.append({
                    "boxes": pr_cpu["boxes"].numpy().tolist(),
                    "labels": pr_cpu["labels"].numpy().tolist(),
                    "scores": pr_cpu["scores"].numpy().tolist(),
                })
                dump_targets.append({
                    "boxes": tg["boxes"].numpy().tolist(),
                    "labels": tg["labels"].numpy().tolist(),
                    "frame_rel": tg["frame_rel"],
                })
            done = start + len(list(idxs))
            if done % 100 < args.batch_size or done == n:
                log(f"  {done}/{n} frames ({1000*(time.time()-t0)/max(1,done):.0f} ms/frame)")

    metrics = evaluate_map(all_preds, all_targets, CITYSCAPES_CLASSES, iou_threshold=0.5)
    log(f"metrics: {metrics}")

    out = Path(args.result).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "source_ckpt": str(args.ckpt),
            "dataset": f"cityscapes_{args.split}",
            "n_frames": n,
            "class_mapping": "ALDI COCO: Car<-{car,truck}, Pedestrian<-{person}, Cyclist<-{rider}",
            "metrics": metrics,
        }, f, indent=2)
    log(f"saved metrics -> {out}")

    if args.preds_out:
        po = Path(args.preds_out).expanduser()
        with po.open("w") as f:
            json.dump({
                "source_ckpt": str(args.ckpt),
                "n_frames": n,
                "predictions": dump_preds,
                "targets": dump_targets,
            }, f)
        log(f"saved preds+targets -> {po} ({po.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
