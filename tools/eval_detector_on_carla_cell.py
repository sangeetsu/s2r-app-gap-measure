#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a trained torchvision detector on one CARLA cell (appearance-gap grid)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from s2rgap.data.carla import CARLADetection
from s2rgap.eval_detection import evaluate_map, save_eval_json

CLASSES = {"__background__": 0, "Car": 1, "Pedestrian": 2, "Cyclist": 3}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cell", required=True)
    p.add_argument("--carla-root", default=str(Path.home()/"datasets"/"carla_extracted"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--frames-slice", default=None, help="start:end of sorted cell frames, e.g. 800:1000 (held-out anchor split)")
    p.add_argument("--result", required=True)
    p.add_argument("--min-box-height", type=float, default=0.0, help="visibility control: drop GT boxes and predictions shorter than this (px) before mAP")
    a = p.parse_args()
    device = torch.device("cuda")
    from s2rgap.detector import load_detector_from_ckpt
    m = load_detector_from_ckpt(a.ckpt, device)   # rebuilds fasterrcnn / fcos / retinanet from the checkpoint's saved args
    ds = CARLADetection(root=a.carla_root, cells=[a.cell])
    if a.frames_slice:
        s, e = (int(x) for x in a.frames_slice.split(":")); ds = torch.utils.data.Subset(ds, range(s, min(e, len(ds))))
    n = min(a.limit, len(ds)) if a.limit else len(ds)
    preds, tgts = [], []
    with torch.inference_mode():
        for s in range(0, n, a.batch_size):
            imgs, ts = [], []
            for i in range(s, min(s+a.batch_size, n)):
                im, t = ds[i]; imgs.append(im.to(device)); ts.append(t)
            for pr, t in zip(m(imgs), ts):
                pr = {k: v.detach().cpu() for k, v in pr.items()}
                if a.min_box_height > 0:
                    kp = (pr["boxes"][:, 3] - pr["boxes"][:, 1]) >= a.min_box_height; pr = {k: v[kp] for k, v in pr.items()}
                    kt = (t["boxes"][:, 3] - t["boxes"][:, 1]) >= a.min_box_height; t = {**t, "boxes": t["boxes"][kt], "labels": t["labels"][kt]}
                preds.append(pr)
                tgts.append(t)
    met = evaluate_map(preds, tgts, CLASSES, iou_threshold=0.5)
    print(f"[cell-eval] RESULT {Path(a.ckpt).parent.name} on {a.cell}: "
          f"mAP {met['mAP@IoU=0.5']*100:.2f}", flush=True)
    save_eval_json(met, Path(a.result), extra={"ckpt": a.ckpt, "cell": a.cell, "n_frames": n, "min_box_height": a.min_box_height})
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
