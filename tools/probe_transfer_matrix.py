#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TASK-CONDITIONAL gap: probe-transfer matrix over the CARLA grid (+KITTI).

Distribution distances (FID/MMD/CORAL) over frozen features predicted downstream drop only
weakly in-family and not at all for sim->real (ablation_log §15.19). The hypothesis tested
here: a gap measured THROUGH the task readout — train a 4-class box probe on source cell S,
test it on target T — tracks the detector's drop better, because it is V-usable information
under the task's function family rather than a geometry of the feature cloud.

For every source cell S: fit the probe (same protocol as probe_transfer_eval.py), record
in-domain held-out accuracy, then evaluate on every other cell and on KITTI val.
dIV(S->T) := acc_in(S) - acc(S->T)   (foreground-only and 4-class both saved).

Output pairs with results/calib/dmap_matrix.json for the predictor comparison.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT/"src")); sys.path.insert(0, str(REPO_ROOT/"tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection
from domain_matrix_probe import gather, fit_eval, CARLA_IMG_HW
from sklearn.model_selection import train_test_split

CELLS = ["Town01_ClearNoon","Town01_OvercastNoon","Town01_SoftRainNoon","Town01_ClearNight",
         "Town03_ClearNoon","Town03_OvercastNoon","Town03_SoftRainNoon","Town03_ClearNight"]
def log(m): print(f"[ptm] {m}", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--carla-features", default=str(Path.home()/"datasets/carla_extracted/spatial_features_dinov3.npz"))
    p.add_argument("--kitti-features", default=str(Path.home()/"datasets/kitti/spatial_features_dinov3.npz"))
    p.add_argument("--carla-root", default=str(Path.home()/"datasets/carla_extracted"))
    p.add_argument("--kitti-root", default=str(Path.home()/"datasets/kitti/extracted/training"))
    p.add_argument("--kitti-splits", default=str(Path.home()/"datasets/kitti/splits.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--extra-root", default=None, help="second CARLA root (e.g. /mnt/features/carla_new)")
    p.add_argument("--extra-features", default=None, help="DINOv3 npz for --extra-root")
    p.add_argument("--anchored", action="store_true", help="probe trained on frames 0:800, in-domain acc on 800:1000")
    p.add_argument("--out", default="results/calib/probe_transfer_matrix_dinov3.json")
    a = p.parse_args()
    rng = np.random.default_rng(a.seed); dev = torch.device("cuda")
    lc = SpatialFeatureLookup.from_npz(a.carla_features); lk = SpatialFeatureLookup.from_npz(a.kitti_features)

    feats = {}; roots = [(a.carla_root, lc, list(CELLS))]   # copy: CELLS is extended below
    if a.extra_root and a.extra_features:
        ex = sorted(d.name for d in Path(a.extra_root).iterdir() if d.is_dir() and len(list(d.glob("frame_*.jpg")))>=1000)
        roots.append((a.extra_root, SpatialFeatureLookup.from_npz(a.extra_features), ex)); CELLS.extend(ex)
    holdout = {}
    for root, lk_, cl in roots:
        for c in cl:
            ds = CARLADetection(root=root, cells=[c])
            if a.anchored:
                import torch as _t
                X, y = gather(lk_, _t.utils.data.Subset(ds, range(0, 800)), lambda d,i: CARLA_IMG_HW, rng, 4, c+" 0:800")
                holdout[c] = gather(lk_, _t.utils.data.Subset(ds, range(800, len(ds))), lambda d,i: CARLA_IMG_HW, rng, 4, c+" 800:")
            else:
                X, y = gather(lk_, ds, lambda d,i: CARLA_IMG_HW, rng, 4, c)
            feats[c] = (X, y)
    kimg = Path(a.kitti_root)/"image_2"
    def khw(d,i):
        with Image.open(kimg/d.frame_names[i]) as im: return (im.height, im.width)
    kv = KITTIDetection(kitti_root=a.kitti_root, split_json=a.kitti_splits, split="val")
    feats["KITTI"] = gather(lk, kv, khw, rng, 4, "KITTI val")
    # second real target: Cityscapes val with ALDI boxes (rels "city/base.png" match the cache)
    csf = Path.home()/"datasets/cityscapes/spatial_features_dinov3_val.npz"
    ann = REPO_ROOT/"nvme/datasets/aldi_labels/cityscapes_val_instances.json"
    if csf.exists() and ann.exists():
        from s2rgap.data.cityscapes_coco import CityscapesCOCO
        lcs = SpatialFeatureLookup.from_npz(csf)
        cs = CityscapesCOCO(Path.home()/"datasets/cityscapes/extracted/leftImg8bit", ann, split="val")
        feats["Cityscapes"] = gather(lcs, cs, lambda d,i: (1024, 2048), rng, 4, "Cityscapes val")

    out = {"cells": CELLS, "matrix": {}}
    for S in CELLS:
        X, y = feats[S]
        if a.anchored:
            Xtr, ytr = X, y; tests = {"__in__": holdout[S]}
            tests.update({T: holdout[T] for T in CELLS if T != S}); tests.update({T: feats[T] for T in feats if T not in CELLS})
        else:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=a.seed)
            tests = {"__in__": (Xte, yte)}; tests.update({T: feats[T] for T in list(feats) if T != S})
        t0 = time.time()
        res = fit_eval(Xtr, ytr, tests, dev, seed=a.seed)
        acc_in, fg_in = res["__in__"]["accuracy"], res["__in__"]["accuracy_foreground_only"]
        row = {"in_domain": {"acc": acc_in, "acc_fg": fg_in}}
        for T, r in res.items():
            if T == "__in__": continue
            row[T] = {"acc": r["accuracy"], "acc_fg": r["accuracy_foreground_only"],
                      "dIV": acc_in - r["accuracy"], "dIV_fg": fg_in - r["accuracy_foreground_only"],
                      "macro_f1": r["macro_f1"]}
        out["matrix"][S] = row
        log(f"{S:<22} in {acc_in*100:5.1f}  " + "  ".join(f"{T.split('_')[-1][:3] if T!='KITTI' else 'KIT'}:{row[T]['acc']*100:4.1f}"
            for T in CELLS+["KITTI"] if T != S) + (f"  CS:{row['Cityscapes']['acc']*100:4.1f}" if "Cityscapes" in row else "") + f"  ({time.time()-t0:.0f}s)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    log(f"saved -> {a.out}")
if __name__ == "__main__": raise SystemExit(main())
