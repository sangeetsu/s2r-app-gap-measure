#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""2x2 domain matrix: are frozen FM features WORSE on synthetic images than on real ones?

Motivation (2026-08-24): every FM we use was pretrained on real-world imagery
(e.g. the model id facebook/dinov3-vitl16-pretrain-lvd1689m names LVD-1689M), so CARLA
renders are out-of-distribution for the encoder. If synthetic features are degraded,
then our whole pipeline — which fits adapters and heads on CARLA-side features — is
handicapped at the source, and that is a different diagnosis from "the FM lacks the
knowledge".

Test: train the SAME per-box probe on each domain and evaluate on BOTH, with matched
training-set sizes so the comparison is not confounded by sample count.

  train\\test        CARLA        KITTI
  CARLA           in-domain    transfer
  KITTI           transfer     in-domain

Reading the matrix:
  - CARLA->CARLA >> KITTI->KITTI  => synthetic features are, if anything, EASIER
    (likely lower scene diversity), not degraded.
  - KITTI->KITTI >> CARLA->CARLA  => real features are richer; the synthetic side is
    the weak link and the encoder is being asked to work out of distribution.
  - asymmetric transfer tells which direction the representation gap runs.

Usage:
    PY=python
    $PY tools/domain_matrix_probe.py \
        --carla-features ~/datasets/carla_extracted/spatial_features_dinov3.npz \
        --kitti-features ~/datasets/kitti/spatial_features_dinov3.npz \
        --out results/domain_matrix_dinov3.json
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fnn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection
from linear_probe_object_class import bbox_to_feat_region, random_bg_region

CARLA_IMG_HW = (375, 1242)
CLASS_NAMES = {0: "Background", 1: "Car", 2: "Pedestrian", 3: "Cyclist"}


def log(m): print(f"[matrix] {m}", flush=True)


def gather(lookup, ds, hw_fn, rng, bg_per_frame, tag, max_frames=None):
    X, y = [], []
    feat_hw = lookup.target_hw
    D = int(lookup.z_mmap.shape[-1])
    n = len(ds) if max_frames is None else min(max_frames, len(ds))
    for i in range(n):
        _img, t = ds[i]
        rel = t["frame_rel"]
        if not lookup.has(rel): continue
        z = np.array(lookup.z_mmap[lookup.path_to_idx[rel]], dtype=np.float32)
        hw = hw_fn(ds, i)
        gts = []
        for (x1,y1,x2,y2), lab in zip(t["boxes"].numpy(), t["labels"].numpy()):
            r = bbox_to_feat_region((x1,y1,x2,y2), hw, feat_hw); gts.append(r)
            p = z[r[0]:r[2], r[1]:r[3], :].reshape(-1, D)
            if len(p): X.append(p.mean(0)); y.append(int(lab))
        for _ in range(bg_per_frame):
            r = random_bg_region(feat_hw, rng, gts)
            if r is None: continue
            p = z[r[0]:r[2], r[1]:r[3], :].reshape(-1, D)
            if len(p): X.append(p.mean(0)); y.append(0)
    log(f"{tag}: {len(X)} samples from {n} frames")
    return np.asarray(X, np.float32), np.asarray(y, np.int64)


class MLP(nn.Module):
    def __init__(s, d, h=512, c=4):
        super().__init__(); s.f1=nn.Linear(d,h); s.dr=nn.Dropout(0.1); s.f2=nn.Linear(h,c)
    def forward(s,x): return s.f2(s.dr(Fnn.relu(s.f1(x))))


def fit_eval(Xtr, ytr, tests, device, epochs=50, lr=1e-3, bs=256, seed=42):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-6
    m = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(((Xtr-mu)/sd).astype(np.float32)),
                                       torch.from_numpy(ytr)), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        m.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); Fnn.cross_entropy(m(xb), yb).backward(); opt.step()
    m.eval()
    out = {}
    from sklearn.metrics import classification_report
    for name, (Xe, ye) in tests.items():
        with torch.no_grad():
            pr = []
            Xn = torch.from_numpy(((Xe-mu)/sd).astype(np.float32))
            for i in range(0, len(Xn), 8192):
                pr.append(m(Xn[i:i+8192].to(device)).argmax(1).cpu().numpy())
            pr = np.concatenate(pr)
        rep = classification_report(ye, pr, labels=[0,1,2,3],
                                    target_names=[CLASS_NAMES[i] for i in range(4)],
                                    output_dict=True, zero_division=0)
        fg = ye > 0
        out[name] = {"accuracy": float((pr==ye).mean()),
                     "accuracy_foreground_only": float((pr[fg]==ye[fg]).mean()),
                     "macro_f1": rep["macro avg"]["f1-score"],
                     "per_class_f1": {CLASS_NAMES[i]: rep[CLASS_NAMES[i]]["f1-score"] for i in range(4)},
                     "n": int(len(ye))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--carla-features", required=True)
    p.add_argument("--kitti-features", required=True)
    p.add_argument("--carla-root", default=str(Path.home()/"datasets"/"carla_extracted"))
    p.add_argument("--kitti-root", default=str(Path.home()/"datasets"/"kitti"/"extracted"/"training"))
    p.add_argument("--kitti-splits", default=str(Path.home()/"datasets"/"kitti"/"splits.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--bg-per-frame", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lc = SpatialFeatureLookup.from_npz(args.carla_features)
    lk = SpatialFeatureLookup.from_npz(args.kitti_features)
    kimg = Path(args.kitti_root)/"image_2"
    def khw(d,i):
        with Image.open(kimg/d.frame_names[i]) as im: return (im.height, im.width)

    carla = CARLADetection(root=args.carla_root)
    ktr = KITTIDetection(kitti_root=args.kitti_root, split_json=args.kitti_splits, split="train")
    kte = KITTIDetection(kitti_root=args.kitti_root, split_json=args.kitti_splits, split="val")

    Xc, yc = gather(lc, carla, lambda d,i: CARLA_IMG_HW, rng, args.bg_per_frame, "CARLA all")
    Xktr, yktr = gather(lk, ktr, khw, rng, args.bg_per_frame, "KITTI train")
    Xkte, ykte = gather(lk, kte, khw, rng, args.bg_per_frame, "KITTI val")

    # CARLA held-out split
    from sklearn.model_selection import train_test_split
    Xctr, Xcte, yctr, ycte = train_test_split(Xc, yc, test_size=0.2, stratify=yc, random_state=args.seed)

    # match training-set sizes so accuracy is not confounded by sample count
    n = min(len(Xctr), len(Xktr))
    idx_c = rng.choice(len(Xctr), n, replace=False); idx_k = rng.choice(len(Xktr), n, replace=False)
    log(f"matched training size: {n} samples per domain")

    tests = {"CARLA_heldout": (Xcte, ycte), "KITTI_val": (Xkte, ykte)}
    log("training probe on CARLA ...")
    from_carla = fit_eval(Xctr[idx_c], yctr[idx_c], tests, device, seed=args.seed)
    log("training probe on KITTI ...")
    from_kitti = fit_eval(Xktr[idx_k], yktr[idx_k], tests, device, seed=args.seed)

    log("")
    log("            2x2 DOMAIN MATRIX (per-box 4-class accuracy, matched train size)")
    log(f"{'train \\\\ test':<16}{'CARLA held-out':>18}{'KITTI val':>14}")
    log(f"{'CARLA':<16}{from_carla['CARLA_heldout']['accuracy']*100:17.2f}%{from_carla['KITTI_val']['accuracy']*100:13.2f}%")
    log(f"{'KITTI':<16}{from_kitti['CARLA_heldout']['accuracy']*100:17.2f}%{from_kitti['KITTI_val']['accuracy']*100:13.2f}%")
    log("")
    log("            foreground-only (background class excluded)")
    log(f"{'CARLA':<16}{from_carla['CARLA_heldout']['accuracy_foreground_only']*100:17.2f}%{from_carla['KITTI_val']['accuracy_foreground_only']*100:13.2f}%")
    log(f"{'KITTI':<16}{from_kitti['CARLA_heldout']['accuracy_foreground_only']*100:17.2f}%{from_kitti['KITTI_val']['accuracy_foreground_only']*100:13.2f}%")
    ci, ki = from_carla['CARLA_heldout']['accuracy']*100, from_kitti['KITTI_val']['accuracy']*100
    log("")
    log(f"IN-DOMAIN CEILING: synthetic {ci:.2f}%  vs  real {ki:.2f}%   (real − synthetic = {ki-ci:+.2f} pp)")

    json.dump({"features": args.carla_features, "matched_train_n": int(n),
               "probe_trained_on_CARLA": from_carla, "probe_trained_on_KITTI": from_kitti,
               "in_domain_synthetic": ci, "in_domain_real": ki, "real_minus_synthetic_pp": ki-ci,
               "args": vars(args)}, open(args.out, "w"), indent=2)
    log(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
