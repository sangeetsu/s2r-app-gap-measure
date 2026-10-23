#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DENSE (per-patch) readout of frozen FM features — does the localization signal exist?

Every negative localization result we have so far used a BOX-POOLED readout, which
destroys within-box spatial structure by construction (reviewer tcrE's criticism of the
CoRL probe was correct). This script tests the complementary question with the same
cached features: read the frozen feature grid PER PATCH, FCOS-style, and ask whether an
individual patch feature knows (a) whether it lies on an object, and (b) where that
object's four boundaries are.

Readout is deliberately minimal and position-agnostic: a shared 2-layer MLP applied
independently to each patch vector. No spatial context mixing, no FPN, no attention.
Anything it recovers is information already present in that single frozen patch vector.

Trained on CARLA labels only, evaluated on real KITTI GT — so it also measures whether
the localization signal (if any) is domain-invariant.

Targets per patch, FCOS-style:
  cls  : 0=background, 1=Car, 2=Pedestrian, 3=Cyclist (smallest containing GT box wins)
  ltrb : distances from patch centre to the box's left/top/right/bottom edges,
         normalized by image width/height.

Reported:
  - per-patch classification F1 (does the patch know it is on an object?)
  - decoded-box IoU vs the assigned GT box, for correctly-classified foreground patches
    (does the patch know where the object ENDS?)
  - a distance-only control: predicting ltrb from patch POSITION alone (no FM features),
    which bounds how much of the IoU is trivially inferable from image geometry/priors.

Usage:
    PY=python
    $PY tools/dense_localization_probe.py \
        --carla-features ~/datasets/carla_extracted/spatial_features_dinov3.npz \
        --kitti-features ~/datasets/kitti/spatial_features_dinov3.npz \
        --out results/dense_loc_probe_dinov3.json
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection

CARLA_IMG_HW = (375, 1242)
CLASS_NAMES = {0: "Background", 1: "Car", 2: "Pedestrian", 3: "Cyclist"}


def log(m): print(f"[dense] {m}", flush=True)


def build_targets(boxes, labels, img_hw, feat_hw):
    """Per-patch (cls, ltrb) FCOS-style targets. Smallest containing box wins."""
    H, W = img_hw
    gh, gw = feat_hw
    ys = (np.arange(gh) + 0.5) * (H / gh)
    xs = (np.arange(gw) + 0.5) * (W / gw)
    cx, cy = np.meshgrid(xs, ys)                       # (gh, gw)
    cls = np.zeros((gh, gw), np.int64)
    ltrb = np.zeros((gh, gw, 4), np.float32)
    if len(boxes) == 0:
        return cls, ltrb
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    for i in np.argsort(areas)[::-1]:                  # large first, small overwrite
        x1, y1, x2, y2 = boxes[i]
        inside = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
        if not inside.any():
            continue
        cls[inside] = int(labels[i])
        ltrb[inside, 0] = ((cx - x1) / W)[inside]
        ltrb[inside, 1] = ((cy - y1) / H)[inside]
        ltrb[inside, 2] = ((x2 - cx) / W)[inside]
        ltrb[inside, 3] = ((y2 - cy) / H)[inside]
    return cls, ltrb


def decode_iou(pred_ltrb, true_ltrb, cx, cy, W, H):
    """IoU between the box decoded from predicted ltrb and the true box."""
    def to_box(t):
        return np.stack([cx - t[:, 0]*W, cy - t[:, 1]*H, cx + t[:, 2]*W, cy + t[:, 3]*H], 1)
    pb, tb = to_box(pred_ltrb), to_box(true_ltrb)
    lt = np.maximum(pb[:, :2], tb[:, :2]); rb = np.minimum(pb[:, 2:], tb[:, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[:, 0]*wh[:, 1]
    ap = np.clip(pb[:, 2]-pb[:, 0], 0, None)*np.clip(pb[:, 3]-pb[:, 1], 0, None)
    at = (tb[:, 2]-tb[:, 0])*(tb[:, 3]-tb[:, 1])
    return inter / (ap + at - inter + 1e-9)


class DenseHead(nn.Module):
    def __init__(self, d, hidden=512, ncls=4):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.cls = nn.Linear(hidden, ncls)
        self.reg = nn.Linear(hidden, 4)
    def forward(self, x):
        h = self.trunk(x)
        return self.cls(h), Fnn.softplus(self.reg(h))


def collect(lookup, dataset, img_hw_fn, n_frames, stride, tag, max_patches=None):
    X, Yc, Yr, POS = [], [], [], []
    feat_hw = lookup.target_hw
    t0 = time.time()
    used = 0
    for idx in range(len(dataset)):
        if used >= n_frames: break
        _img, tgt = dataset[idx]
        rel = tgt["frame_rel"]
        if not lookup.has(rel): continue
        img_hw = img_hw_fn(dataset, idx)
        z = np.array(lookup.z_mmap[lookup.path_to_idx[rel]], dtype=np.float32)
        cls, ltrb = build_targets(tgt["boxes"].numpy(), tgt["labels"].numpy(), img_hw, feat_hw)
        gh, gw = feat_hw
        ys = (np.arange(gh) + 0.5) * (img_hw[0] / gh)
        xs = (np.arange(gw) + 0.5) * (img_hw[1] / gw)
        cx, cy = np.meshgrid(xs, ys)
        # keep all foreground patches + a stride-sampled subset of background
        fg = cls > 0
        bg = ~fg
        bg_keep = np.zeros_like(bg); bg_keep.flat[::stride] = True; bg = bg & bg_keep
        sel = fg | bg
        X.append(z[sel].astype(np.float16)); Yc.append(cls[sel]); Yr.append(ltrb[sel])
        POS.append(np.stack([cx[sel]/img_hw[1], cy[sel]/img_hw[0]], 1).astype(np.float32))
        used += 1
        if used % 1000 == 0: log(f"  {tag} {used}/{n_frames} frames, {sum(len(x) for x in X)} patches")
    X = np.concatenate(X); Yc = np.concatenate(Yc); Yr = np.concatenate(Yr); POS = np.concatenate(POS)
    if max_patches and len(X) > max_patches:
        # keep every foreground patch, subsample background, to respect the RAM budget
        rng = np.random.default_rng(0)
        fg = np.where(Yc > 0)[0]; bg = np.where(Yc == 0)[0]
        n_bg = max(0, max_patches - len(fg))
        keep = np.concatenate([fg, rng.choice(bg, min(n_bg, len(bg)), replace=False)])
        rng.shuffle(keep)
        X, Yc, Yr, POS = X[keep], Yc[keep], Yr[keep], POS[keep]
        log(f"{tag}: subsampled to {len(X)} patches ({len(fg)} fg kept in full)")
    log(f"{tag}: {used} frames, {len(X)} patches in {time.time()-t0:.0f}s")
    return X, Yc, Yr, POS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--carla-features", required=True)
    p.add_argument("--kitti-features", required=True)
    p.add_argument("--carla-root", default=str(Path.home()/"datasets"/"carla_extracted"))
    p.add_argument("--kitti-root", default=str(Path.home()/"datasets"/"kitti"/"extracted"/"training"))
    p.add_argument("--kitti-splits", default=str(Path.home()/"datasets"/"kitti"/"splits.json"))
    p.add_argument("--out", required=True)
    p.add_argument("--train-frames", type=int, default=4000)
    p.add_argument("--bg-stride", type=int, default=17)
    p.add_argument("--max-train-patches", type=int, default=400000,
                   help="RAM cap; all foreground patches are kept, background is subsampled.")
    p.add_argument("--max-test-patches", type=int, default=250000)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    lc = SpatialFeatureLookup.from_npz(args.carla_features)
    lk = SpatialFeatureLookup.from_npz(args.kitti_features)
    D = int(lc.z_mmap.shape[-1])
    log(f"feature dim {D}, grid {lc.target_hw}")

    carla = CARLADetection(root=args.carla_root)
    kitti = KITTIDetection(kitti_root=args.kitti_root, split_json=args.kitti_splits, split="val")
    kimg = Path(args.kitti_root)/"image_2"

    Xtr, Yctr, Yrtr, Ptr = collect(lc, carla, lambda d, i: CARLA_IMG_HW,
                                   args.train_frames, args.bg_stride, "CARLA",
                                   max_patches=args.max_train_patches)
    def kitti_hw(d, i):
        with Image.open(kimg/d.frame_names[i]) as im: return (im.height, im.width)
    Xte, Ycte, Yrte, Pte = collect(lk, kitti, kitti_hw, len(kitti), args.bg_stride, "KITTI",
                                   max_patches=args.max_test_patches)

    mu = Xtr[:200000].astype(np.float32).mean(0); sd = Xtr[:200000].astype(np.float32).std(0) + 1e-6
    model = DenseHead(D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    mu_t = torch.from_numpy(mu.astype(np.float32)).to(device)
    sd_t = torch.from_numpy(sd.astype(np.float32)).to(device)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr),  # fp16, normalized per batch
                                        torch.from_numpy(Yctr), torch.from_numpy(Yrtr))
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    log(f"training dense head on {len(Xtr)} CARLA patches for {args.epochs} epochs")
    for ep in range(args.epochs):
        model.train()
        for xb, cb, rb in dl:
            xb = (xb.to(device).float() - mu_t) / sd_t
            cb, rb = cb.to(device), rb.to(device)
            opt.zero_grad()
            lc_, lr_ = model(xb)
            loss = Fnn.cross_entropy(lc_, cb)
            fg = cb > 0
            if fg.any():
                loss = loss + 5.0 * Fnn.l1_loss(lr_[fg], rb[fg])
            loss.backward(); opt.step()
    model.eval()

    def evaluate(X, Yc, Yr, P, hw_note, tag):
        with torch.no_grad():
            pc, pr = [], []
            Xn = torch.from_numpy(X)
            for i in range(0, len(Xn), 16384):
                a, b = model((Xn[i:i+16384].to(device).float() - mu_t) / sd_t)
                pc.append(a.argmax(1).cpu().numpy()); pr.append(b.cpu().numpy())
            pc, pr = np.concatenate(pc), np.concatenate(pr)
        from sklearn.metrics import classification_report
        rep = classification_report(Yc, pc, labels=[0,1,2,3],
                                    target_names=[CLASS_NAMES[i] for i in range(4)],
                                    output_dict=True, zero_division=0)
        fg = Yc > 0
        correct_fg = fg & (pc == Yc)
        # decode IoU using each patch's own centre; W,H folded into normalized ltrb
        res = {"patch_accuracy": float((pc == Yc).mean()),
               "fg_patch_recall": float((pc[fg] > 0).mean()) if fg.any() else None,
               "per_class_f1": {CLASS_NAMES[i]: rep[CLASS_NAMES[i]]["f1-score"] for i in range(4)},
               "n_patches": int(len(Yc)), "n_fg": int(fg.sum())}
        for name, mask in [("all_fg", fg), ("correctly_classified_fg", correct_fg)]:
            if mask.sum() < 10: continue
            # normalized coords: treat W=H=1 so ltrb are already fractions
            iou = decode_iou(pr[mask], Yr[mask], P[mask,0], P[mask,1], 1.0, 1.0)
            res[f"box_iou_{name}"] = {"mean": float(iou.mean()),
                                      "median": float(np.median(iou)),
                                      "frac_iou_ge_0.5": float((iou >= 0.5).mean()),
                                      "frac_iou_ge_0.75": float((iou >= 0.75).mean()),
                                      "n": int(mask.sum())}
        log(f"--- {tag} ({hw_note}): patch acc {res['patch_accuracy']:.4f}, "
            f"fg recall {res['fg_patch_recall']:.4f}")
        for k in ["all_fg", "correctly_classified_fg"]:
            if f"box_iou_{k}" in res:
                b = res[f"box_iou_{k}"]
                log(f"    decoded-box IoU [{k}]: mean {b['mean']:.4f} median {b['median']:.4f} "
                    f"@0.5 {b['frac_iou_ge_0.5']:.3f} @0.75 {b['frac_iou_ge_0.75']:.3f} (n={b['n']})")
        return res

    # position-only control: how much of the IoU is inferable without FM features at all?
    log("training position-only control (no FM features)")
    ctrl = DenseHead(2).to(device)
    copt = torch.optim.Adam(ctrl.parameters(), lr=args.lr)
    cds = torch.utils.data.TensorDataset(torch.from_numpy(Ptr.astype(np.float32)),
                                         torch.from_numpy(Yctr), torch.from_numpy(Yrtr))
    cdl = torch.utils.data.DataLoader(cds, batch_size=args.batch_size, shuffle=True)
    for ep in range(args.epochs):
        ctrl.train()
        for xb, cb, rb in cdl:
            xb, cb, rb = xb.to(device).float(), cb.to(device), rb.to(device)
            copt.zero_grad()
            a, b = ctrl(xb)
            l = Fnn.cross_entropy(a, cb)
            fg = cb > 0
            if fg.any(): l = l + 5.0*Fnn.l1_loss(b[fg], rb[fg])
            l.backward(); copt.step()
    ctrl.eval()
    with torch.no_grad():
        cb_ = []
        Pt = torch.from_numpy(Pte)
        for i in range(0, len(Pt), 16384):
            _, b = ctrl(Pt[i:i+16384].to(device)); cb_.append(b.cpu().numpy())
        cb_ = np.concatenate(cb_)
    fgte = Ycte > 0
    ctrl_iou = decode_iou(cb_[fgte], Yrte[fgte], Pte[fgte,0], Pte[fgte,1], 1.0, 1.0)
    log(f"--- POSITION-ONLY CONTROL on KITTI: decoded-box IoU mean {ctrl_iou.mean():.4f} "
        f"median {np.median(ctrl_iou):.4f} @0.5 {(ctrl_iou>=0.5).mean():.3f}")

    out = {"features": args.carla_features, "feat_dim": D, "grid": list(lc.target_hw),
           "carla_train": evaluate(Xtr, Yctr, Yrtr, Ptr, "in-domain fit", "CARLA train"),
           "kitti_transfer": evaluate(Xte, Ycte, Yrte, Pte, "TRANSFER", "KITTI val"),
           "position_only_control_kitti": {"box_iou_all_fg": {
               "mean": float(ctrl_iou.mean()), "median": float(np.median(ctrl_iou)),
               "frac_iou_ge_0.5": float((ctrl_iou >= 0.5).mean()), "n": int(fgte.sum())}},
           "args": vars(args)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    log(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
