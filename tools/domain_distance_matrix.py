#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Appearance-gap measurement: candidate distances between domains in frozen FM feature space.

Pivot target (2026-08-28): a calibrated measurement framework that PREDICTS downstream
drop from feature-space distance, rather than a detector that chases mAP.

The CARLA extraction is a clean 2x4 grid — Town01/Town03 x {ClearNoon, OvercastNoon,
SoftRainNoon, ClearNight}, 1000 frames each. That grid separates the two things a gap
metric must not confuse:
  * APPEARANCE  — weather/lighting varied at FIXED town (same geometry, same layout)
  * CONTENT     — town varied at FIXED weather (same renderer/lighting, different scenes)
Plus two real targets (KITTI, Cityscapes) for the sim->real axis.

Computes several candidate metrics, at two readout levels, for every domain pair:

  metrics   fid      Frechet/Gaussian-W2 on (mean, cov)  — the FID construction
            mmd_rbf  maximum mean discrepancy, median-heuristic bandwidth
            coral    covariance (second-moment) distance, as in Deep CORAL
            cos_mean cosine distance between mean embeddings
            domacc   accuracy of a linear domain classifier (P3 showed a single domain
                     direction carries most separability, so this is a strong baseline)
  levels    image    mean over all patch tokens in a frame
            box      mean-pooled inside GT boxes (object-centric)

Later paired with the measured dmAP matrix from per-cell detectors to fit and validate
distance -> drop.

Usage:
    PY=python
    $PY tools/domain_distance_matrix.py --out results/domain_distance_dinov3.json
"""

from __future__ import annotations

import argparse, gc, json, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import linalg

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection
from linear_probe_object_class import bbox_to_feat_region

CARLA_IMG_HW = (375, 1242)


def log(m): print(f"[dist] {m}", flush=True)


# ---------------- metrics ----------------
MAXN = 1200   # OOM guard: this box has 31 GB and was killed twice at 13 GB anon-rss

def _cap(x, seed=0):
    if len(x) <= MAXN: return np.ascontiguousarray(x, dtype=np.float32)
    idx = np.random.default_rng(seed).choice(len(x), MAXN, replace=False)
    return np.ascontiguousarray(x[idx], dtype=np.float32)


def fid(a, b):
    """Frechet distance between Gaussian fits — the FID construction."""
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2*np.trace(covmean))


def mmd_rbf(a, b, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(n, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(n, len(b)), replace=False)]
    z = np.vstack([a, b])
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y> — avoids the (n,n,D) broadcast tensor
    # (3000x3000x1024 floats = 37 GB, which OOM-killed the first run on this 31 GB box)
    sq = (z**2).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2*(z @ z.T), 0)
    sigma2 = np.median(d2[d2 > 0]) + 1e-12          # median heuristic
    k = np.exp(-d2 / sigma2)
    na = len(a)
    kaa = k[:na, :na]; kbb = k[na:, na:]; kab = k[:na, na:]
    return float(kaa.mean() + kbb.mean() - 2*kab.mean())


def coral(a, b):
    """Deep-CORAL second-moment distance, normalized by dimension."""
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    return float(((s1 - s2)**2).sum() / (4 * a.shape[1]**2))


def cos_mean(a, b):
    m1, m2 = a.mean(0), b.mean(0)
    return float(1 - m1.dot(m2)/(np.linalg.norm(m1)*np.linalg.norm(m2) + 1e-12))


def domacc(a, b, seed=0):
    """Linear domain-classifier accuracy (0.5 = indistinguishable, 1.0 = trivially separable)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b), 3000)
    X = np.vstack([a[rng.choice(len(a), n, replace=False)], b[rng.choice(len(b), n, replace=False)]])
    y = np.r_[np.zeros(n), np.ones(n)]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=300).fit((Xtr-mu)/sd, ytr)
    return float(clf.score((Xte-mu)/sd, yte))


# ---------------- feature collection ----------------
def frame_and_box_feats(lookup, rels, boxes_by_rel, img_hw_by_rel, max_frames):
    """Return (image-level mean per frame, box-pooled per GT box)."""
    img_f, box_f = [], []
    fh = lookup.target_hw
    for i, rel in enumerate(rels[:max_frames]):
        if not lookup.has(rel): continue
        z = np.array(lookup.z_mmap[lookup.path_to_idx[rel]], dtype=np.float32)
        img_f.append(z.reshape(-1, z.shape[-1]).mean(0))
        for b in boxes_by_rel.get(rel, []):
            r = bbox_to_feat_region(tuple(b), img_hw_by_rel[rel], fh)
            p = z[r[0]:r[2], r[1]:r[3], :].reshape(-1, z.shape[-1])
            if len(p): box_f.append(p.mean(0))
    return np.array(img_f, np.float32), np.array(box_f, np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--carla-features", default=str(Path.home()/"datasets"/"carla_extracted"/"spatial_features_dinov3.npz"))
    p.add_argument("--kitti-features", default=str(Path.home()/"datasets"/"kitti"/"spatial_features_dinov3.npz"))
    p.add_argument("--cityscapes-features", default=str(Path.home()/"datasets"/"cityscapes"/"spatial_features_dinov3_val.npz"))
    p.add_argument("--carla-root", default=str(Path.home()/"datasets"/"carla_extracted"))
    p.add_argument("--kitti-root", default=str(Path.home()/"datasets"/"kitti"/"extracted"/"training"))
    p.add_argument("--kitti-splits", default=str(Path.home()/"datasets"/"kitti"/"splits.json"))
    p.add_argument("--max-frames", type=int, default=1000)
    p.add_argument("--extra-root", default=None, help="second CARLA root (e.g. /mnt/features/carla_new)")
    p.add_argument("--extra-features", default=None, help="DINOv3 npz for --extra-root")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    domains = {}

    # --- CARLA, split into its 8 cells (town x weather) ---
    roots = [(args.carla_root, args.carla_features)]
    if args.extra_root and args.extra_features and Path(args.extra_features).exists():
        roots.append((args.extra_root, args.extra_features))
    cells = {}
    for root, feats in roots:
        lc = SpatialFeatureLookup.from_npz(feats)
        carla = CARLADetection(root=root)
        for i in range(len(carla)):
            _img, t = carla[i]
            rel = t["frame_rel"]
            cell = rel.split("/")[0] if "/" in rel else "all"
            cells.setdefault(cell, {"rels": [], "boxes": {}, "hw": {}, "lookup": lc})
            cells[cell]["rels"].append(rel)
            cells[cell]["boxes"][rel] = t["boxes"].numpy()
            cells[cell]["hw"][rel] = CARLA_IMG_HW
    log(f"CARLA cells: {sorted(cells)}")
    for cell, d in sorted(cells.items()):
        f, b = frame_and_box_feats(d["lookup"], d["rels"], d["boxes"], d["hw"], args.max_frames)
        domains[cell] = {"image": f, "box": b}
        log(f"  {cell}: {len(f)} frames, {len(b)} boxes")

    # --- KITTI val ---
    lk = SpatialFeatureLookup.from_npz(args.kitti_features)
    kitti = KITTIDetection(kitti_root=args.kitti_root, split_json=args.kitti_splits, split="val")
    kimg = Path(args.kitti_root)/"image_2"
    rels, bx, hw = [], {}, {}
    for i in range(len(kitti)):
        _img, t = kitti[i]
        rel = t["frame_rel"]; rels.append(rel); bx[rel] = t["boxes"].numpy()
        with Image.open(kimg/rel) as im: hw[rel] = (im.height, im.width)
    f, b = frame_and_box_feats(lk, rels, bx, hw, args.max_frames)
    domains["KITTI"] = {"image": f, "box": b}
    log(f"  KITTI: {len(f)} frames, {len(b)} boxes")

    # --- Cityscapes val (image level only; box features need the ALDI json path) ---
    if Path(args.cityscapes_features).exists():
        lcs = SpatialFeatureLookup.from_npz(args.cityscapes_features)
        f = np.array([np.array(lcs.z_mmap[i], dtype=np.float32).reshape(-1, lcs.z_mmap.shape[-1]).mean(0)
                      for i in range(min(args.max_frames, len(lcs)))], np.float32)
        domains["Cityscapes"] = {"image": f, "box": np.zeros((0, f.shape[1]), np.float32)}
        log(f"  Cityscapes: {len(f)} frames (image level only)")

    # --- pairwise distances ---
    names = sorted(domains)
    out = {"domains": names, "metrics": {}}
    for level in ["image", "box"]:
        log("")
        log(f"=== {level.upper()}-LEVEL distances ===")
        res = {}
        for a, b in combinations(names, 2):
            A, B = domains[a][level], domains[b][level]
            if len(A) < 50 or len(B) < 50: continue
            A, B = _cap(A, 1), _cap(B, 2)
            t0 = time.time()
            r = {"fid": fid(A, B), "mmd_rbf": mmd_rbf(A, B), "coral": coral(A, B),
                 "cos_mean": cos_mean(A, B), "domain_acc": domacc(A, B)}
            res[f"{a}||{b}"] = r
            del A, B; gc.collect()
            log(f"  {a:>22} vs {b:<22} fid {r['fid']:9.1f}  mmd {r['mmd_rbf']:.4f}  "
                f"domacc {r['domain_acc']:.3f}  ({time.time()-t0:.0f}s)")
        out["metrics"][level] = res

    # --- the structural check the grid exists for ---
    log("")
    log("=== APPEARANCE vs CONTENT separation (image level, FID) ===")
    im = out["metrics"]["image"]
    def get(a, b):
        return im.get(f"{a}||{b}", im.get(f"{b}||{a}", {})).get("fid")
    same_town_diff_weather, diff_town_same_weather = [], []
    towns = ["Town01", "Town03"]; weathers = ["ClearNoon","OvercastNoon","SoftRainNoon","ClearNight"]
    for t in towns:
        for w1, w2 in combinations(weathers, 2):
            v = get(f"{t}_{w1}", f"{t}_{w2}")
            if v is not None: same_town_diff_weather.append(v)
    for w in weathers:
        v = get(f"Town01_{w}", f"Town03_{w}")
        if v is not None: diff_town_same_weather.append(v)
    if same_town_diff_weather and diff_town_same_weather:
        log(f"  APPEARANCE axis (same town, different weather): mean FID {np.mean(same_town_diff_weather):8.1f}  n={len(same_town_diff_weather)}")
        log(f"  CONTENT    axis (different town, same weather): mean FID {np.mean(diff_town_same_weather):8.1f}  n={len(diff_town_same_weather)}")
        sim2real = [get(c, "KITTI") for c in cells if get(c, "KITTI") is not None]
        if sim2real: log(f"  SIM->REAL  axis (CARLA cell vs KITTI):          mean FID {np.mean(sim2real):8.1f}  n={len(sim2real)}")
        out["axis_summary"] = {"appearance_mean_fid": float(np.mean(same_town_diff_weather)),
                               "content_mean_fid": float(np.mean(diff_town_same_weather)),
                               "sim2real_mean_fid": float(np.mean(sim2real)) if sim2real else None}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    log(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
