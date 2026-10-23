#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Linear-probe diagnostic: do cached foundation features carry per-object class info?

This is the §10.4 / §10.7 diagnostic from ``plan_2026-05-14.md``. We extract a
single feature vector per CARLA bounding box by mean-pooling the cached spatial
feature map over the box-overlapping region (no ROIAlign refinement; this is a
*lower bound* on what an actual detection head could extract). We also draw
random background patches. Then we train a **linear** classifier (one matrix +
softmax, no nonlinearity) on a 80/20 split and report per-class accuracy.

Decision rule (per plan):
  - ≥ 70% test accuracy: features carry rich object-class info -> reverse
    distillation should work cleanly. Proceed to full method.
  - 30-70%: partial signal -> proceed with caution / shorter time-box.
  - < 30%: features genuinely lack object discrimination -> accept Path C.

Usage:
    PY=python
    $PY tools/linear_probe_object_class.py \
        --spatial-features ~/datasets/carla_extracted/spatial_features.npz \
        --carla-root ~/datasets/carla_extracted \
        --out results/linear_probe_s2rgap_postmerge.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection


# Carla image dimensions are fixed at extraction time.
DEFAULT_IMG_HW = (375, 1242)


def log(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def bbox_to_feat_region(
    bbox: tuple[float, float, float, float],
    img_hw: tuple[int, int],
    feat_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map an original-pixel bbox (x1,y1,x2,y2) to feature-grid (i1,j1,i2,j2)."""
    x1, y1, x2, y2 = bbox
    H_img, W_img = img_hw
    H_feat, W_feat = feat_hw
    s_w = W_img / W_feat
    s_h = H_img / H_feat
    i1 = max(0, int(np.floor(y1 / s_h)))
    j1 = max(0, int(np.floor(x1 / s_w)))
    i2 = min(H_feat, int(np.ceil(y2 / s_h)))
    j2 = min(W_feat, int(np.ceil(x2 / s_w)))
    # Ensure at least 1 cell of coverage.
    if i2 <= i1:
        i2 = i1 + 1
    if j2 <= j1:
        j2 = j1 + 1
    return i1, j1, i2, j2


def random_bg_region(
    feat_hw: tuple[int, int],
    rng: np.random.Generator,
    gt_regions: list[tuple[int, int, int, int]],
    max_tries: int = 20,
    min_size: int = 1,
    max_size: int = 3,
) -> tuple[int, int, int, int] | None:
    """Pick a random feature region that does not overlap any GT region."""
    H, W = feat_hw
    for _ in range(max_tries):
        h_box = int(rng.integers(min_size, max(min_size + 1, max_size + 1)))
        w_box = int(rng.integers(min_size, max(min_size + 1, max_size + 1)))
        i1 = int(rng.integers(0, max(1, H - h_box + 1)))
        j1 = int(rng.integers(0, max(1, W - w_box + 1)))
        i2, j2 = i1 + h_box, j1 + w_box
        overlap = False
        for gi1, gj1, gi2, gj2 in gt_regions:
            if not (i2 <= gi1 or gi2 <= i1 or j2 <= gj1 or gj2 <= j1):
                overlap = True
                break
        if not overlap:
            return i1, j1, i2, j2
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--spatial-features", required=True, help="path to the .npz cache")
    p.add_argument("--carla-root", default=str(Path.home() / "datasets" / "carla_extracted"))
    p.add_argument("--out", required=True, help="JSON output path")
    p.add_argument(
        "--max-frames", type=int, default=8000,
        help="cap on how many CARLA frames to use",
    )
    p.add_argument(
        "--bg-per-frame", type=int, default=4,
        help="number of random background samples drawn per frame",
    )
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--reduce", choices=["mean", "max"], default="mean",
        help="how to pool the spatial feature region to a single vector",
    )
    p.add_argument(
        "--max-iter", type=int, default=200,
        help="LogisticRegression max_iter; bump if it warns about convergence",
    )
    p.add_argument(
        "--C", type=float, default=1.0,
        help="Inverse L2 regularization strength for LogReg",
    )
    p.add_argument(
        "--probe-type",
        choices=["linear", "mlp"],
        default="linear",
        help="linear = sklearn LogReg (original behavior). mlp = 2-layer PyTorch MLP "
             "(D -> mlp-hidden -> n_classes, ReLU + dropout). Gives the non-linear upper "
             "bound on per-bbox class info — the gap between linear and mlp quantifies "
             "non-linearly-entangled information per Tier-1 of publishability_plan.md.",
    )
    p.add_argument("--mlp-hidden", type=int, default=512,
                   help="Hidden size for --probe-type mlp.")
    p.add_argument("--mlp-epochs", type=int, default=50,
                   help="Epochs to train the MLP probe.")
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-batch-size", type=int, default=256)
    p.add_argument("--mlp-dropout", type=float, default=0.1)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    log(f"loading cached spatial features from {args.spatial_features}")
    lookup = SpatialFeatureLookup.from_npz(args.spatial_features)
    log(
        f"  {len(lookup)} frames, target_hw={lookup.target_hw}, "
        f"D = {lookup.z_mmap.shape[-1]}"
    )

    log(f"loading CARLA detection labels from {args.carla_root}")
    base = CARLADetection(root=args.carla_root)
    log(f"  {len(base)} frames")

    feat_hw = lookup.target_hw  # (H_feat, W_feat)
    D = int(lookup.z_mmap.shape[-1])

    # Class label mapping (matches CARLA_CLASSES / KITTI_CLASSES):
    # 0 = background, 1 = Car, 2 = Pedestrian, 3 = Cyclist.
    class_names = {0: "Background", 1: "Car", 2: "Pedestrian", 3: "Cyclist"}

    n_frames = min(args.max_frames, len(base))
    log(f"extracting per-bbox feature vectors from {n_frames} frames ...")

    X: list[np.ndarray] = []
    y: list[int] = []

    t0 = time.time()
    for idx in range(n_frames):
        _img_tensor, target = base[idx]
        rel = target["frame_rel"]
        if not lookup.has(rel):
            continue
        # Spatial feat: (H, W, D) fp16 from mmap. Copy to fp32 once per frame.
        z = np.array(lookup.z_mmap[lookup.path_to_idx[rel]], dtype=np.float32)
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()
        gt_regions = []
        for (x1, y1, x2, y2), lab in zip(boxes, labels):
            i1, j1, i2, j2 = bbox_to_feat_region(
                (x1, y1, x2, y2), DEFAULT_IMG_HW, feat_hw
            )
            gt_regions.append((i1, j1, i2, j2))
            patch = z[i1:i2, j1:j2, :].reshape(-1, D)
            if patch.shape[0] == 0:
                continue
            if args.reduce == "mean":
                vec = patch.mean(axis=0)
            else:
                vec = patch.max(axis=0)
            X.append(vec)
            y.append(int(lab))

        # Background samples (class 0).
        for _ in range(args.bg_per_frame):
            reg = random_bg_region(feat_hw, rng, gt_regions)
            if reg is None:
                continue
            i1, j1, i2, j2 = reg
            patch = z[i1:i2, j1:j2, :].reshape(-1, D)
            if patch.shape[0] == 0:
                continue
            vec = patch.mean(axis=0) if args.reduce == "mean" else patch.max(axis=0)
            X.append(vec)
            y.append(0)

        if (idx + 1) % 1000 == 0:
            log(f"  processed {idx+1}/{n_frames} frames, samples so far: {len(X)}")

    t_extract = time.time() - t0
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    log(f"extracted {len(X_arr)} samples in {t_extract:.1f}s")
    cnts = {int(k): int(v) for k, v in zip(*np.unique(y_arr, return_counts=True))}
    log(f"  class distribution: {cnts}")

    # 80/20 split, stratified.
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=args.test_frac, stratify=y_arr, random_state=args.seed
    )
    log(f"split: {len(X_train)} train / {len(X_test)} test")

    if args.probe_type == "linear":
        from sklearn.linear_model import LogisticRegression
        log(f"training LINEAR classifier (LogReg, C={args.C}, max_iter={args.max_iter}) ...")
        t0 = time.time()
        clf = LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            solver="lbfgs",
            n_jobs=-1,
            random_state=args.seed,
        )
        clf.fit(X_train, y_train)
        t_train = time.time() - t0
        log(f"trained in {t_train:.1f}s")
        acc_train = float(clf.score(X_train, y_train))
        acc_test = float(clf.score(X_test, y_test))
        y_pred = clf.predict(X_test)
    else:
        # MLP probe (PyTorch) — 2-layer with ReLU + dropout.
        import torch
        import torch.nn as nn
        import torch.nn.functional as Fnn
        from torch.utils.data import DataLoader, TensorDataset
        log(f"training MLP probe (D={D} -> hidden={args.mlp_hidden} -> n_classes=4, dropout={args.mlp_dropout}, "
            f"lr={args.mlp_lr}, batch_size={args.mlp_batch_size}, epochs={args.mlp_epochs}) ...")
        torch.manual_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_classes = 4

        # Standardize features (mean/std on train) for stable MLP training.
        feat_mean = X_train.mean(axis=0)
        feat_std = X_train.std(axis=0) + 1e-6
        X_train_n = ((X_train - feat_mean) / feat_std).astype(np.float32)
        X_test_n = ((X_test - feat_mean) / feat_std).astype(np.float32)

        class MLPProbe(nn.Module):
            def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, hidden)
                self.drop = nn.Dropout(dropout)
                self.fc2 = nn.Linear(hidden, n_classes)
            def forward(self, x):
                h = Fnn.relu(self.fc1(x))
                h = self.drop(h)
                return self.fc2(h)

        model = MLPProbe(D, args.mlp_hidden, n_classes, args.mlp_dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.mlp_lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        X_train_t = torch.from_numpy(X_train_n)
        y_train_t = torch.from_numpy(y_train.astype(np.int64))
        X_test_t = torch.from_numpy(X_test_n).to(device)
        y_test_t = torch.from_numpy(y_test.astype(np.int64)).to(device)
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=args.mlp_batch_size, shuffle=True, num_workers=0,
        )

        t0 = time.time()
        for epoch in range(args.mlp_epochs):
            model.train()
            tot_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                tot_loss += float(loss.detach()); n_batches += 1
            if (epoch + 1) % 10 == 0 or epoch == args.mlp_epochs - 1:
                model.eval()
                with torch.no_grad():
                    logits_te = model(X_test_t)
                    pred_te = logits_te.argmax(dim=1)
                    acc_te = float((pred_te == y_test_t).float().mean())
                log(f"  epoch {epoch+1}/{args.mlp_epochs}: train_loss={tot_loss/max(1,n_batches):.4f}, test_acc={acc_te:.4f}")
        t_train = time.time() - t0
        log(f"trained MLP in {t_train:.1f}s")

        # Final eval.
        model.eval()
        with torch.no_grad():
            logits_te = model(X_test_t)
            pred_te = logits_te.argmax(dim=1)
            y_pred = pred_te.cpu().numpy()
            acc_test = float((pred_te == y_test_t).float().mean())
            # train acc on full train set
            X_train_full_t = torch.from_numpy(X_train_n).to(device)
            y_train_full_t = torch.from_numpy(y_train.astype(np.int64)).to(device)
            logits_tr = model(X_train_full_t)
            acc_train = float((logits_tr.argmax(dim=1) == y_train_full_t).float().mean())

    log(f"train acc = {acc_train:.4f}, test acc = {acc_test:.4f}")

    # Per-class accuracy
    report = classification_report(
        y_test, y_pred,
        labels=[0, 1, 2, 3],
        target_names=[class_names[i] for i in [0, 1, 2, 3]],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]).tolist()

    log("Per-class precision/recall/f1 (test):")
    for cname in [class_names[i] for i in [0, 1, 2, 3]]:
        r = report[cname]
        log(f"  {cname:12s}: prec={r['precision']:.4f} rec={r['recall']:.4f} f1={r['f1-score']:.4f} support={int(r['support'])}")

    log(f"Confusion matrix (rows=true, cols=pred, order={[class_names[i] for i in [0,1,2,3]]}):")
    for row in cm:
        log(f"  {row}")

    # Decision interpretation
    if acc_test >= 0.70:
        verdict = "PASS_STRONG: linear probe ≥ 70% — features encode rich object-class info; reverse distillation should work cleanly"
    elif acc_test >= 0.30:
        verdict = "PASS_WEAK: linear probe 30-70% — partial signal; reverse distillation may help"
    else:
        verdict = "FAIL: linear probe < 30% — features lack object discrimination; reverse distillation unlikely to rescue"
    log(f"VERDICT: {verdict}")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "spatial_features": str(args.spatial_features),
        "probe_type": args.probe_type,
        "n_frames": n_frames,
        "n_samples": int(len(X_arr)),
        "feat_hw": list(feat_hw),
        "feat_dim": D,
        "class_distribution": cnts,
        "split": {"train": int(len(X_train)), "test": int(len(X_test))},
        "train_accuracy": acc_train,
        "test_accuracy": acc_test,
        "per_class_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": [class_names[i] for i in [0, 1, 2, 3]],
        "verdict": verdict,
        "args": vars(args),
        "wall_seconds": {"extract": t_extract, "train": t_train},
    }
    with out.open("w") as f:
        json.dump(blob, f, indent=2)
    log(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
