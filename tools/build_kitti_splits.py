#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Produce reproducible 80/20 train/val splits over the 7481 KITTI training
frames, persist filenames to JSON. KITTI's test set has no public 2D labels,
so this lifts the standard "use training split for val" practice.

Usage:
    python tools/build_kitti_splits.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kitti-root",
        default=str(Path.home() / "datasets" / "kitti" / "extracted" / "training"),
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=str(Path.home() / "datasets" / "kitti" / "splits.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.kitti_root).expanduser()
    img_dir = root / "image_2"
    lbl_dir = root / "label_2"
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"[splits] FAIL: missing {img_dir} or {lbl_dir}")
        return 1

    images = sorted(p.name for p in img_dir.glob("*.png"))
    labels_present = {p.stem for p in lbl_dir.glob("*.txt")}
    frames = [name for name in images if Path(name).stem in labels_present]
    print(f"[splits] {len(images)} images, {len(labels_present)} labels, {len(frames)} paired")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(frames))
    n_train = int(round(args.train_frac * len(frames)))
    train = sorted(frames[i] for i in perm[:n_train])
    val = sorted(frames[i] for i in perm[n_train:])

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "kitti_root": str(root),
                "train_frac": args.train_frac,
                "seed": args.seed,
                "n_train": len(train),
                "n_val": len(val),
                "train": train,
                "val": val,
            },
            indent=2,
        )
    )
    print(f"[splits] train={len(train)} val={len(val)} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
