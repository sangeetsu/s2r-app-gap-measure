#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract DINOv2 spatial feature maps for an entire dataset.

Uses ``facebook/dinov2-with-registers-large`` by default (ViT-L/14 + 4 register
tokens, Darcet et al. ICLR 2024 — the current canonical "DINOv2"). Standard
DINOv2 protocol: resize to 518×518 (the model's native pretraining resolution),
normalize with ImageNet stats, extract last-layer patch tokens, drop the CLS +
4 register tokens, reshape to (37, 37, 1024).

Saved schema mirrors ``tools/extract_alpamayo_feature_maps.py`` so the cached
features can be consumed by the same downstream code (``SpatialFeatureLookup``,
``tools/linear_probe_object_class.py``, ``tools/train_alpamayo_labeler.py``,
``tools/train_reverse_distill.py``) by changing only the ``--spatial-features``
+ ``--in-channels`` flags.

Usage:
    PY=python
    $PY tools/extract_dinov2_features.py --kind carla \\
        --root ~/datasets/carla_extracted \\
        --out ~/datasets/carla_extracted/spatial_features_dinov2.npz

    $PY tools/extract_dinov2_features.py --kind kitti \\
        --root ~/datasets/kitti/extracted/training/image_2 \\
        --out ~/datasets/kitti/spatial_features_dinov2.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.path.insert(0, str(REPO_ROOT / "tools"))
from frame_index import gather_carla, gather_cityscapes, gather_kitti


DEFAULT_MODEL_ID = "facebook/dinov2-with-registers-large"
DEFAULT_IMG_SIZE = 518
DEFAULT_PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_preprocess(img_h: int, img_w: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(
            (img_h, img_w),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["kitti", "cityscapes", "carla"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help="HF model id. Default: facebook/dinov2-with-registers-large (ViT-L/14 + 4 registers).")
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE,
                   help="Square input size fed to DINOv2 if --img-h/--img-w not set. Multiple of patch_size (14). "
                        "Default 518 = DINOv2 native pretraining resolution; produces a 37×37 grid. "
                        "Use rectangular --img-h/--img-w instead for aspect-mismatched datasets (e.g. KITTI/CARLA 3.31:1).")
    p.add_argument("--img-h", type=int, default=None,
                   help="Optional rectangular input height (must be multiple of patch_size=14). "
                        "When set together with --img-w, overrides --img-size and feeds rectangular input "
                        "to DINOv2 (position embeddings are interpolated automatically by transformers). "
                        "Recommended for non-square images to preserve aspect ratio. "
                        "Example for KITTI/CARLA (1242x375): --img-h 350 --img-w 1218 → 25×87 grid.")
    p.add_argument("--img-w", type=int, default=None,
                   help="Optional rectangular input width (must be multiple of patch_size=14). "
                        "See --img-h.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="cap N for debugging")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--compress", action="store_true",
                   help="use np.savez_compressed (slower, smaller)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.kind == "kitti":
        paths, rels = gather_kitti(root)
        cells = [""] * len(paths)
    elif args.kind == "cityscapes":
        paths, rels = gather_cityscapes(root)
        cells = [""] * len(paths)
    else:
        paths, rels, cells = gather_carla(root)

    if not paths:
        print(f"[dinov2-extract] FAIL: no frames found at {root}")
        return 1

    if args.limit:
        paths = paths[: args.limit]
        rels = rels[: args.limit]
        cells = cells[: args.limit]

    print(f"[dinov2-extract] kind={args.kind} root={root} -> {len(paths)} frames")

    if not torch.cuda.is_available():
        print("[dinov2-extract] FAIL: no CUDA")
        return 1
    device = torch.device("cuda")

    print(f"[dinov2-extract] loading {args.model_id} (fp16)...")
    t0 = time.time()
    model = AutoModel.from_pretrained(args.model_id, dtype=torch.float16).to(device).eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    n_registers = getattr(model.config, "num_register_tokens", 0)
    cls_offset = 1 + n_registers
    hidden = model.config.hidden_size
    patch_size = getattr(model.config, "patch_size", DEFAULT_PATCH_SIZE)
    if args.img_h is not None and args.img_w is not None:
        img_h, img_w = args.img_h, args.img_w
    else:
        img_h = img_w = args.img_size
    for name, v in (("--img-h", img_h), ("--img-w", img_w)):
        if v % patch_size != 0:
            print(f"[dinov2-extract] FAIL: {name}={v} not divisible by patch_size {patch_size}")
            return 1
    grid_h = img_h // patch_size
    grid_w = img_w // patch_size
    target_hw = (grid_h, grid_w)
    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[dinov2-extract] loaded in {time.time()-t0:.1f}s | hidden={hidden}, "
          f"patch_size={patch_size}, img_hw=({img_h}, {img_w}), grid={grid_h}x{grid_w}, "
          f"n_registers={n_registers}, cls_offset={cls_offset}, "
          f"n_params={n_params_m:.1f}M")

    preprocess = make_preprocess(img_h, img_w)

    # Write features directly to the SpatialFeatureLookup sidecar (.z_spatial.npy)
    # via np.lib.format.open_memmap — avoids a 30+ GB RAM peak that np.empty(N,H,W,D)
    # would otherwise force when N×H×W×D×2 doesn't fit. The sidecar is a real .npy
    # (proper header) that `np.load(..., mmap_mode='r')` in SpatialFeatureLookup picks
    # up directly without re-extracting from the .npz.
    sidecar = out.with_suffix(".z_spatial.npy")
    print(f"[dinov2-extract] writing features directly to sidecar: {sidecar}")
    z_spatial = np.lib.format.open_memmap(
        sidecar, mode="w+", dtype=np.float16,
        shape=(len(paths), grid_h, grid_w, hidden),
    )
    spatial_hw_native = np.empty((len(paths), 2), dtype=np.int16)
    spatial_hw_native[:] = (grid_h, grid_w)
    bytes_total = len(paths) * grid_h * grid_w * hidden * 2
    print(f"[dinov2-extract] expected on-disk size (raw): {bytes_total/1e9:.2f} GB")

    t_total = 0.0
    bs = args.batch_size
    for start in range(0, len(paths), bs):
        sub_paths = paths[start : start + bs]
        batch_imgs = [preprocess(Image.open(p).convert("RGB")) for p in sub_paths]
        x = torch.stack(batch_imgs).to(device, dtype=torch.float16, non_blocking=True)
        t0 = time.time()
        with torch.no_grad():
            outputs = model(pixel_values=x, interpolate_pos_encoding=True)
            patches = outputs.last_hidden_state[:, cls_offset:, :]
            patches = patches.reshape(x.shape[0], grid_h, grid_w, hidden)
        patches_np = patches.detach().cpu().numpy().astype(np.float16)
        z_spatial[start : start + len(sub_paths)] = patches_np
        t_total += time.time() - t0
        done = start + len(sub_paths)
        if (done // args.log_every) > ((done - len(sub_paths)) // args.log_every) or done == len(paths):
            print(f"[dinov2-extract] {done}/{len(paths)} encoded "
                  f"({1000*t_total/done:.1f} ms/frame avg, "
                  f"est total {t_total/done*len(paths)/60:.1f} min)")

    # Flush memmap and release the handle before saving the .npz.
    z_spatial.flush()
    del z_spatial

    # Save .npz with metadata only (sidecar holds the big z_spatial.npy already).
    # SpatialFeatureLookup.from_npz: reads paths/target_hw from the .npz, then
    # checks if sidecar exists (yes here) — so it skips re-extraction and mmap's
    # the sidecar directly. Skipping z_spatial here avoids a redundant 30+ GB write.
    save_fn = np.savez_compressed if args.compress else np.savez
    save_fn(
        out,
        spatial_hw_native=spatial_hw_native,
        target_hw=np.array(target_hw, dtype=np.int16),
        paths=np.array(rels, dtype=object),
        cells=np.array(cells, dtype=object),
    )
    npz_sz = out.stat().st_size / 1e9
    sidecar_sz = sidecar.stat().st_size / 1e9
    print(f"[dinov2-extract] saved metadata to {out} ({npz_sz*1000:.1f} MB) "
          f"+ sidecar {sidecar} ({sidecar_sz:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
