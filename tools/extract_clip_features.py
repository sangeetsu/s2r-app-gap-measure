#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract CLIP spatial feature maps for an entire dataset.

Uses ``openai/clip-vit-large-patch14`` (CLIP ViT-L/14) as the 4th foundation
in the dose-response curve. Trained via image-text contrastive learning — no
spatial supervision. Expected dose-response position: between Cosmos (reconstruction)
and Alpamayo R1 (trajectory). Quantifies "what does a text-aligned foundation
recover from sim2real driving detection."

Schema-compatible with ``SpatialFeatureLookup``. Rectangular 1218×350 input
(aspect-matched) → 87×25 patch grid × 1024 channels (CLIP-L hidden_size).
Position embeddings interpolated automatically by transformers.

Usage:
    PY=python
    $PY tools/extract_clip_features.py --kind carla \\
        --root ~/datasets/carla_extracted \\
        --out ~/datasets/carla_extracted/spatial_features_clip.npz \\
        --img-h 350 --img-w 1218
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
from transformers import CLIPVisionModel

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.path.insert(0, str(REPO_ROOT / "tools"))
from frame_index import gather_carla, gather_cityscapes, gather_kitti


DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"
DEFAULT_IMG_SIZE = 224
DEFAULT_PATCH_SIZE = 14
# CLIP's official normalization (different from DINOv2's ImageNet stats).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def make_preprocess(img_h: int, img_w: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_h, img_w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["kitti", "cityscapes", "carla"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--img-h", type=int, default=350, help="Rect input height (default 350 for KITTI/CARLA aspect).")
    p.add_argument("--img-w", type=int, default=1218, help="Rect input width (default 1218 for KITTI/CARLA aspect).")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-every", type=int, default=500)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.kind == "kitti":
        paths, rels = gather_kitti(root); cells = [""] * len(paths)
    elif args.kind == "cityscapes":
        paths, rels = gather_cityscapes(root); cells = [""] * len(paths)
    else:
        paths, rels, cells = gather_carla(root)

    if not paths:
        print(f"[clip-extract] FAIL: no frames at {root}")
        return 1
    if args.limit:
        paths = paths[: args.limit]; rels = rels[: args.limit]; cells = cells[: args.limit]
    print(f"[clip-extract] kind={args.kind} -> {len(paths)} frames")

    if not torch.cuda.is_available():
        print("[clip-extract] FAIL: no CUDA")
        return 1
    device = torch.device("cuda")

    print(f"[clip-extract] loading {args.model_id} (fp16)...")
    t0 = time.time()
    model = CLIPVisionModel.from_pretrained(args.model_id, dtype=torch.float16).to(device).eval()
    for pp in model.parameters():
        pp.requires_grad_(False)
    hidden = model.config.hidden_size
    patch_size = model.config.patch_size
    if args.img_h % patch_size != 0 or args.img_w % patch_size != 0:
        print(f"[clip-extract] FAIL: img_h={args.img_h} or img_w={args.img_w} not multiples of patch_size={patch_size}")
        return 1
    grid_h = args.img_h // patch_size
    grid_w = args.img_w // patch_size
    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[clip-extract] loaded in {time.time()-t0:.1f}s | hidden={hidden}, "
          f"patch_size={patch_size}, img_hw=({args.img_h}, {args.img_w}), grid={grid_h}x{grid_w}, "
          f"n_params={n_params_m:.1f}M")

    preprocess = make_preprocess(args.img_h, args.img_w)

    sidecar = out.with_suffix(".z_spatial.npy")
    print(f"[clip-extract] writing features to sidecar: {sidecar}")
    z_spatial = np.lib.format.open_memmap(
        sidecar, mode="w+", dtype=np.float16,
        shape=(len(paths), grid_h, grid_w, hidden),
    )
    spatial_hw_native = np.empty((len(paths), 2), dtype=np.int16)
    spatial_hw_native[:] = (grid_h, grid_w)
    bytes_total = len(paths) * grid_h * grid_w * hidden * 2
    print(f"[clip-extract] expected on-disk size (raw): {bytes_total/1e9:.2f} GB")

    t_total = 0.0
    bs = args.batch_size
    for start in range(0, len(paths), bs):
        sub_paths = paths[start : start + bs]
        batch_imgs = [preprocess(Image.open(p).convert("RGB")) for p in sub_paths]
        x = torch.stack(batch_imgs).to(device, dtype=torch.float16, non_blocking=True)
        t0 = time.time()
        with torch.no_grad():
            outputs = model(pixel_values=x, interpolate_pos_encoding=True)
            # last_hidden_state: (B, 1+grid_h*grid_w, hidden) -- drop CLS at index 0
            patches = outputs.last_hidden_state[:, 1:, :]
            patches = patches.reshape(x.shape[0], grid_h, grid_w, hidden)
        patches_np = patches.detach().cpu().numpy().astype(np.float16)
        z_spatial[start : start + len(sub_paths)] = patches_np
        t_total += time.time() - t0
        done = start + len(sub_paths)
        if (done // args.log_every) > ((done - len(sub_paths)) // args.log_every) or done == len(paths):
            print(f"[clip-extract] {done}/{len(paths)} encoded ({1000*t_total/done:.1f} ms/frame avg, "
                  f"est total {t_total/done*len(paths)/60:.1f} min)")

    z_spatial.flush()
    del z_spatial

    np.savez(
        out,
        spatial_hw_native=spatial_hw_native,
        target_hw=np.array((grid_h, grid_w), dtype=np.int16),
        paths=np.array(rels, dtype=object),
        cells=np.array(cells, dtype=object),
    )
    npz_sz = out.stat().st_size / 1e9
    sidecar_sz = sidecar.stat().st_size / 1e9
    print(f"[clip-extract] saved metadata to {out} ({npz_sz*1000:.1f} MB) "
          f"+ sidecar {sidecar} ({sidecar_sz:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
