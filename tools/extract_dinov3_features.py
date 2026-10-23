#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract DINOv3 spatial feature maps for an entire dataset (T2.4 of publishability_plan.md §12.6).

Uses ``facebook/dinov3-vitl16-pretrain-lvd1689m`` by default (ViT-L/16 + register tokens,
Siméoni et al. Meta AI, arXiv:2508.10104, Aug 2025). Same protocol as
tools/extract_dinov2_features.py but with patch size 16 instead of 14, so the
recommended rectangular input is 1216×352 (multiples of 16, closest to our
DINOv2 1218×350 = multiples of 14).

Output grid: 22×76×1024 (compare DINOv2-L at 25×87×1024).

Saved schema mirrors tools/extract_dinov2_features.py so the cached features
are drop-in compatible with SpatialFeatureLookup + downstream
linear_probe/labeler/RD training code.

PREREQUISITE: DINOv3 is a GATED HuggingFace repo. Before running:
  1. Log into HuggingFace and request access at
     https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
  2. Accept the DINOv3 License + share contact info with Meta.
  3. After approval, set: huggingface-cli login (or export HF_TOKEN=...).

Usage (once access is granted):
    PY=python
    $PY tools/extract_dinov3_features.py --kind carla \\
        --root ~/datasets/carla_extracted \\
        --img-h 352 --img-w 1216 \\
        --out ~/datasets/carla_extracted/spatial_features_dinov3.npz

    $PY tools/extract_dinov3_features.py --kind kitti \\
        --root ~/datasets/kitti/extracted/training/image_2 \\
        --img-h 352 --img-w 1216 \\
        --out ~/datasets/kitti/spatial_features_dinov3.npz
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


DEFAULT_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_IMG_SIZE = 512  # 32×16 = 512 (closest multiple of 16 to DINOv2's default 518)
DEFAULT_PATCH_SIZE = 16
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
    p.add_argument("--kind", choices=["kitti", "cityscapes", "carla", "generic"], required=True,
                   help="generic = flat dir of .jpg/.png (e.g. COCO train2017); rels are filenames.")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                   help="HF model id. Default: facebook/dinov3-vitl16-pretrain-lvd1689m (ViT-L/16 + registers).")
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE,
                   help="Square input size when --img-h/--img-w not set. Multiple of patch_size (16).")
    p.add_argument("--img-h", type=int, default=None,
                   help="Rectangular input height (must be multiple of patch_size=16). "
                        "Recommended for KITTI/CARLA: --img-h 352 --img-w 1216 → 22×76 grid.")
    p.add_argument("--img-w", type=int, default=None,
                   help="Rectangular input width (must be multiple of patch_size=16).")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--compress", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.kind == "kitti":
        paths, rels = gather_kitti(root)
        cells = [""] * len(paths)
    elif args.kind == "generic":
        paths = sorted(list(root.glob("*.jpg")) + list(root.glob("*.png")))
        rels = [p.name for p in paths]
        cells = [""] * len(paths)
    elif args.kind == "cityscapes":
        paths, rels = gather_cityscapes(root)
        cells = [""] * len(paths)
    else:
        paths, rels, cells = gather_carla(root)

    if not paths:
        print(f"[dinov3-extract] FAIL: no frames found at {root}")
        return 1

    if args.limit:
        paths = paths[: args.limit]
        rels = rels[: args.limit]
        cells = cells[: args.limit]

    print(f"[dinov3-extract] kind={args.kind} root={root} -> {len(paths)} frames")

    if not torch.cuda.is_available():
        print("[dinov3-extract] FAIL: no CUDA")
        return 1
    device = torch.device("cuda")

    print(f"[dinov3-extract] loading {args.model_id} (bf16 compute, fp16 storage)...")
    print(f"[dinov3-extract] NOTE: DINOv3 is a gated HF repo — if loading fails with 403, request access at")
    print(f"[dinov3-extract]       https://huggingface.co/{args.model_id} and run huggingface-cli login")
    print(f"[dinov3-extract] NOTE: DINOv3 produces NaN in pure fp16 at rectangular aspect ratios (verified 2026-05-17);")
    print(f"[dinov3-extract]       we use bf16 (same memory as fp16, fp32-equivalent dynamic range) for compute,")
    print(f"[dinov3-extract]       then cast outputs to fp16 for on-disk storage.")
    t0 = time.time()
    model = AutoModel.from_pretrained(args.model_id, dtype=torch.bfloat16).to(device).eval()
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
            print(f"[dinov3-extract] FAIL: {name}={v} not divisible by patch_size {patch_size}")
            return 1
    grid_h = img_h // patch_size
    grid_w = img_w // patch_size
    target_hw = (grid_h, grid_w)
    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[dinov3-extract] loaded in {time.time()-t0:.1f}s | hidden={hidden}, "
          f"patch_size={patch_size}, img_hw=({img_h}, {img_w}), grid={grid_h}x{grid_w}, "
          f"n_registers={n_registers}, cls_offset={cls_offset}, "
          f"n_params={n_params_m:.1f}M")

    preprocess = make_preprocess(img_h, img_w)

    sidecar = out.with_suffix(".z_spatial.npy")
    print(f"[dinov3-extract] writing features directly to sidecar: {sidecar}")
    z_spatial = np.lib.format.open_memmap(
        sidecar, mode="w+", dtype=np.float16,
        shape=(len(paths), grid_h, grid_w, hidden),
    )
    spatial_hw_native = np.empty((len(paths), 2), dtype=np.int16)
    spatial_hw_native[:] = (grid_h, grid_w)
    bytes_total = len(paths) * grid_h * grid_w * hidden * 2
    print(f"[dinov3-extract] expected on-disk size (raw): {bytes_total/1e9:.2f} GB")

    t_total = 0.0
    bs = args.batch_size
    for start in range(0, len(paths), bs):
        sub_paths = paths[start : start + bs]
        batch_imgs = [preprocess(Image.open(p).convert("RGB")) for p in sub_paths]
        x = torch.stack(batch_imgs).to(device, dtype=torch.bfloat16, non_blocking=True)
        t0 = time.time()
        with torch.no_grad():
            outputs = model(pixel_values=x, interpolate_pos_encoding=True)
            patches = outputs.last_hidden_state[:, cls_offset:, :]
            patches = patches.reshape(x.shape[0], grid_h, grid_w, hidden)
        # Cast bf16 -> fp16 for on-disk storage (same memory footprint as before).
        # bf16 → fp16: bf16 has wider exponent range, but our values are in [-3, 3] so no overflow.
        patches_np = patches.detach().to(torch.float16).cpu().numpy()
        # Safety: abort early if any NaN slipped in (e.g., if a future model variant has new
        # fp16-incompatible operations). Better to fail fast than to corrupt the whole cache.
        if np.isnan(patches_np).any():
            n_nan = np.isnan(patches_np).sum()
            n_total = patches_np.size
            raise RuntimeError(
                f"[dinov3-extract] FAIL: detected {n_nan}/{n_total} NaN values in extracted features "
                f"at batch start={start}. bf16 should prevent this — please file an issue."
            )
        z_spatial[start : start + len(sub_paths)] = patches_np
        t_total += time.time() - t0
        done = start + len(sub_paths)
        if (done // args.log_every) > ((done - len(sub_paths)) // args.log_every) or done == len(paths):
            print(f"[dinov3-extract] {done}/{len(paths)} encoded "
                  f"({1000*t_total/done:.1f} ms/frame avg, "
                  f"est total {t_total/done*len(paths)/60:.1f} min)")

    z_spatial.flush()
    del z_spatial

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
    print(f"[dinov3-extract] saved metadata to {out} ({npz_sz*1000:.1f} MB) "
          f"+ sidecar {sidecar} ({sidecar_sz:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
