#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train a Faster R-CNN detector on a synthetic (CARLA / translated CARLA)
source and evaluate KITTI val mAP. Optionally add an auxiliary head that
regresses a foundation-model latent for each frame (Strategy i of the paper).

Conditions covered by this single tool:
  --source-data carla              (raw CARLA, no adaptation)
  --source-data carla-translated   (EPE / HyPER-GAN / etc.; supply --source-root)
  --aux-latents PATH               (turn on the auxiliary head; the file's
                                    per-frame latents become regression targets)
  --aux-target {raw, coral-aligned}
                                   (apply CORAL alignment to the latents
                                    before they become targets; needs
                                    --coral-fit-pair to specify which real
                                    target to align to)

Usage examples:
  # Day 2 baseline (vanilla CARLA, no aux):
  python tools/train_detector.py \
      --source-data carla --source-root ~/datasets/carla_extracted \
      --val-data kitti --val-splits ~/datasets/kitti/splits.json \
      --epochs 12 --batch-size 4 \
      --out checkpoints/vanilla_carla \
      --result results/vanilla_carla_kitti_val.json

  # Day 3 Strategy (i) (aux head + CORAL-aligned Alpamayo targets):
  python tools/train_detector.py \
      --source-data carla --source-root ~/datasets/carla_extracted \
      --aux-latents ~/datasets/carla_extracted/latents.npz \
      --aux-target coral-aligned \
      --coral-fit-real ~/datasets/kitti/latents.npz \
      --aux-lambda 0.1 \
      --val-data kitti --val-splits ~/datasets/kitti/splits.json \
      --epochs 12 --batch-size 4 \
      --out checkpoints/strategy_i \
      --result results/strategy_i_kitti_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch.nn.functional as F

from s2rgap.adaptation.deep_coral import deep_coral_loss
from s2rgap.adaptation.distillation import PatchProjector, patch_distill_loss
from s2rgap.adaptation.feature_level import fit_coral
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection, CARLA_CLASSES
from s2rgap.data.kitti import KITTIDetection, KITTI_CLASSES, collate_fn
from s2rgap.data.kitti_pseudo import KITTIPseudoLabels
from s2rgap.detector import FasterRCNNWithAuxLatent, build_detector
from s2rgap.eval_detection import evaluate_map, save_eval_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", choices=["carla", "carla-translated", "kitti"], default="carla")
    parser.add_argument("--source-root", default=str(Path.home() / "datasets" / "carla_extracted"))
    parser.add_argument("--val-data", choices=["kitti"], default="kitti")
    parser.add_argument("--val-splits", default=str(Path.home() / "datasets" / "kitti" / "splits.json"))
    parser.add_argument("--kitti-root", default=str(Path.home() / "datasets" / "kitti" / "extracted" / "training"))
    parser.add_argument(
        "--source-frames-slice", default=None,
        help="start:end slice of the (sorted) source frames, e.g. 0:800 to train on the first "
             "800 frames of a cell and hold out the last 200 as the in-domain ANCHOR. The "
             "appearance-gap framework needs drop-relative-to-anchor (ablation_log §15.20).")
    parser.add_argument(
        "--source-cells", default=None,
        help="Comma-separated CARLA cell names (e.g. Town01_ClearNoon,Town03_ClearNoon) to "
             "restrict the source set. Used to build the appearance-gap calibration grid: "
             "training on one cell and evaluating across cells isolates the appearance axis "
             "(weather at fixed town) from the content axis (town at fixed weather).")
    parser.add_argument(
        "--pseudo-image-root", default=None,
        help="Image root for --kitti-pseudo-labels frames (dir containing image_2/). "
             "Defaults to --kitti-root. Set this to point pseudo/AL-labeled frames at a "
             "different dataset than the val set (e.g. a flat Cityscapes symlink dir), so "
             "the val split stays KITTI while the labeled target frames come from elsewhere.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--arch", choices=["fasterrcnn", "fcos", "retinanet", "ddetr", "frcnn_swin"], default="fasterrcnn", help="detector family (torchvision R50-FPN; ddetr / frcnn_swin added 2026-10-17)")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd", help="sgd = every run before 2026-10-17")
    parser.add_argument("--backbone-lr-mult", type=float, default=1.0, help="lr multiplier for parameters whose name contains 'backbone' (adamw only)")
    parser.add_argument("--init", choices=["coco", "imagenet", "scratch"], default="coco",
        help="detector initialisation: coco = COCO-detection pretrained (default, all runs before 2026-10-14); imagenet = ImageNet backbone only; scratch")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-steps", default=None,
        help="comma-separated epochs for MultiStepLR x0.1 (e.g. 16,22). Default (None) = legacy StepLR(step_size=3), used by all runs before 2026-10-14.")
    parser.add_argument("--warmup-iters", type=int, default=0, help="linear LR warmup from 0.001x over this many iterations")
    parser.add_argument("--clip-grad", type=float, default=None, help="max gradient norm (None = no clipping; all runs before 2026-10-16 unclipped)")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--aux-latents", default=None, help="Path to per-frame foundation latents (.npz with z, paths)")
    parser.add_argument("--aux-target", choices=["raw", "coral-aligned"], default="raw")
    parser.add_argument("--coral-fit-real", default=None, help="Real latents .npz used to fit CORAL when --aux-target=coral-aligned")
    parser.add_argument("--coral-eps", type=float, default=1e-4)
    parser.add_argument("--aux-lambda", type=float, default=0.1)
    parser.add_argument("--out", default="checkpoints/detector")
    parser.add_argument("--result", default="results/detector_eval.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="One short training step + one eval batch — just verify the pipeline.")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="If --out contains a latest_epoch.pt, resume from it.")
    # Deep CORAL mode (Sun & Saenko ECCV-W 2016): distribution-level feature
    # alignment between source and target during training. Replaces the
    # per-frame aux-head regression of --aux-target. See plan_2026-05-13.md.
    parser.add_argument(
        "--method",
        choices=["pattern_a", "deepcoral", "dist_only"],
        default="pattern_a",
        help="pattern_a = aux head predicts foundation latent (existing). "
        "deepcoral = align backbone-feature covariance between CARLA and KITTI batches. "
        "dist_only = patch-level cosine distillation from a frozen foundation teacher "
        "(DINO Teacher recipe, no labeler or Mean Teacher).",
    )
    parser.add_argument(
        "--target-images-root",
        default=str(Path.home() / "datasets" / "kitti" / "extracted" / "training"),
        help="Root of the unlabeled real target images (used for deepcoral and dist_only).",
    )
    parser.add_argument(
        "--coral-lambda",
        type=float,
        default=0.5,
        help="Weight on the Deep CORAL loss (Sun & Saenko 2016 default ~0.5).",
    )
    parser.add_argument(
        "--coral-normalize",
        action="store_true",
        help="Per-channel z-score features before CORAL (fixes near-zero "
             "gradient when post-pool FPN activations are tiny).",
    )
    # --- dist_only (DINO Teacher patch distillation) ---------------------
    parser.add_argument(
        "--source-spatial-features",
        default=None,
        help="Path to source-domain Alpamayo spatial features .npz "
             "(produced by tools/extract_alpamayo_feature_maps.py).",
    )
    parser.add_argument(
        "--target-spatial-features",
        default=None,
        help="Path to target-domain Alpamayo spatial features .npz (unlabeled KITTI train).",
    )
    parser.add_argument(
        "--dist-lambda",
        type=float,
        default=1.0,
        help="Weight on the patch distillation loss (DINO Teacher default λ_sim=1.0).",
    )
    parser.add_argument(
        "--dist-fpn-level",
        default="pool",
        choices=["0", "1", "2", "3", "pool", "all"],
        help="Which FPN level to distill at. 'pool' (default) is the coarsest "
             "and closest to Alpamayo's W/28 grid. 'all' distills at every FPN "
             "level simultaneously (each bilinear-resized to teacher's 7x25 "
             "grid, mean over levels), addressing the small-object collapse "
             "when single-level coarse distillation underconstrains fine FPN "
             "scales.",
    )
    parser.add_argument(
        "--dist-projector-hidden",
        type=int,
        default=1024,
        help="Hidden dim of the 2-layer MLP_g projector (DINO Teacher uses 1024).",
    )
    parser.add_argument(
        "--dist-teacher-channels",
        type=int,
        default=4096,
        help="Channel dim of the teacher spatial map (Alpamayo Qwen3-VL = 4096).",
    )
    # --- KITTI pseudo-labels (orthogonal to --method) -------------------
    parser.add_argument(
        "--kitti-pseudo-labels",
        default=None,
        help="Path to JSON pseudo-labels from tools/label_kitti_with_teacher.py. "
             "When set, KITTI train images (with pseudo-boxes filtered by --pseudo-tau) "
             "are ConcatDataset'd onto the CARLA source dataset.",
    )
    parser.add_argument(
        "--pseudo-tau",
        type=float,
        default=0.7,
        help="Confidence threshold for pseudo-labels. DINO Teacher default 0.7.",
    )
    parser.add_argument(
        "--pseudo-min-box-area",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--pseudo-max-dets",
        type=int,
        default=100,
        help="Per-image cap on retained pseudo-detections (top by score).",
    )
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def build_aux_targets(args: argparse.Namespace) -> tuple[Path | None, callable | None]:
    """Resolve aux-target handling: optionally fit CORAL and return a
    per-frame latent transform that the CARLA Dataset will pre-apply."""
    if args.aux_latents is None:
        return None, None
    aux_path = Path(args.aux_latents).expanduser()
    if args.aux_target == "raw":
        return aux_path, None

    if args.aux_target == "coral-aligned":
        if args.coral_fit_real is None:
            raise ValueError("--aux-target coral-aligned requires --coral-fit-real")
        log(f"fitting CORAL ({aux_path.name} -> {Path(args.coral_fit_real).name}) eps={args.coral_eps}")
        z_sim = np.load(aux_path)["z"]
        z_real = np.load(args.coral_fit_real)["z"]
        transform = fit_coral(z_sim, z_real, eps=args.coral_eps)

        def per_frame_apply(z: np.ndarray) -> np.ndarray:
            return transform.apply(z.reshape(1, -1)).reshape(-1)

        return aux_path, per_frame_apply
    raise ValueError(f"unknown aux-target {args.aux_target}")


def build_train_loader(args: argparse.Namespace) -> tuple[DataLoader, int, bool]:
    aux_path, aux_transform = build_aux_targets(args)
    use_aux = aux_path is not None

    if args.source_data in ("carla", "carla-translated"):
        ds = CARLADetection(
            root=args.source_root,
            cells=[c.strip() for c in args.source_cells.split(",")] if args.source_cells else None,
            latents_npz=aux_path,
            latent_transform=aux_transform,
        )
        if args.source_frames_slice:
            a, b = (int(x) for x in args.source_frames_slice.split(":"))
            ds = torch.utils.data.Subset(ds, range(a, min(b, len(ds))))
            log(f"source frames slice {a}:{b} -> {len(ds)} frames")
    elif args.source_data == "kitti":
        ds = KITTIDetection(
            kitti_root=args.kitti_root,
            split_json=args.val_splits,
            split="train",
        )
    else:
        raise ValueError(args.source_data)
    # Optional pseudo-labels: ConcatDataset KITTI train images with labeler
    # predictions onto whatever the source dataset is. This is orthogonal to
    # --method (works with vanilla / pattern_a / deepcoral / dist_only).
    if args.kitti_pseudo_labels is not None:
        pseudo_ds = KITTIPseudoLabels(
            kitti_root=(args.pseudo_image_root or args.kitti_root),
            pseudo_json=args.kitti_pseudo_labels,
            score_thresh=args.pseudo_tau,
            min_box_area=args.pseudo_min_box_area,
            max_dets_per_image=args.pseudo_max_dets,
        )
        log(
            f"pseudo-labels: {len(pseudo_ds)} KITTI train frames at τ={args.pseudo_tau} "
            f"from {args.kitti_pseudo_labels}"
        )
        ds = torch.utils.data.ConcatDataset([ds, pseudo_ds])
    if args.limit_train is not None:
        ds = torch.utils.data.Subset(ds, range(min(args.limit_train, len(ds))))
    log(f"train: {len(ds)} total frames (source={args.source_data}, "
        f"pseudo={'+KITTI' if args.kitti_pseudo_labels else 'none'}); aux={use_aux}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    return loader, len(ds), use_aux


def build_val_loader(args: argparse.Namespace) -> tuple[DataLoader, int]:
    if args.val_data == "kitti":
        ds = KITTIDetection(
            kitti_root=args.kitti_root,
            split_json=args.val_splits,
            split="val",
        )
    else:
        raise ValueError(args.val_data)
    if args.limit_val is not None:
        ds = torch.utils.data.Subset(ds, range(min(args.limit_val, len(ds))))
    log(f"val: {len(ds)} frames from {args.val_data}")
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        collate_fn=collate_fn,
    )
    return loader, len(ds)


def stack_aux_targets(targets: list[dict[str, torch.Tensor]]) -> torch.Tensor | None:
    """If every target carries an alpamayo_latent, return (B, D); else None."""
    if not all("alpamayo_latent" in t for t in targets):
        return None
    return torch.stack([t["alpamayo_latent"] for t in targets], dim=0)


# --------------------------------------------------------------------------
# Deep CORAL support (Sun & Saenko, ECCV-W 2016)
# --------------------------------------------------------------------------


class _BackbonePoolCapture:
    """Registers a forward hook on FasterRCNN's backbone FPN, capturing the
    coarsest level ("pool") so we can pool it to a (B, 256) vector for the
    Deep CORAL loss."""

    def __init__(self, detector) -> None:
        self._captured: dict[str, torch.Tensor] = {}
        detector.backbone.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output) -> None:
        if isinstance(output, dict) and "pool" in output:
            self._captured["fpn"] = output["pool"]

    def pop_pooled(self) -> torch.Tensor:
        feat = self._captured.pop("fpn")
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, 256)


def _target_backbone_forward(detector, images: list[torch.Tensor]) -> None:
    """Run target-domain images through FasterRCNN's transform + backbone only.
    The FPN feature is captured via the registered hook. We discard the RPN /
    ROI-head outputs — only the backbone gradient matters for Deep CORAL."""
    images_lst, _ = detector.transform(images, None)
    _ = detector.backbone(images_lst.tensors)


def build_target_image_loader(args: argparse.Namespace) -> DataLoader:
    """KITTI training images, used unlabeled, for Deep CORAL distribution matching."""
    ds = KITTIDetection(
        kitti_root=args.target_images_root,
        split_json=args.val_splits,
        split="train",
    )
    if args.limit_train is not None:
        ds = torch.utils.data.Subset(ds, range(min(args.limit_train, len(ds))))
    log(f"target (unlabeled, for Deep CORAL): {len(ds)} KITTI train images")
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(1, args.num_workers // 2),
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )


# --------------------------------------------------------------------------
# dist_only support (DINO Teacher patch-level distillation, no labeler)
# --------------------------------------------------------------------------


FPN_ALL_LEVELS = ("0", "1", "2", "3", "pool")


class _FPNCapture:
    """Captures one or more named levels of FasterRCNN's FPN output.

    For a single level, ``pop()`` returns the captured tensor for backward
    compatibility. For multiple levels (``--dist-fpn-level all``), pass the
    list of level names and use ``pop_dict()`` to get back the {level: tensor}
    mapping.
    """

    def __init__(self, detector, levels: str | list[str]) -> None:
        if isinstance(levels, str):
            if levels == "all":
                self.levels: list[str] = list(FPN_ALL_LEVELS)
            else:
                self.levels = [levels]
        else:
            self.levels = list(levels)
        self._captured: dict[str, torch.Tensor] = {}
        detector.backbone.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output) -> None:
        if isinstance(output, dict):
            self._captured = {k: output[k] for k in self.levels if k in output}

    def pop(self) -> torch.Tensor:
        """Single-level convenience. Errors if multiple levels are tracked."""
        if len(self.levels) != 1:
            raise RuntimeError(
                f"pop() requires exactly one level, got {self.levels}; use pop_dict()"
            )
        captured = self._captured
        self._captured = {}
        return captured[self.levels[0]]

    def pop_dict(self) -> dict[str, torch.Tensor]:
        captured = self._captured
        self._captured = {}
        return captured


def _teacher_feat_to_nchw(maps_bhwd: torch.Tensor) -> torch.Tensor:
    """(B, H, W, D) fp16 from disk -> (B, D, H, W) NCHW for the loss."""
    return maps_bhwd.permute(0, 3, 1, 2).contiguous()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not torch.cuda.is_available():
        log("FAIL: CUDA not available")
        return 1
    device = torch.device("cuda")
    # Drop any leftover CUDA cache from prior runs in the same Python kernel.
    # (No-op for a fresh subprocess but cheap defense against hot-reload setups.)
    torch.cuda.empty_cache()
    log(f"device: {torch.cuda.get_device_name(0)}, vram_total={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}G, "
        f"vram_allocated_at_start={torch.cuda.memory_allocated()/1e9:.2f}G")

    train_loader, n_train, use_aux = build_train_loader(args)
    val_loader, n_val = build_val_loader(args)

    use_deepcoral = args.method == "deepcoral"
    use_dist_only = args.method == "dist_only"
    target_loader = (
        build_target_image_loader(args) if (use_deepcoral or use_dist_only) else None
    )
    pool_capture: _BackbonePoolCapture | None = None
    fpn_capture: _FPNCapture | None = None
    projector: PatchProjector | None = None
    src_spatial: SpatialFeatureLookup | None = None
    tgt_spatial: SpatialFeatureLookup | None = None

    # Build model.
    if use_deepcoral:
        if use_aux:
            log("WARN: --method deepcoral overrides Pattern-A aux head; ignoring --aux-latents")
        model = build_detector(num_classes=4, init=args.init, arch=args.arch)
        pool_capture = _BackbonePoolCapture(model)
        log(f"detector: FasterRCNN + Deep CORAL (coral_lambda={args.coral_lambda})")
    elif use_dist_only:
        if use_aux:
            log("WARN: --method dist_only overrides Pattern-A aux head; ignoring --aux-latents")
        if not args.source_spatial_features or not args.target_spatial_features:
            raise ValueError(
                "--method dist_only requires --source-spatial-features AND --target-spatial-features"
            )
        model = build_detector(num_classes=4, init=args.init, arch=args.arch)
        # FPN out_channels = 256 for ResNet-50 FPN; teacher channels = 4096 for Alpamayo.
        projector = PatchProjector(
            in_channels=256,
            out_channels=args.dist_teacher_channels,
            hidden=args.dist_projector_hidden,
        )
        fpn_capture = _FPNCapture(model, levels=args.dist_fpn_level)
        log(f"loading source spatial features: {args.source_spatial_features}")
        src_spatial = SpatialFeatureLookup.from_npz(args.source_spatial_features)
        log(f"  -> {len(src_spatial)} frames, target_hw={src_spatial.target_hw}")
        log(f"loading target spatial features: {args.target_spatial_features}")
        tgt_spatial = SpatialFeatureLookup.from_npz(args.target_spatial_features)
        log(f"  -> {len(tgt_spatial)} frames, target_hw={tgt_spatial.target_hw}")
        log(
            f"detector: FasterRCNN + patch distillation "
            f"(level={args.dist_fpn_level}, lambda={args.dist_lambda}, "
            f"projector 256 -> {args.dist_projector_hidden} -> {args.dist_teacher_channels})"
        )
    elif use_aux:
        # Determine aux dim from one sample's latent.
        sample_target = train_loader.dataset[0][1] if not isinstance(train_loader.dataset, torch.utils.data.Subset) else train_loader.dataset.dataset[0][1]
        if "alpamayo_latent" not in sample_target:
            raise RuntimeError("Aux requested but training samples don't carry alpamayo_latent")
        aux_dim_out = int(sample_target["alpamayo_latent"].shape[0])
        model = FasterRCNNWithAuxLatent(num_classes=4, aux_dim_out=aux_dim_out, aux_lambda=args.aux_lambda)
        log(f"detector: FasterRCNN + aux head (out_dim={aux_dim_out}, lambda={args.aux_lambda})")
    else:
        model = build_detector(num_classes=4, init=args.init, arch=args.arch)
        log(f"detector: {args.arch} (init {args.init}, no aux head)")
    model.to(device)
    if projector is not None:
        projector.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if projector is not None:
        trainable += [p for p in projector.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        groups = [{"params": [p for n, p in named if "backbone" in n], "lr": args.lr * args.backbone_lr_mult},
                  {"params": [p for n, p in named if "backbone" not in n], "lr": args.lr}]
        optimizer = torch.optim.AdamW([g for g in groups if g["params"]], lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            trainable,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    for g in optimizer.param_groups: g["base_lr_"] = g["lr"]   # warmup scales each group's own base lr (SGD: one group, = args.lr)
    if args.lr_steps:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(x) for x in args.lr_steps.split(",")], gamma=0.1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training.log"

    # Resume support: if --resume and latest_epoch.pt is present, restore.
    start_epoch = 0
    if args.resume:
        latest_path = out_dir / "latest_epoch.pt"
        if latest_path.exists():
            ckpt = torch.load(latest_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            if projector is not None and "projector" in ckpt:
                projector.load_state_dict(ckpt["projector"])
            start_epoch = int(ckpt["epoch"]) + 1
            log(f"resuming from epoch {start_epoch} (ckpt {latest_path})")
        else:
            log(f"--resume passed but {latest_path} not found; starting fresh")
    if start_epoch == 0:
        log_path.write_text("")

    total_steps = 0
    t0_train = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        target_iter = iter(target_loader) if target_loader is not None else None
        for images, targets in train_loader:
            images = [im.to(device) for im in images]
            targets_on_dev = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]

            aux_targets = stack_aux_targets(targets) if use_aux else None
            if aux_targets is not None:
                aux_targets = aux_targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            if use_deepcoral:
                # Source forward: full FR-CNN (det losses + capture src pool features)
                losses = model(images, targets_on_dev)
                src_pooled = pool_capture.pop_pooled()
                # Target forward: backbone only (captures tgt pool features)
                try:
                    tgt_images, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    tgt_images, _ = next(target_iter)
                tgt_images = [im.to(device) for im in tgt_images]
                _target_backbone_forward(model, tgt_images)
                tgt_pooled = pool_capture.pop_pooled()
                # Deep CORAL loss between batch feature distributions.
                coral = deep_coral_loss(
                    src_pooled, tgt_pooled, normalize_features=args.coral_normalize
                )
                losses["loss_coral"] = args.coral_lambda * coral
            elif use_dist_only:
                # Source (CARLA) forward: full FR-CNN det losses + capture FPN feat(s).
                losses = model(images, targets_on_dev)
                src_feats = fpn_capture.pop_dict()  # {level: (B, 256, h, w)}
                src_paths = [t["frame_rel"] for t in targets]
                src_teacher = src_spatial.get_batch(src_paths).to(device)  # (B,H,W,D) fp16
                src_teacher_nchw = _teacher_feat_to_nchw(src_teacher)
                # Multi-level: mean over levels; single-level: returns the same value.
                loss_src = sum(
                    patch_distill_loss(feat, src_teacher_nchw, projector=projector)
                    for feat in src_feats.values()
                ) / max(1, len(src_feats))

                # Target (unlabeled KITTI) forward: backbone only + capture FPN feat(s).
                try:
                    tgt_images, tgt_targets = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    tgt_images, tgt_targets = next(target_iter)
                tgt_images_dev = [im.to(device) for im in tgt_images]
                _target_backbone_forward(model, tgt_images_dev)
                tgt_feats = fpn_capture.pop_dict()
                tgt_paths = [t["frame_rel"] for t in tgt_targets]
                tgt_teacher = tgt_spatial.get_batch(tgt_paths).to(device)
                tgt_teacher_nchw = _teacher_feat_to_nchw(tgt_teacher)
                loss_tgt = sum(
                    patch_distill_loss(feat, tgt_teacher_nchw, projector=projector)
                    for feat in tgt_feats.values()
                ) / max(1, len(tgt_feats))

                # DINO Teacher applies the same λ_sim to both src and tgt patch losses.
                losses["loss_dist_src"] = args.dist_lambda * loss_src
                losses["loss_dist_tgt"] = args.dist_lambda * loss_tgt
            elif isinstance(model, FasterRCNNWithAuxLatent):
                losses = model(images, targets_on_dev, aux_targets=aux_targets)
            else:
                losses = model(images, targets_on_dev)
            total_loss = sum(losses.values())
            total_loss.backward()
            if args.clip_grad: torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
            if args.warmup_iters and total_steps < args.warmup_iters:   # global step: warmup may span epochs (ends before 1st milestone)
                f = 0.001 + (1 - 0.001) * total_steps / args.warmup_iters
                for g in optimizer.param_groups: g["lr"] = g["base_lr_"] * f
            elif args.warmup_iters and total_steps == args.warmup_iters:
                for g in optimizer.param_groups: g["lr"] = g["base_lr_"]
            optimizer.step()

            epoch_loss += float(total_loss.detach())
            n_batches += 1
            total_steps += 1
            if total_steps % 50 == 0:
                with log_path.open("a") as f:
                    f.write(f"step {total_steps} epoch {epoch} loss {total_loss.item():.4f} parts {{ {', '.join(f'{k}={v.item():.4f}' for k,v in losses.items())} }}\n")
            if args.smoke and total_steps >= 5:
                break
        lr_scheduler.step()
        log(f"epoch {epoch + 1}/{args.epochs} avg_loss={epoch_loss/max(1,n_batches):.4f} time={time.time()-t0_train:.0f}s")

        # Per-epoch checkpoint so a crash doesn't waste hours.
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "args": vars(args),
        }
        if projector is not None:
            ckpt["projector"] = projector.state_dict()
        torch.save(ckpt, out_dir / "latest_epoch.pt")
        if args.smoke:
            break

    train_seconds = time.time() - t0_train
    log(f"training done in {train_seconds:.0f}s ({n_train} frames * {args.epochs} epochs)")

    # Evaluate on val.
    log("evaluating on val ...")
    model.eval()
    if isinstance(model, FasterRCNNWithAuxLatent):
        underlying = model.detector
    else:
        underlying = model
    all_preds: list[dict[str, torch.Tensor]] = []
    all_tgts: list[dict[str, torch.Tensor]] = []
    t0_eval = time.time()
    with torch.inference_mode():
        for images, targets in val_loader:
            images = [im.to(device) for im in images]
            preds = underlying(images)
            for p, t in zip(preds, targets):
                all_preds.append({k: v.detach().cpu() for k, v in p.items()})
                all_tgts.append({k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in t.items()})
            if args.smoke and len(all_preds) >= 4:
                break
    eval_seconds = time.time() - t0_eval

    metrics = evaluate_map(all_preds, all_tgts, KITTI_CLASSES, iou_threshold=0.5)
    log(f"eval: {metrics}")

    # Persist.
    final_ckpt = {"state_dict": model.state_dict(), "args": vars(args)}
    if projector is not None:
        final_ckpt["projector"] = projector.state_dict()
    torch.save(final_ckpt, out_dir / "last.pt")
    save_eval_json(
        metrics,
        Path(args.result).expanduser(),
        extra={
            "args": vars(args),
            "n_train": n_train,
            "n_val": n_val,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "device": torch.cuda.get_device_name(0),
        },
    )
    log(f"checkpoint -> {out_dir}/last.pt, metrics -> {args.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
