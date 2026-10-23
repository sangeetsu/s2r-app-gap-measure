# SPDX-License-Identifier: Apache-2.0
"""Faster R-CNN with an optional auxiliary head that predicts a frozen
foundation-model latent for the input frame.

The base detector is `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
We register a forward hook on the backbone (FPN) to capture the deepest
feature map ("pool"), pool it to a 256-D vector, and project to the target
foundation-latent dimension via a 2-layer MLP. Loss = standard FR-CNN loss +
λ · MSE(aux_pred, aux_target).

This is the architecture for Strategy (i) in the paper. Targets are typically
CORAL-aligned Alpamayo latents, but the same scaffold supports DINOv2 or any
other foundation encoder by swapping the target source.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# FPN out_channels for resnet50_fpn is 256. Pool over spatial dims to get (B, 256).
FPN_OUT_CHANNELS = 256


def build_detector(num_classes: int = 4, init: str = "coco", arch: str = "fasterrcnn"):
    """arch: fasterrcnn (default; all runs before 2026-10-16) | fcos | retinanet (torchvision R50-FPN; imagenet/scratch init only)
    | ddetr (Deformable DETR, HF transformers, ImageNet ResNet-50 backbone) | frcnn_swin (Faster R-CNN, ImageNet Swin-T + FPN) — 2026-10-17, E5."""
    if arch == "ddetr":
        if init == "coco": raise ValueError("ddetr: imagenet/scratch only")
        return DeformableDETRDetector(num_classes, init)
    if arch == "frcnn_swin":
        if init == "coco": raise ValueError("frcnn_swin: imagenet/scratch only")
        return FasterRCNN(SwinTFPN(init), num_classes=num_classes)
    if arch != "fasterrcnn":
        from torchvision.models import ResNet50_Weights
        from torchvision.models.detection import fcos_resnet50_fpn, retinanet_resnet50_fpn
        if init == "coco":
            raise ValueError("coco init is only wired for fasterrcnn")
        wb = ResNet50_Weights.IMAGENET1K_V1 if init == "imagenet" else None
        fn = {"fcos": fcos_resnet50_fpn, "retinanet": retinanet_resnet50_fpn}[arch]
        return fn(weights=None, weights_backbone=wb, num_classes=num_classes)
    return _build_fasterrcnn(num_classes, init)


def load_detector_from_ckpt(path, device):
    """Rebuild the architecture recorded in the checkpoint's saved args (fasterrcnn when absent) and load weights."""
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck.get("args") or {}
    m = build_detector(num_classes=4, init=a.get("init", "coco") if a.get("arch", "fasterrcnn") != "fasterrcnn" else "coco", arch=a.get("arch", "fasterrcnn"))
    m.load_state_dict(ck.get("state_dict") or ck.get("model") or ck)
    return m.to(device).eval()


class SwinTFPN(nn.Module):
    """torchvision Swin-T (ImageNet-1k) stages 1-4 (strides 4/8/16/32, NHWC -> NCHW) + FPN(256) + max-pool level: drop-in for the R50-FPN body."""
    def __init__(self, init: str = "imagenet"):
        super().__init__()
        from torchvision.models import swin_t, Swin_T_Weights
        from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
        self.body = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if init == "imagenet" else None).features
        self.fpn = FeaturePyramidNetwork([96, 192, 384, 768], 256, extra_blocks=LastLevelMaxPool())
        self.out_channels = 256
    def forward(self, x):
        outs = OrderedDict()
        for i, m in enumerate(self.body):
            x = m(x)
            if i in (1, 3, 5, 7): outs[str(len(outs))] = x.permute(0, 3, 1, 2).contiguous()
        return self.fpn(outs)


class DeformableDETRDetector(nn.Module):
    """Deformable DETR (Zhu et al., ICLR 2021; HF transformers implementation, box refinement on, single stage, 300 queries) behind the torchvision
    detection API: forward(list[CHW in 0..1], targets) -> {"loss"} in train, list of {boxes xyxy px, labels 1..C, scores} in eval. Images are
    resized with the torchvision rule (shorter side 800, longer <= 1333), ImageNet-normalised, zero-padded with a pixel mask."""
    def __init__(self, num_classes: int = 4, init: str = "imagenet", min_size: int = 800, max_size: int = 1333):
        super().__init__()
        from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection
        cfg = DeformableDetrConfig(use_timm_backbone=False, backbone="microsoft/resnet-50", use_pretrained_backbone=(init == "imagenet"),
                                   backbone_kwargs={"out_indices": [2, 3, 4]}, num_labels=num_classes - 1, num_queries=300, with_box_refine=True, two_stage=False)
        self.model = DeformableDetrForObjectDetection(cfg); self.C = num_classes - 1; self.min_size, self.max_size = min_size, max_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1), persistent=False)
    def _prep(self, images):
        xs, sizes = [], []
        for im in images:
            h, w = im.shape[-2:]; s = min(self.min_size / min(h, w), self.max_size / max(h, w)); nh, nw = int(round(h * s)), int(round(w * s))
            xs.append((F.interpolate(im[None], size=(nh, nw), mode="bilinear", align_corners=False)[0] - self.mean) / self.std); sizes.append((h, w, nh, nw))
        H = -(-max(z[2] for z in sizes) // 32) * 32; W = -(-max(z[3] for z in sizes) // 32) * 32
        pix = images[0].new_zeros(len(xs), 3, H, W); mask = torch.zeros(len(xs), H, W, dtype=torch.long, device=pix.device)
        for i, (x, z) in enumerate(zip(xs, sizes)): pix[i, :, :z[2], :z[3]] = x; mask[i, :z[2], :z[3]] = 1
        return pix, mask, sizes
    def forward(self, images, targets=None):
        pix, mask, sizes = self._prep(images)
        if self.training:
            lab = []
            for t, (h, w, _, _) in zip(targets, sizes):
                b = t["boxes"].float(); cx = (b[:, 0] + b[:, 2]) / 2 / w; cy = (b[:, 1] + b[:, 3]) / 2 / h; bw = (b[:, 2] - b[:, 0]) / w; bh = (b[:, 3] - b[:, 1]) / h
                lab.append({"class_labels": t["labels"] - 1, "boxes": torch.stack([cx, cy, bw, bh], 1)})
            return {"loss": self.model(pixel_values=pix, pixel_mask=mask, labels=lab).loss}
        out = self.model(pixel_values=pix, pixel_mask=mask); prob = out.logits.sigmoid(); res = []
        for i, (h, w, _, _) in enumerate(sizes):
            sc, idx = prob[i].flatten().topk(min(100, prob[i].numel())); q = idx // self.C; c = idx % self.C; bx = out.pred_boxes[i][q]
            xyxy = torch.stack([(bx[:, 0] - bx[:, 2] / 2) * w, (bx[:, 1] - bx[:, 3] / 2) * h, (bx[:, 0] + bx[:, 2] / 2) * w, (bx[:, 1] + bx[:, 3] / 2) * h], 1)
            res.append({"boxes": xyxy, "labels": c + 1, "scores": sc})
        return res


def _build_fasterrcnn(num_classes: int = 4, init: str = "coco") -> FasterRCNN:
    """Return a torchvision Faster R-CNN with the heads sized to `num_classes`.

    num_classes counts the background class. For our pipeline that's 4
    (background + Car + Pedestrian + Cyclist).
    init: "coco"     = full COCO-detection weights (backbone+FPN+RPN+heads) — real-world DETECTION pretraining;
          "imagenet" = ImageNet-classification ResNet-50 backbone only; FPN/RPN/heads learn detection from the
                       training data alone (standard source-only setup in sim-to-real detection benchmarks);
          "scratch"  = no pretrained weights at all.
    """
    if init == "coco":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    elif init == "imagenet":
        from torchvision.models import ResNet50_Weights
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V1)
    elif init == "scratch":
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    else:
        raise ValueError(f"unknown init {init!r}")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class AuxHead(nn.Module):
    """2-layer MLP that maps pooled FPN features to a foundation-latent target."""

    def __init__(self, in_channels: int = FPN_OUT_CHANNELS, hidden: int = 1024, out_dim: int = 4096):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, fpn_feature: torch.Tensor) -> torch.Tensor:
        # fpn_feature: (B, C, H, W)
        pooled = self.pool(fpn_feature).flatten(1)  # (B, C)
        return self.fc(pooled)


class FasterRCNNWithAuxLatent(nn.Module):
    """Wrap a torchvision FasterRCNN with a per-image auxiliary latent regression.

    Training: returns the detector loss dict augmented with `loss_aux`
    (already scaled by `aux_lambda`).
    Eval: returns the detector's prediction list (unchanged).

    Args:
        num_classes: incl. background. 4 for our paper.
        aux_dim_out: target foundation-latent dimension. 4096 for Alpamayo,
                     1024 for DINOv2 ViT-L/14, etc.
        aux_lambda: weight applied to MSE auxiliary loss before adding to the
                    detector loss. 0.1 is a reasonable default; do a sweep.
        aux_feature_key: which FPN level to take. "pool" is the coarsest level
                         (smallest spatial dim, most semantic).
    """

    def __init__(
        self,
        num_classes: int = 4,
        aux_dim_out: int = 4096,
        aux_lambda: float = 0.1,
        aux_feature_key: str = "pool",
    ) -> None:
        super().__init__()
        self.detector = build_detector(num_classes=num_classes)
        self.aux_head = AuxHead(out_dim=aux_dim_out)
        self.aux_lambda = aux_lambda
        self.aux_feature_key = aux_feature_key
        self._captured: dict[str, torch.Tensor] = {}
        self.detector.backbone.register_forward_hook(self._capture)

    def _capture(self, _module: nn.Module, _inputs: Any, output: dict[str, torch.Tensor]) -> None:
        # backbone (FPN) returns an OrderedDict of multi-scale feature maps
        # keyed "0", "1", "2", "3", "pool". Cache the one we want.
        if self.aux_feature_key in output:
            self._captured["fpn"] = output[self.aux_feature_key]

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
        aux_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        # torchvision's FR-CNN takes a list of (C, H, W) tensors.
        det_out = self.detector(images, targets)
        if self.training and aux_targets is not None:
            fpn_feature = self._captured["fpn"]
            # When the FPN is wrapped in DDP or similar, multiple images may
            # share a batch dim; ensure shapes align.
            if fpn_feature.shape[0] != aux_targets.shape[0]:
                raise RuntimeError(
                    f"FPN batch {fpn_feature.shape[0]} != aux_targets batch {aux_targets.shape[0]}"
                )
            aux_pred = self.aux_head(fpn_feature)
            aux_loss = F.mse_loss(aux_pred, aux_targets)
            det_out["loss_aux"] = self.aux_lambda * aux_loss
        return det_out

    @torch.inference_mode()
    def predict(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        self.eval()
        return self.detector(images)
