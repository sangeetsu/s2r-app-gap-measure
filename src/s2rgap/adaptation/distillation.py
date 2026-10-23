# SPDX-License-Identifier: Apache-2.0
"""Patch-level cosine distillation from a frozen foundation teacher.

Mirrors the DINO Teacher recipe (Larue et al., arXiv:2503.23220, 2025):

    L_sim = (1 / (N * H * W)) * sum_n sum_{i,j} (
        1 - cos( interp(MLP(student_feat))_{n,i,j} ,  teacher_feat_{n,i,j} )
    )

The student feature map (here, an FPN level from a ResNet-50 + FPN detector)
is bilinear-interpolated to the teacher's spatial grid, projected through a
small MLP to match the teacher's channel dim, and then per-patch cosine
distance is averaged over all patches and images.

Teacher features are cached on disk (frozen Alpamayo / Cosmos-Reason / Qwen3-VL
visual tower output, reshaped to (H_post, W_post, D); see
``FrameEncoder.encode_spatial``). At training time we read fp16 maps,
upcast to fp32 (or whatever dtype the student is in), and feed them here.

This module deliberately does *not* assume what the student's input shape is
or where the teacher's features come from -- it accepts already-extracted
tensors so it stays unit-testable on CPU without loading Alpamayo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "PatchProjector",
    "patch_distill_loss",
    "stack_variable_spatial",
]


class PatchProjector(nn.Module):
    """2-layer MLP projecting student channels to the teacher channel dim.

    Applied per-patch (1x1 conv on (B, C_in, H, W)) so it's parameter-light
    and structure-preserving. The DINO Teacher paper calls this MLP_g.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int = 1024) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W)
        return self.fc2(self.act(self.fc1(x)))


def patch_distill_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    projector: nn.Module | None = None,
    align_to: str = "teacher",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-patch cosine distillation loss.

    Args:
        student_feat: (B, C_s, H_s, W_s) feature map from the detector's
            backbone or FPN. Must be 4-D NCHW.
        teacher_feat: (B, C_t, H_t, W_t) feature map from the frozen teacher
            (e.g. Alpamayo). Must be 4-D NCHW. Channel-first (we keep
            teacher's per-patch (H_post, W_post, D) maps reshaped to NCHW
            before passing in).
        projector: optional channel-projection module (e.g. PatchProjector).
            Applied to student_feat to bring it to C_t. If None, requires
            C_s == C_t.
        align_to: "teacher" (default) resizes student to teacher grid; "student"
            resizes teacher to student grid. The DINO Teacher recipe aligns
            student to teacher.
        eps: numerical floor for the cosine denominator.

    Returns:
        Scalar loss tensor on the student's device/dtype.
    """
    if student_feat.dim() != 4:
        raise ValueError(f"student_feat must be 4-D NCHW, got {tuple(student_feat.shape)}")
    if teacher_feat.dim() != 4:
        raise ValueError(f"teacher_feat must be 4-D NCHW, got {tuple(teacher_feat.shape)}")
    if student_feat.shape[0] != teacher_feat.shape[0]:
        raise ValueError(
            f"batch mismatch: student {student_feat.shape[0]} vs teacher {teacher_feat.shape[0]}"
        )

    # Channel projection.
    s = projector(student_feat) if projector is not None else student_feat
    if s.shape[1] != teacher_feat.shape[1]:
        raise ValueError(
            f"channel mismatch after projection: student {s.shape[1]} vs teacher {teacher_feat.shape[1]} "
            "(pass a projector that maps to teacher channels)"
        )

    # Spatial alignment.
    if align_to == "teacher":
        s = F.interpolate(s, size=teacher_feat.shape[-2:], mode="bilinear", align_corners=False)
        t = teacher_feat
    elif align_to == "student":
        t = F.interpolate(
            teacher_feat, size=student_feat.shape[-2:], mode="bilinear", align_corners=False
        )
    else:
        raise ValueError(f"align_to must be 'teacher' or 'student', got {align_to!r}")

    # Per-patch cosine distance averaged over (B, H, W).
    # Match dtypes (teacher loaded fp16 from disk; student may be fp32).
    t = t.to(dtype=s.dtype)
    cos = F.cosine_similarity(s, t, dim=1, eps=eps)  # (B, H, W)
    return (1.0 - cos).mean()


def stack_variable_spatial(
    maps: list[torch.Tensor],
    target_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Stack a list of (H_i, W_i, D) maps into (B, D, H, W) NCHW.

    Qwen3-VL is native-resolution so a batch of frames may have different
    post-merger spatial dims. We resize all maps to ``target_hw`` (defaults
    to the smallest H, W in the batch) with bilinear interpolation, then
    permute to NCHW. The result is suitable for ``patch_distill_loss``.

    Args:
        maps: list of (H_i, W_i, D) tensors (typically fp16 from disk).
        target_hw: optional (H, W) to resize all maps to. Defaults to the
            per-batch minimum.

    Returns:
        Tensor of shape (B, D, H, W) on the same device/dtype as ``maps[0]``.
    """
    if not maps:
        raise ValueError("stack_variable_spatial got empty list")

    if target_hw is None:
        target_hw = (min(m.shape[0] for m in maps), min(m.shape[1] for m in maps))
    h, w = target_hw

    out = []
    for m in maps:
        # (H_i, W_i, D) -> (1, D, H_i, W_i)
        nchw = m.permute(2, 0, 1).unsqueeze(0)
        if nchw.shape[-2:] != (h, w):
            nchw = F.interpolate(nchw.float(), size=(h, w), mode="bilinear", align_corners=False).to(
                dtype=m.dtype
            )
        out.append(nchw)
    return torch.cat(out, dim=0)
