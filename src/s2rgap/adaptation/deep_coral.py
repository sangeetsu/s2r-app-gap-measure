# SPDX-License-Identifier: Apache-2.0
"""Deep CORAL (Sun & Saenko, ECCV-W 2016) — distributional feature alignment.

Loss (Eq. 1 of the paper):

    L_CORAL = (1 / (4 d^2)) * || C_S - C_T ||_F^2

where C_S, C_T are unbiased batch-level covariance matrices of source and
target features, and d is the feature dimensionality.

Used as an auxiliary loss when training a detector: each iteration's batch
contains N source images (labeled, with detection targets) and N target
images (unlabeled — just images). Both are passed through the detector
backbone; we capture the pooled feature for each, then add this loss to the
standard detection loss.

This replaces the per-frame regression aux head (Pattern A from
plan_2026-05-12.md) which saturated at the constant predictor — see
plan_2026-05-13.md §2 for the diagnosis.
"""

from __future__ import annotations

import torch
from torch import Tensor


def feature_covariance(z: Tensor, eps: float = 0.0) -> Tensor:
    """Unbiased batch covariance with mean removal.

    Args:
        z:   (N, D) feature matrix. N >= 2.
        eps: optional ridge added to the diagonal for numerical stability.
             Default 0.0; pass a small value (e.g. 1e-5) if you intend to
             invert C or feed it to a gradient-sensitive operation.

    Returns:
        (D, D) covariance matrix.
    """
    if z.dim() != 2:
        raise ValueError(f"expected 2-D feature batch, got shape {tuple(z.shape)}")
    n = z.size(0)
    if n < 2:
        raise ValueError(f"need at least 2 samples for batch covariance, got {n}")
    mu = z.mean(dim=0, keepdim=True)
    centered = z - mu
    cov = centered.t() @ centered / (n - 1)
    if eps > 0:
        d = cov.size(0)
        cov = cov + eps * torch.eye(d, device=cov.device, dtype=cov.dtype)
    return cov


def deep_coral_loss(
    z_src: Tensor,
    z_tgt: Tensor,
    normalize_features: bool = False,
) -> Tensor:
    """Deep CORAL loss between two feature batches.

    Args:
        z_src: (N_s, D) source-domain features (e.g. CARLA backbone-pool).
        z_tgt: (N_t, D) target-domain features (e.g. KITTI backbone-pool).
        normalize_features: if True, z-score features per-channel using the
            *pooled* (source ∪ target) standard deviation before computing
            covariance. This prevents tiny-activation channels from dominating
            the loss scale — without it, post-pooling FPN features can produce
            covariance ~1e-6 and effectively zero CORAL gradient (observed
            empirically on our setup, see plan_2026-05-13.md).

    Returns:
        scalar loss tensor, gradient flows back through z_src and z_tgt.
    """
    if z_src.size(1) != z_tgt.size(1):
        raise ValueError(
            f"source dim {z_src.size(1)} != target dim {z_tgt.size(1)}"
        )
    if normalize_features:
        all_z = torch.cat([z_src, z_tgt], dim=0)
        sigma = all_z.std(dim=0, keepdim=True).clamp(min=1e-6)
        z_src = z_src / sigma
        z_tgt = z_tgt / sigma
    d = z_src.size(1)
    c_src = feature_covariance(z_src)
    c_tgt = feature_covariance(z_tgt)
    diff = c_src - c_tgt
    return diff.pow(2).sum() / (4.0 * d * d)


def deep_coral_loss_with_mean(z_src: Tensor, z_tgt: Tensor, mean_weight: float = 0.0) -> Tensor:
    """Optional extension: also penalize first-moment (mean) mismatch.

    The original Deep CORAL paper only matches second-order statistics. Some
    later UDA variants add a mean-alignment term ||mu_S - mu_T||^2 / D. Pass
    mean_weight > 0 to include it.
    """
    coral = deep_coral_loss(z_src, z_tgt)
    if mean_weight <= 0:
        return coral
    mu_src = z_src.mean(dim=0)
    mu_tgt = z_tgt.mean(dim=0)
    mean_term = (mu_src - mu_tgt).pow(2).mean()
    return coral + mean_weight * mean_term
