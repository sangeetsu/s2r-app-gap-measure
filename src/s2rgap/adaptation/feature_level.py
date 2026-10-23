# SPDX-License-Identifier: Apache-2.0
"""Feature-level latent-space adaptation (Strategy i, headline of the paper).

Implements CORAL [Sun & Saenko, 2016] in closed form: whiten source latents
using the source covariance, then re-color with the target covariance, plus a
mean shift. The transformation is fit once from a paired (sim, real) sample
and then applied to all sim latents the downstream pipeline consumes.

CORAL is chosen over MMD-based or adversarial alignment because:
  - it admits a closed-form solution (no optimization loop, fully reproducible)
  - it has been used in feature-level UDA since 2016 and is the canonical
    second-order baseline reviewers expect
  - on 4096-dim features it runs in seconds on CPU, no GPU overhead

Reference: Baochen Sun and Kate Saenko, "Deep CORAL: Correlation Alignment for
Deep Domain Adaptation," ECCV 2016 Workshops.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CoralAlignment:
    """Closed-form CORAL alignment transform fitted on (sim, real) latents.

    Apply with `aligned = transform.apply(z_sim_new)` for any new sim latents
    that share the same dimensionality as the fit data.
    """

    mean_sim: np.ndarray
    mean_real: np.ndarray
    M: np.ndarray
    eps: float = field(default=1e-5)

    def apply(self, z: np.ndarray) -> np.ndarray:
        if z.ndim != 2 or z.shape[1] != self.M.shape[0]:
            raise ValueError(
                f"input shape {z.shape} incompatible with transform dim {self.M.shape[0]}"
            )
        return (z - self.mean_sim) @ self.M + self.mean_real


def _sym_sqrt(matrix: np.ndarray, eps: float, invert: bool) -> np.ndarray:
    """Symmetric (inverse) square root of a positive-semi-definite matrix."""
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, eps, None)
    if invert:
        roots = 1.0 / np.sqrt(eigvals)
    else:
        roots = np.sqrt(eigvals)
    return (eigvecs * roots) @ eigvecs.T


def fit_coral(z_sim: np.ndarray, z_real: np.ndarray, eps: float = 1e-5) -> CoralAlignment:
    """Fit a closed-form CORAL transform from sim to real.

    Args:
        z_sim:   (N_sim,  D) sim-domain latents (e.g. CARLA frames).
        z_real:  (N_real, D) real-domain latents (e.g. KITTI frames).
        eps:     ridge added to the covariance diagonal for numerical stability.

    Returns:
        A `CoralAlignment` transform such that `transform.apply(z_sim_new)`
        produces sim latents with second-order statistics matched to z_real.
    """
    if z_sim.ndim != 2 or z_real.ndim != 2:
        raise ValueError("z_sim and z_real must be 2-D arrays")
    if z_sim.shape[1] != z_real.shape[1]:
        raise ValueError(
            f"feature dim mismatch: sim={z_sim.shape[1]}, real={z_real.shape[1]}"
        )

    # Compute means and covariances in float64 for stable eigendecompositions.
    z_sim64 = z_sim.astype(np.float64, copy=False)
    z_real64 = z_real.astype(np.float64, copy=False)

    mu_sim = z_sim64.mean(axis=0, keepdims=True)
    mu_real = z_real64.mean(axis=0, keepdims=True)

    cov_sim = np.cov(z_sim64, rowvar=False) + eps * np.eye(z_sim64.shape[1])
    cov_real = np.cov(z_real64, rowvar=False) + eps * np.eye(z_real64.shape[1])

    # Whitening: Sigma_sim^{-1/2}; coloring: Sigma_real^{1/2}.
    whiten = _sym_sqrt(cov_sim, eps, invert=True)
    color = _sym_sqrt(cov_real, eps, invert=False)
    M = whiten @ color

    return CoralAlignment(
        mean_sim=mu_sim.astype(np.float32),
        mean_real=mu_real.astype(np.float32),
        M=M.astype(np.float32),
        eps=eps,
    )


def domain_separation(z_a: np.ndarray, z_b: np.ndarray) -> float:
    """L2 distance between domain means, normalized by the pooled std.

    Same metric used in tools/encode_real_samples.py and tools/encode_carla_pilot.py.
    Lower means more aligned; values around the real-vs-real baseline indicate
    successful sim-to-real alignment.
    """
    pooled = np.concatenate([z_a, z_b], axis=0)
    return float(np.linalg.norm(z_a.mean(axis=0) - z_b.mean(axis=0)) / pooled.std())
