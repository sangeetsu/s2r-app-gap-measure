# SPDX-License-Identifier: Apache-2.0
"""Latent-space adaptation strategies for the Alpamayo + CARLA paper."""

from s2rgap.adaptation.feature_level import CoralAlignment, fit_coral
from s2rgap.adaptation.deep_coral import (
    deep_coral_loss,
    deep_coral_loss_with_mean,
    feature_covariance,
)
from s2rgap.adaptation.distillation import (
    PatchProjector,
    patch_distill_loss,
    stack_variable_spatial,
)
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup

__all__ = [
    "CoralAlignment",
    "fit_coral",
    "deep_coral_loss",
    "deep_coral_loss_with_mean",
    "feature_covariance",
    "PatchProjector",
    "patch_distill_loss",
    "stack_variable_spatial",
    "SpatialFeatureLookup",
]
