# SPDX-License-Identifier: Apache-2.0
"""Detection datasets: KITTI (real, eval target) and CARLA (sim, train source)."""

from s2rgap.data.kitti import KITTIDetection, KITTI_CLASSES
from s2rgap.data.carla import CARLADetection, CARLA_CLASSES

__all__ = ["KITTIDetection", "KITTI_CLASSES", "CARLADetection", "CARLA_CLASSES"]
