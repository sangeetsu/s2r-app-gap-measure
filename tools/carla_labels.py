# SPDX-License-Identifier: Apache-2.0
"""Project CARLA actor 3D bounding boxes to a 2D image-space label.

Given a CARLA camera and the world's actors at a specific tick, return a list
of KITTI-style detection records (class, bbox xyxy, truncation, occlusion).
We restrict to vehicle and walker classes since KITTI has 'Car', 'Pedestrian',
'Cyclist' as primary categories.

The projection follows the standard pinhole model with intrinsics derived from
the camera's image_size and FOV attributes. Camera extrinsics come from
`camera.get_transform()`; CARLA's left-handed coordinate convention is handled
by an explicit basis change.

This module is intentionally side-effect-free and importable both inside the
extractor and in any standalone label-generation pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

import carla


# KITTI-style detection class mapping. CARLA blueprint type IDs look like
# `vehicle.tesla.model3` / `walker.pedestrian.0001`. We coalesce all four-wheel
# vehicles to `Car`, all walkers to `Pedestrian`. Two-wheelers in CARLA show
# up as `vehicle.bh.crossbike` etc.; treat them as Cyclist.
TWO_WHEELED_KEYWORDS = ("bike", "moped", "harley", "kawasaki", "yamaha", "vespa", "ninja")


def classify_actor(actor: carla.Actor) -> str | None:
    type_id = actor.type_id
    if type_id.startswith("walker.pedestrian"):
        return "Pedestrian"
    if type_id.startswith("vehicle."):
        try:
            wheels = int(actor.attributes.get("number_of_wheels", "4"))
        except (TypeError, ValueError):
            wheels = 4
        if wheels < 4 or any(k in type_id for k in TWO_WHEELED_KEYWORDS):
            return "Cyclist"
        return "Car"
    return None


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_attributes(cls, image_width: int, image_height: int, hfov_deg: float) -> "CameraIntrinsics":
        # Pinhole camera: focal_x = (W/2) / tan(hfov/2). Vertical FOV is
        # implicit from the aspect ratio.
        focal = image_width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
        return cls(
            width=image_width,
            height=image_height,
            fx=focal,
            fy=focal,
            cx=image_width / 2.0,
            cy=image_height / 2.0,
        )


def _bbox_corners_world(actor: carla.Actor) -> np.ndarray:
    """Return the 8 world-space corners of an actor's local bounding box.

    CARLA stores each actor's bounding box in actor-local coordinates with the
    `extent` along x (length), y (width), z (height). The local origin sits at
    the actor's pivot, so we compose with `actor.get_transform()` to reach
    world coordinates.
    """
    bb = actor.bounding_box
    e = bb.extent  # half-sizes
    # 8 corners in local frame, applying bb.location offset.
    local = np.array(
        [
            [+e.x, +e.y, +e.z],
            [+e.x, +e.y, -e.z],
            [+e.x, -e.y, +e.z],
            [+e.x, -e.y, -e.z],
            [-e.x, +e.y, +e.z],
            [-e.x, +e.y, -e.z],
            [-e.x, -e.y, +e.z],
            [-e.x, -e.y, -e.z],
        ],
        dtype=np.float64,
    )
    local += np.array([bb.location.x, bb.location.y, bb.location.z], dtype=np.float64)

    actor_xform = actor.get_transform()
    actor_matrix = np.asarray(actor_xform.get_matrix(), dtype=np.float64)
    homog = np.concatenate([local, np.ones((8, 1))], axis=1)
    world = (actor_matrix @ homog.T).T[:, :3]
    return world


def _world_to_camera(world_pts: np.ndarray, camera: carla.Sensor) -> np.ndarray:
    """Transform world-space points into the camera's local frame."""
    inv = np.asarray(camera.get_transform().get_inverse_matrix(), dtype=np.float64)
    homog = np.concatenate([world_pts, np.ones((world_pts.shape[0], 1))], axis=1)
    cam_pts = (inv @ homog.T).T[:, :3]
    return cam_pts


def _project(cam_pts: np.ndarray, intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-frame points to image pixels using the standard pinhole.

    CARLA's camera coordinate convention is x=forward, y=right, z=up. The
    pinhole used by the rendering RGB camera maps that to image pixels via
    u = fx * (y / x) + cx, v = fy * (-z / x) + cy (depth is +x).

    Returns:
        pixels:    (N, 2) array of (u, v) image coordinates
        in_front:  (N,)   bool mask, True if the point is in front of the camera
    """
    x = cam_pts[:, 0]
    y = cam_pts[:, 1]
    z = cam_pts[:, 2]
    in_front = x > 1e-3
    safe_x = np.where(in_front, x, 1.0)
    u = intr.fx * (y / safe_x) + intr.cx
    v = intr.fy * (-z / safe_x) + intr.cy
    return np.stack([u, v], axis=-1), in_front


@dataclass
class Projected2DBBox:
    cls: str
    actor_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels, clipped
    truncation: float                          # 0.0 = fully in frame, 1.0 = entirely out
    in_front_corners: int                      # 0..8 corners with positive depth
    distance_m: float                          # ego-to-actor distance


def project_actors(
    world: carla.World,
    camera: carla.Sensor,
    intrinsics: CameraIntrinsics,
    ego_actor_id: int,
    min_visible_corners: int = 3,
    max_distance_m: float = 80.0,
    min_pixel_area: int = 50,
) -> list[Projected2DBBox]:
    """Return KITTI-style 2D detections for every visible actor in `world`.

    Args:
        world:                  current CARLA world snapshot (caller is responsible
                                for calling this on the same tick the camera image
                                belongs to).
        camera:                 CARLA RGB camera sensor whose transform gives
                                the extrinsics.
        intrinsics:             camera intrinsics derived from image size + FOV.
        ego_actor_id:           id of the ego vehicle, excluded from output.
        min_visible_corners:    drop boxes where fewer than this many of the
                                eight corners are in front of the camera.
        max_distance_m:         drop actors farther than this from the camera.
        min_pixel_area:         drop boxes whose clipped area is below this.

    Returns:
        list of Projected2DBBox (one per accepted actor).
    """
    cam_loc = camera.get_transform().location
    cam_pos = np.array([cam_loc.x, cam_loc.y, cam_loc.z], dtype=np.float64)

    detections: list[Projected2DBBox] = []
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego_actor_id:
            continue
        cls = classify_actor(actor)
        if cls is None:
            continue
        det = _project_one_actor(
            actor=actor,
            cls=cls,
            camera=camera,
            intr=intrinsics,
            cam_pos=cam_pos,
            min_visible_corners=min_visible_corners,
            max_distance_m=max_distance_m,
            min_pixel_area=min_pixel_area,
        )
        if det is not None:
            detections.append(det)
    for actor in world.get_actors().filter("walker.pedestrian.*"):
        cls = classify_actor(actor)
        if cls is None:
            continue
        det = _project_one_actor(
            actor=actor,
            cls=cls,
            camera=camera,
            intr=intrinsics,
            cam_pos=cam_pos,
            min_visible_corners=min_visible_corners,
            max_distance_m=max_distance_m,
            min_pixel_area=min_pixel_area,
        )
        if det is not None:
            detections.append(det)
    return detections


def _project_one_actor(
    actor: carla.Actor,
    cls: str,
    camera: carla.Sensor,
    intr: CameraIntrinsics,
    cam_pos: np.ndarray,
    min_visible_corners: int,
    max_distance_m: float,
    min_pixel_area: int,
) -> Projected2DBBox | None:
    actor_loc = actor.get_location()
    actor_pos = np.array([actor_loc.x, actor_loc.y, actor_loc.z], dtype=np.float64)
    distance = float(np.linalg.norm(actor_pos - cam_pos))
    if distance > max_distance_m:
        return None

    world_corners = _bbox_corners_world(actor)
    cam_corners = _world_to_camera(world_corners, camera)
    pixels, in_front = _project(cam_corners, intr)
    visible_count = int(in_front.sum())
    if visible_count < min_visible_corners:
        return None

    # Use only in-front corners for the AABB; if some corners are behind the
    # camera we under-estimate the box but never project nonsense.
    visible_pixels = pixels[in_front]
    if visible_pixels.size == 0:
        return None

    x1 = float(visible_pixels[:, 0].min())
    y1 = float(visible_pixels[:, 1].min())
    x2 = float(visible_pixels[:, 0].max())
    y2 = float(visible_pixels[:, 1].max())

    # Compute truncation as the fraction of the unclipped box that falls
    # outside the image rectangle.
    full_w = max(1e-6, x2 - x1)
    full_h = max(1e-6, y2 - y1)
    full_area = full_w * full_h

    cx1 = max(0.0, min(intr.width, x1))
    cy1 = max(0.0, min(intr.height, y1))
    cx2 = max(0.0, min(intr.width, x2))
    cy2 = max(0.0, min(intr.height, y2))
    clipped_w = max(0.0, cx2 - cx1)
    clipped_h = max(0.0, cy2 - cy1)
    clipped_area = clipped_w * clipped_h

    if clipped_area < min_pixel_area:
        return None

    truncation = 1.0 - clipped_area / full_area

    return Projected2DBBox(
        cls=cls,
        actor_id=actor.id,
        bbox=(cx1, cy1, cx2, cy2),
        truncation=truncation,
        in_front_corners=visible_count,
        distance_m=distance,
    )


def to_kitti_label_line(det: Projected2DBBox) -> str:
    """Format a Projected2DBBox as a single KITTI label-file line.

    KITTI label format (training):
        type truncation occlusion alpha bbox.x1 bbox.y1 bbox.x2 bbox.y2 \
        height width length pos.x pos.y pos.z rotation_y

    We don't infer 3D dimensions, alpha, or rotation_y from the projected
    actor (we have them in CARLA, but the paper's downstream task is 2D
    detection, so we leave the 3D fields as -1 / 0).
    """
    occlusion = 0  # CARLA doesn't expose occlusion directly; mark as "unknown" via 3
    if det.in_front_corners < 6:
        occlusion = 1
    if det.in_front_corners < 4:
        occlusion = 2
    x1, y1, x2, y2 = det.bbox
    return (
        f"{det.cls} {det.truncation:.2f} {occlusion} 0.00 "
        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
        f"-1 -1 -1 -1000 -1000 -1000 -10"
    )


def detections_to_kitti_label_text(detections: Iterable[Projected2DBBox]) -> str:
    return "\n".join(to_kitti_label_line(d) for d in detections) + ("\n" if detections else "")
