#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract CARLA RGB frames at KITTI-rectified sensor calibration.

Single forward-facing camera, ego on autopilot, optional NPC traffic for visual
variety. Sweeps over (town, weather) configurations. Output is JPEG q95 frames
plus a per-cell manifest (jsonl) so the encoder/loader pipelines can find them.

This is the CoRL 2026 Alpamayo+CARLA paper's synthetic source. It deliberately
does NOT load Alpamayo or do any inference — frame extraction and latent
encoding are decoupled so the encoder can change without re-running CARLA.

Usage (smoke):
    python tools/extract_carla_frames.py \
        --towns Town01 \
        --weathers ClearNoon \
        --frames-per-cell 100 \
        --out-dir ~/datasets/carla_extracted \
        --warmup-ticks 30 \
        --skip-ticks 5
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import socket
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import carla
from carla_labels import (
    CameraIntrinsics,
    detections_to_kitti_label_text,
    project_actors,
)


# KITTI-rectified left-camera calibration: 1242 x 375, focal_x ~ 721.5 px.
# H-FOV = 2 * atan(1242 / (2 * 721.5)) ≈ 81.4°.
KITTI_IMAGE_WIDTH = 1242
KITTI_IMAGE_HEIGHT = 375
KITTI_HFOV_DEG = 81.43

# Camera mount: forward-facing, mid-roof of the ego vehicle, no tilt.
CAMERA_TRANSFORM = carla.Transform(
    carla.Location(x=1.6, y=0.0, z=1.65),
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
)

# Built-in CARLA weather presets we sweep over.
WEATHER_PRESETS: dict[str, carla.WeatherParameters] = {
    "ClearNoon": carla.WeatherParameters.ClearNoon,
    "OvercastNoon": carla.WeatherParameters.CloudyNoon,
    "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
    "ClearSunset": carla.WeatherParameters.ClearSunset,
    "WetNoon": carla.WeatherParameters.WetNoon,
    "MidRainSunset": carla.WeatherParameters.MidRainSunset,
    "HardRainNight": carla.WeatherParameters.HardRainNight,
    "ClearNight": carla.WeatherParameters.ClearNight,
}

DEFAULT_SERVER_CONFIG = REPO_ROOT / "carla_server_profiles.toml"


# ---------------------------------------------------------------------------
# Server profile / connection management (mirrors carla_alpamayo_bridge.py
# but decoupled so the extractor doesn't import the bridge's heavy deps).
# ---------------------------------------------------------------------------


@dataclass
class ServerProfile:
    name: str
    host: str = "127.0.0.1"
    port: int = 2000
    tm_port: int = 8000
    connect_timeout_seconds: float = 10.0
    startup_timeout_seconds: float = 90.0
    launch_if_unreachable: bool = False
    terminate_on_exit: bool = False
    working_directory: str | None = None
    launch_command: list[str] | None = None
    description: str | None = None

    @classmethod
    def from_table(cls, name: str, table: dict[str, Any]) -> "ServerProfile":
        launch_command = table.get("launch_command")
        if launch_command is not None and not isinstance(launch_command, list):
            raise ValueError(f"profile `{name}` launch_command must be a list of strings")
        return cls(
            name=name,
            host=table.get("host", "127.0.0.1"),
            port=int(table.get("port", 2000)),
            tm_port=int(table.get("tm_port", 8000)),
            connect_timeout_seconds=float(table.get("connect_timeout_seconds", 10.0)),
            startup_timeout_seconds=float(table.get("startup_timeout_seconds", 90.0)),
            launch_if_unreachable=bool(table.get("launch_if_unreachable", False)),
            terminate_on_exit=bool(table.get("terminate_on_exit", False)),
            working_directory=table.get("working_directory"),
            launch_command=launch_command,
            description=table.get("description"),
        )


def load_server_profile(config_path: Path, profile_name: str | None) -> ServerProfile:
    if not config_path.exists():
        raise FileNotFoundError(f"server config not found: {config_path}")
    with config_path.open("rb") as h:
        cfg = tomllib.load(h)
    profiles = {n: ServerProfile.from_table(n, t) for n, t in cfg.get("profiles", {}).items()}
    name = profile_name or cfg.get("active_profile") or "local_package"
    if name not in profiles:
        raise KeyError(f"profile `{name}` not in {sorted(profiles)}")
    return profiles[name]


def try_connect(profile: ServerProfile) -> carla.Client | None:
    try:
        with socket.create_connection((profile.host, profile.port), timeout=2.0):
            pass
    except OSError:
        return None
    try:
        client = carla.Client(profile.host, profile.port)
        client.set_timeout(profile.connect_timeout_seconds)
        client.get_world()
        return client
    except RuntimeError:
        return None


def ensure_server(profile: ServerProfile) -> tuple[carla.Client, subprocess.Popen | None]:
    client = try_connect(profile)
    if client is not None:
        log(f"connected to existing CARLA at {profile.host}:{profile.port}")
        return client, None

    if not profile.launch_if_unreachable:
        raise RuntimeError(f"CARLA at {profile.host}:{profile.port} unreachable and no launch command")

    if not profile.launch_command:
        raise RuntimeError(f"profile `{profile.name}` missing launch_command")

    cwd = Path(profile.working_directory).expanduser() if profile.working_directory else None
    log(f"launching CARLA via profile `{profile.name}` ...")
    proc = subprocess.Popen(profile.launch_command, cwd=cwd)

    deadline = time.monotonic() + profile.startup_timeout_seconds
    while time.monotonic() < deadline:
        client = try_connect(profile)
        if client is not None:
            log(f"connected to launched CARLA at {profile.host}:{profile.port}")
            return client, proc
        time.sleep(2.0)

    raise RuntimeError(f"CARLA did not become ready within {profile.startup_timeout_seconds}s")


# ---------------------------------------------------------------------------
# Sensor stream
# ---------------------------------------------------------------------------


@dataclass
class FrameSample:
    sim_frame: int
    sim_time: float
    image_rgb: np.ndarray


class SyncCamera:
    def __init__(self, world: carla.World, parent: carla.Actor, width: int, height: int, fov: float):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        bp.set_attribute("sensor_tick", "0.0")
        self._q: queue.Queue[FrameSample] = queue.Queue()
        self.sensor = world.spawn_actor(bp, CAMERA_TRANSFORM, attach_to=parent)
        self.sensor.listen(self._on_image)

    def _on_image(self, image: carla.Image) -> None:
        raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        rgb = raw[:, :, :3][:, :, ::-1].copy()
        self._q.put(FrameSample(sim_frame=image.frame, sim_time=image.timestamp, image_rgb=rgb))

    def get_for_frame(self, target_frame: int, timeout: float = 10.0) -> FrameSample:
        while True:
            try:
                s = self._q.get(timeout=timeout)
            except queue.Empty as exc:
                raise RuntimeError(f"camera timed out waiting for frame {target_frame}") from exc
            if s.sim_frame < target_frame:
                continue
            if s.sim_frame != target_frame:
                raise RuntimeError(f"camera returned frame {s.sim_frame}, expected {target_frame}")
            return s

    def destroy(self) -> None:
        try:
            self.sensor.stop()
        except RuntimeError:
            pass
        try:
            self.sensor.destroy()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# NPC traffic for visual variety
# ---------------------------------------------------------------------------


def spawn_npc_traffic(
    world: carla.World,
    tm: carla.TrafficManager,
    n_vehicles: int,
    rng: random.Random,
    bike_fraction: float = 0.15,
) -> list[carla.Actor]:
    """Spawn a mix of 4-wheel vehicles and 2-wheelers (the latter become
    the Cyclist class in our label format). Drops the prior 4-wheel-only
    filter that was producing single-class CARLA data.
    """
    bp_lib = world.get_blueprint_library()
    all_vehicle_bps = list(bp_lib.filter("vehicle.*"))
    cars = [b for b in all_vehicle_bps if int(b.get_attribute("number_of_wheels")) >= 4]
    bikes = [b for b in all_vehicle_bps if int(b.get_attribute("number_of_wheels")) < 4]

    spawn_points = world.get_map().get_spawn_points()
    rng.shuffle(spawn_points)

    n_bikes_target = int(round(bike_fraction * n_vehicles))
    n_cars_target = n_vehicles - n_bikes_target

    actors: list[carla.Actor] = []
    n_cars = 0
    n_bikes = 0
    for sp in spawn_points[: n_vehicles * 3]:  # over-provision; spawns may fail
        if n_cars + n_bikes >= n_vehicles:
            break
        # Alternate so we don't exhaust car spawn points before placing bikes.
        want_bike = (n_bikes < n_bikes_target) and (bikes and rng.random() < bike_fraction * 1.5)
        pool = bikes if want_bike and bikes else cars
        if not pool:
            continue
        bp = rng.choice(pool)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        bp.set_attribute("role_name", "npc")
        v = world.try_spawn_actor(bp, sp)
        if v is None:
            continue
        v.set_autopilot(True, tm.get_port())
        actors.append(v)
        if want_bike and bikes:
            n_bikes += 1
        else:
            n_cars += 1
    log(f"spawned {n_cars + n_bikes}/{n_vehicles} NPC vehicles (cars={n_cars}, two-wheelers={n_bikes})")
    return actors


def spawn_npc_walkers(
    client: carla.Client,
    world: carla.World,
    n_walkers: int,
    rng: random.Random,
) -> tuple[list[carla.Actor], list[carla.Actor]]:
    """Spawn pedestrians with WalkerAIController for natural ambulation.

    Returns (walker_actors, controller_actors). Both lists must be destroyed
    at cell end — controllers first via .stop() then via DestroyActor.
    """
    bp_lib = world.get_blueprint_library()
    walker_bps = list(bp_lib.filter("walker.pedestrian.*"))
    if not walker_bps:
        log("no walker blueprints available — skipping pedestrians")
        return [], []
    controller_bp = bp_lib.find("controller.ai.walker")

    # 1. Sample nav-mesh locations.
    transforms: list[carla.Transform] = []
    attempts = 0
    while len(transforms) < n_walkers and attempts < n_walkers * 5:
        attempts += 1
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            transforms.append(carla.Transform(loc))

    # 2. Spawn walker actors batched.
    batch = []
    walker_speeds: list[float] = []
    for t in transforms:
        bp = rng.choice(walker_bps)
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        if bp.has_attribute("speed"):
            recs = bp.get_attribute("speed").recommended_values
            walker_speeds.append(float(recs[1]) if len(recs) > 1 else 1.4)  # walking
        else:
            walker_speeds.append(1.4)
        bp.set_attribute("role_name", "npc_walker")
        batch.append(carla.command.SpawnActor(bp, t))

    walker_results = client.apply_batch_sync(batch, True)
    walker_ids: list[int] = []
    walker_kept_speeds: list[float] = []
    for r, sp in zip(walker_results, walker_speeds):
        if not r.error:
            walker_ids.append(r.actor_id)
            walker_kept_speeds.append(sp)
    walker_actors = list(world.get_actors(walker_ids))

    # 3. Spawn AI controllers parented to each walker.
    controller_batch = [
        carla.command.SpawnActor(controller_bp, carla.Transform(), parent_id=wid)
        for wid in walker_ids
    ]
    controller_results = client.apply_batch_sync(controller_batch, True)
    controller_ids = [r.actor_id for r in controller_results if not r.error]
    controller_actors = list(world.get_actors(controller_ids))

    # 4. Tick once so the controllers register, then start them.
    world.tick()
    for ctrl, speed in zip(controller_actors, walker_kept_speeds):
        try:
            ctrl.start()
            ctrl.go_to_location(world.get_random_location_from_navigation())
            ctrl.set_max_speed(speed)
        except RuntimeError:
            pass

    log(f"spawned {len(walker_actors)} pedestrians with {len(controller_actors)} controllers")
    return walker_actors, controller_actors


def spawn_ego(world: carla.World, tm: carla.TrafficManager, rng: random.Random) -> carla.Actor:
    """Spawn the ego vehicle, with retries on the TM's flaky 2 s timeout under load."""
    bp_lib = world.get_blueprint_library()
    bp = bp_lib.find("vehicle.tesla.model3")
    bp.set_attribute("role_name", "extractor_ego")
    spawn_points = list(world.get_map().get_spawn_points())
    rng.shuffle(spawn_points)
    for sp in spawn_points:
        v = world.try_spawn_actor(bp, sp)
        if v is None:
            continue
        # TM register_vehicle can timeout when the world has 150+ actors.
        # Retry up to 3 times with brief backoff before giving up on this
        # actor; if all retries fail, destroy and move on to the next spawn.
        for attempt in range(3):
            try:
                v.set_autopilot(True, tm.get_port())
                log(f"spawned ego at {sp.location}")
                return v
            except RuntimeError as exc:
                if "timeout" in str(exc).lower() and attempt < 2:
                    log(f"TM timeout enabling autopilot (attempt {attempt + 1}/3), retrying ...")
                    time.sleep(0.5)
                    continue
                # Out of retries (or different error) — burn this actor, try next sp
                try:
                    v.destroy()
                except RuntimeError:
                    pass
                break
    raise RuntimeError("could not spawn ego vehicle after exhausting spawn points")


def cleanup_actors(actors: list[carla.Actor]) -> None:
    for a in actors:
        try:
            a.destroy()
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# Per-cell extraction
# ---------------------------------------------------------------------------


def _cleanup_actors(
    client: carla.Client,
    cam: "SyncCamera | None",
    ego: "carla.Actor | None",
    npc_actors: list[carla.Actor],
    walker_actors: list[carla.Actor] | None = None,
    controller_actors: list[carla.Actor] | None = None,
) -> None:
    """Best-effort cleanup that avoids CARLA's std::runtime_error from the TM
    querying actors mid-destroy. Stop sensor + walker controllers first,
    disable autopilot, batch destroy. Cell-level TM and world-settings are
    reset by the caller."""
    if cam is not None:
        try:
            cam.sensor.stop()
        except RuntimeError:
            pass
    # Walker controllers must be stopped before their parent walkers die.
    for ctrl in controller_actors or []:
        try:
            ctrl.stop()
        except RuntimeError:
            pass
    all_vehicles = ([ego] if ego is not None else []) + list(npc_actors)
    for v in all_vehicles:
        try:
            v.set_autopilot(False)
        except RuntimeError:
            pass
    actor_ids: list[int] = []
    if cam is not None:
        actor_ids.append(cam.sensor.id)
    # Order matters: destroy controllers before walkers (CARLA expects this).
    actor_ids.extend(c.id for c in (controller_actors or []))
    actor_ids.extend(w.id for w in (walker_actors or []))
    actor_ids.extend(v.id for v in all_vehicles)
    if actor_ids:
        try:
            client.apply_batch_sync(
                [carla.command.DestroyActor(aid) for aid in actor_ids],
                True,
            )
        except RuntimeError:
            pass


def extract_one_cell(
    client: carla.Client,
    town: str,
    weather_name: str,
    weather_params: carla.WeatherParameters,
    args: argparse.Namespace,
    cell_dir: Path,
    rng: random.Random,
) -> int:
    """Extract args.frames_per_cell frames over multiple short ego trajectories.

    Each trajectory respawns ego at a random map spawn point — this breaks the
    long-run autocorrelation that single-trajectory extraction produces. NPC
    traffic is spawned once at the start of the cell and shared across cycles
    so the surrounding world keeps moving.
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    log(f"=== loading {town} ({weather_name}) ===")
    world = client.load_world(town)
    world.set_weather(weather_params)
    tm = client.get_trafficmanager(args.tm_port)

    original_settings = world.get_settings()
    npc_actors: list[carla.Actor] = []
    walker_actors: list[carla.Actor] = []
    controller_actors: list[carla.Actor] = []
    n_saved = 0

    cycles = max(1, args.cycles_per_cell)
    base_per_cycle = args.frames_per_cell // cycles
    extra = args.frames_per_cell % cycles

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

        npc_actors = spawn_npc_traffic(world, tm, args.npc_vehicles, rng)
        if args.npc_walkers > 0:
            walker_actors, controller_actors = spawn_npc_walkers(
                client, world, args.npc_walkers, rng
            )

        manifest_path = cell_dir / "manifest.jsonl"
        with manifest_path.open("w") as manifest:
            for cycle_idx in range(cycles):
                target_this_cycle = base_per_cycle + (1 if cycle_idx < extra else 0)
                if target_this_cycle == 0:
                    continue

                ego: carla.Actor | None = None
                cam: SyncCamera | None = None
                try:
                    ego = spawn_ego(world, tm, rng)
                    cam = SyncCamera(
                        world, ego, KITTI_IMAGE_WIDTH, KITTI_IMAGE_HEIGHT, KITTI_HFOV_DEG
                    )
                    intrinsics = CameraIntrinsics.from_attributes(
                        KITTI_IMAGE_WIDTH, KITTI_IMAGE_HEIGHT, KITTI_HFOV_DEG
                    )

                    for _ in range(args.warmup_ticks):
                        frame = world.tick()
                        cam.get_for_frame(frame)

                    captured = 0
                    tick_idx = 0
                    while captured < target_this_cycle:
                        frame = world.tick()
                        sample = cam.get_for_frame(frame)
                        tick_idx += 1
                        if (tick_idx - 1) % args.skip_ticks != 0:
                            continue

                        fname = f"frame_{n_saved:06d}.jpg"
                        fpath = cell_dir / fname
                        Image.fromarray(sample.image_rgb).save(
                            fpath, "JPEG", quality=args.jpeg_quality
                        )

                        # Project current actor bounding boxes to 2D and save
                        # KITTI-format labels alongside the frame.
                        detections = project_actors(
                            world=world,
                            camera=cam.sensor,
                            intrinsics=intrinsics,
                            ego_actor_id=ego.id,
                        )
                        label_path = cell_dir / f"frame_{n_saved:06d}.txt"
                        label_path.write_text(detections_to_kitti_label_text(detections))

                        manifest.write(
                            json.dumps(
                                {
                                    "frame": fname,
                                    "label": label_path.name,
                                    "town": town,
                                    "weather": weather_name,
                                    "cycle": cycle_idx,
                                    "sim_frame": sample.sim_frame,
                                    "sim_time": sample.sim_time,
                                    "image_width": KITTI_IMAGE_WIDTH,
                                    "image_height": KITTI_IMAGE_HEIGHT,
                                    "hfov_deg": KITTI_HFOV_DEG,
                                    "n_detections": len(detections),
                                }
                            )
                            + "\n"
                        )
                        n_saved += 1
                        captured += 1

                    if (cycle_idx + 1) % max(1, cycles // 10) == 0 or cycle_idx == cycles - 1:
                        log(
                            f"  {town}/{weather_name}: cycle {cycle_idx + 1}/{cycles}, "
                            f"saved {n_saved}/{args.frames_per_cell}"
                        )

                finally:
                    _cleanup_actors(client, cam, ego, [])  # NPCs persist across cycles

        log(f"--- {town}/{weather_name} done: {n_saved} frames -> {cell_dir} ---")
        return n_saved

    finally:
        try:
            tm.set_synchronous_mode(False)
        except RuntimeError:
            pass
        _cleanup_actors(
            client,
            None,
            None,
            npc_actors,
            walker_actors=walker_actors,
            controller_actors=controller_actors,
        )
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[extract_carla] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-config", default=str(DEFAULT_SERVER_CONFIG))
    parser.add_argument("--server-profile", default=None)
    parser.add_argument("--towns", nargs="+", default=["Town01", "Town03"])
    parser.add_argument("--weathers", nargs="+", default=["ClearNoon"])
    parser.add_argument("--frames-per-cell", type=int, default=100)
    parser.add_argument(
        "--cycles-per-cell",
        type=int,
        default=30,
        help="Number of independent ego respawns per (town, weather) cell. Higher = more spatial diversity.",
    )
    parser.add_argument("--out-dir", default=str(Path.home() / "datasets" / "carla_extracted"))
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.1)
    parser.add_argument(
        "--warmup-ticks",
        type=int,
        default=15,
        help="Ticks of autopilot warmup after each ego respawn (let traffic settle).",
    )
    parser.add_argument(
        "--skip-ticks",
        type=int,
        default=10,
        help="Save 1 in every N ticks within a cycle (skip=10 at 0.1s/tick = 1s gap = ~10m at 35km/h).",
    )
    parser.add_argument("--npc-vehicles", type=int, default=80, help="Mix of cars + ~15% two-wheelers (the Cyclist class).")
    parser.add_argument("--npc-walkers", type=int, default=60, help="Pedestrians with WalkerAIController (Pedestrian class).")
    parser.add_argument("--tm-port", type=int, default=8010)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = load_server_profile(Path(args.server_config), args.server_profile)
    if args.tm_port and args.tm_port != profile.tm_port:
        profile.tm_port = args.tm_port

    for w in args.weathers:
        if w not in WEATHER_PRESETS:
            raise ValueError(f"unknown weather `{w}`. known: {sorted(WEATHER_PRESETS)}")

    out_root = Path(args.out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    client, launched_proc = ensure_server(profile)
    log(f"server version: {client.get_server_version()}")

    total = 0
    t_start = time.time()
    try:
        for town in args.towns:
            for weather in args.weathers:
                cell_dir = out_root / f"{town}_{weather}"
                if (cell_dir / "manifest.jsonl").exists():
                    n_existing = sum(1 for _ in (cell_dir / "manifest.jsonl").open())
                    if n_existing >= args.frames_per_cell:
                        log(f"skipping {cell_dir} ({n_existing} frames already present)")
                        total += n_existing
                        continue
                n = extract_one_cell(
                    client=client,
                    town=town,
                    weather_name=weather,
                    weather_params=WEATHER_PRESETS[weather],
                    args=args,
                    cell_dir=cell_dir,
                    rng=rng,
                )
                total += n

        log(f"=== extraction complete: {total} frames in {time.time() - t_start:.0f}s ===")
        return 0
    finally:
        if launched_proc is not None and profile.terminate_on_exit:
            log("terminating launched CARLA server ...")
            launched_proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
