#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Frame-matched appearance testbed (plan 2026-10-15): capture world STATE once, re-render the identical frames under any weather.

capture: same driving protocol as tools/extract_carla_frames.py (NPC traffic + walkers, ego on autopilot, 30 cycles, warmup, skip ticks)
         but saves, per saved frame, every vehicle/walker pose + blueprint + colour + light state and the ego pose -> states.jsonl.
render:  reload the town, spawn the same actors once (physics off), teleport all of them to the logged poses each frame, apply a
         weather dose, tick once, capture the camera (same rig/intrinsics as the extractor) and project labels (pure geometry).
Every condition (including the clear-noon reference) is rendered by the same replay path, so frames differ only in appearance.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import carla
from extract_carla_frames import (SyncCamera, spawn_npc_traffic, spawn_npc_walkers, spawn_ego, _cleanup_actors, CAMERA_TRANSFORM,
                                  KITTI_IMAGE_WIDTH, KITTI_IMAGE_HEIGHT, KITTI_HFOV_DEG, load_server_profile, ensure_server, DEFAULT_SERVER_CONFIG)
from carla_labels import project_actors, detections_to_kitti_label_text, CameraIntrinsics
def log(m): print(f"[replay] {m}", flush=True)

def T2l(t): return [t.location.x, t.location.y, t.location.z, t.rotation.pitch, t.rotation.yaw, t.rotation.roll]
def l2T(v): return carla.Transform(carla.Location(v[0], v[1], v[2]), carla.Rotation(pitch=v[3], yaw=v[4], roll=v[5]))

BASE = carla.WeatherParameters.ClearNoon
DOSES = {   # name -> overrides on ClearNoon (sun 45 deg, fog 2, cloud 5, wet 0)
    "d00_ref": {}, "d01_sun25": {"sun_altitude_angle": 25.0}, "d02_sun10": {"sun_altitude_angle": 10.0},
    "d03_sun3": {"sun_altitude_angle": 3.0}, "d04_night": {"sun_altitude_angle": -10.0},
    "d05_fog25": {"fog_density": 25.0}, "d06_fog50": {"fog_density": 50.0}, "d07_fog75": {"fog_density": 75.0},
    "d08_wet": {"cloudiness": 60.0, "wetness": 100.0, "precipitation_deposits": 50.0},
    "d09_rain": {"cloudiness": 90.0, "precipitation": 80.0, "precipitation_deposits": 80.0, "wetness": 80.0},
    # held-out doses (ledger §15.65, pre-registered 2026-10-17): never used to calibrate any metric
    "d10_fog35": {"fog_density": 35.0}, "d11_fog65": {"fog_density": 65.0}, "d12_fog90": {"fog_density": 90.0}, "d13_sun6": {"sun_altitude_angle": 6.0},
    "d14_night_rain": {"sun_altitude_angle": -10.0, "cloudiness": 90.0, "precipitation": 80.0, "precipitation_deposits": 80.0, "wetness": 80.0},
    "d15_night_fog50": {"sun_altitude_angle": -10.0, "fog_density": 50.0},
    "d16_sun10_wet": {"sun_altitude_angle": 10.0, "cloudiness": 60.0, "wetness": 100.0, "precipitation_deposits": 50.0},
    "d17_cloud90": {"cloudiness": 90.0}, "d18_dust50": {"dust_storm": 50.0},
    # held-out round 2 (ledger §15.76, pre-registered 2026-10-20): the prospectus's "mixed rain-and-fog" axis + four more unseen doses
    "d19_rain_fog25": {"cloudiness": 90.0, "precipitation": 80.0, "precipitation_deposits": 80.0, "wetness": 80.0, "fog_density": 25.0},
    "d20_rain_fog50": {"cloudiness": 90.0, "precipitation": 80.0, "precipitation_deposits": 80.0, "wetness": 80.0, "fog_density": 50.0},
    "d21_fog15": {"fog_density": 15.0}, "d22_fog45": {"fog_density": 45.0}, "d23_sun0": {"sun_altitude_angle": 0.0},
    "d24_night_wet": {"sun_altitude_angle": -10.0, "cloudiness": 60.0, "wetness": 100.0, "precipitation_deposits": 50.0},
    # held-out round 3 (ledger §15.87, pre-registered 2026-10-20): spans the cost range, for the calibration-formula test
    "d25_fog5": {"fog_density": 5.0}, "d26_fog30": {"fog_density": 30.0}, "d27_fog55": {"fog_density": 55.0}, "d28_fog80": {"fog_density": 80.0},
    "d29_sun15": {"sun_altitude_angle": 15.0}, "d30_sun_m5": {"sun_altitude_angle": -5.0}, "d31_night_fog25": {"sun_altitude_angle": -10.0, "fog_density": 25.0},
    "d32_dust90": {"dust_storm": 90.0},
    "d33_rain_fog75": {"cloudiness": 90.0, "precipitation": 80.0, "precipitation_deposits": 80.0, "wetness": 80.0, "fog_density": 75.0}}
def weather(name):
    w = carla.WeatherParameters(**{k: getattr(BASE, k) for k in ("cloudiness","precipitation","precipitation_deposits","wind_intensity","sun_azimuth_angle","sun_altitude_angle","fog_density","fog_distance","wetness","fog_falloff","scattering_intensity","mie_scattering_scale","rayleigh_scattering_scale")})
    for k, v in DOSES[name].items(): setattr(w, k, v)
    return w

def sync(world, tm, dt):
    s = world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = dt; world.apply_settings(s)
    if tm is not None: tm.set_synchronous_mode(True)

def capture(client, a):
    rng = random.Random(a.seed); out = Path(a.state_dir); out.mkdir(parents=True, exist_ok=True)
    world = client.load_world(a.town); world.set_weather(BASE); tm = client.get_trafficmanager(a.tm_port); tm.set_random_device_seed(a.seed)
    sync(world, tm, a.dt); npcs = spawn_npc_traffic(world, tm, a.npc_vehicles, rng); walkers, ctrls = spawn_npc_walkers(client, world, a.npc_walkers, rng)
    per = [a.frames // a.cycles + (1 if i < a.frames % a.cycles else 0) for i in range(a.cycles)]; n = 0
    with open(out / "states.jsonl", "w") as f:
        f.write(json.dumps({"header": {"town": a.town, "dt": a.dt, "skip": a.skip, "seed": a.seed, "frames": a.frames}}) + "\n")
        for ci, target in enumerate(per):
            ego = None
            try:
                ego = spawn_ego(world, tm, rng)
                for _ in range(a.warmup): world.tick()
                got = 0; k = 0
                while got < target:
                    world.tick(); k += 1
                    if (k - 1) % a.skip: continue
                    acts = []
                    for x in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.pedestrian.*")):
                        if x.id == ego.id: continue
                        try:   # an actor can be destroyed between the list and the query (killed earlier captures)
                            acts.append({"rid": x.id, "type": x.type_id, "color": x.attributes.get("color"), "T": T2l(x.get_transform()),
                                         "light": int(x.get_light_state()) if x.type_id.startswith("vehicle") else 0})
                        except RuntimeError:
                            continue
                    f.write(json.dumps({"i": n, "cycle": ci, "ego": {"type": ego.type_id, "color": ego.attributes.get("color"), "T": T2l(ego.get_transform())}, "actors": acts}) + "\n")
                    f.flush(); n += 1; got += 1
            except RuntimeError as e:
                log(f"  cycle {ci} aborted after {n} states: {e}")   # keep going with the next cycle
            finally:
                if ego is not None: _cleanup_actors(client, None, ego, [])
            if (ci + 1) % 5 == 0: log(f"  capture {a.town}: cycle {ci+1}/{a.cycles}, {n} states")
    log(f"captured {n} states -> {out/'states.jsonl'}")
    # CARLA 0.9.16 client hangs in sync-mode teardown here (observed twice: actors destroyed, then the main thread never returns).
    # The states file is the only product; render() calls load_world, which resets the episode. Exit without teardown.
    sys.stdout.flush(); import os; os._exit(0)

def spawn_like(world, typ, color, idx, role=None):
    bp = world.get_blueprint_library().find(typ)
    if color and bp.has_attribute("color"): bp.set_attribute("color", color)
    if role and bp.has_attribute("role_name"): bp.set_attribute("role_name", role)
    if bp.has_attribute("is_invincible"): bp.set_attribute("is_invincible", "true")
    a = world.spawn_actor(bp, carla.Transform(carla.Location(5000 + 15 * (idx % 60), 5000 + 15 * (idx // 60), 500)))
    a.set_simulate_physics(False); return a

class SemCamera(SyncCamera):
    """sensor.camera.semantic_segmentation on the same rig as the RGB camera; image_rgb holds the RED channel = semantic tag (2-D uint8)."""
    def __init__(self, world, parent, width, height, fov):
        import queue
        bp = world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        for k, v in (("image_size_x", width), ("image_size_y", height), ("fov", fov), ("sensor_tick", 0.0)): bp.set_attribute(k, str(v))
        self._q = queue.Queue(); self.sensor = world.spawn_actor(bp, CAMERA_TRANSFORM, attach_to=parent); self.sensor.listen(self._on_image)
    def _on_image(self, image):
        from extract_carla_frames import FrameSample
        raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self._q.put(FrameSample(sim_frame=image.frame, sim_time=image.timestamp, image_rgb=raw[:, :, 2].copy()))   # BGRA -> R = tag

class RawCamera(SyncCamera):
    """Any CARLA camera on the RGB rig, keeping the raw BGRA buffer (instance segmentation: R = semantic tag, G/B = instance id;
    depth: R + 256 G + 65536 B encodes distance). Added 2026-10-21 for ground-truth Cosmos controls (ledger §15.100)."""
    def __init__(self, world, parent, width, height, fov, typ):
        import queue
        bp = world.get_blueprint_library().find(typ)
        for k, v in (("image_size_x", width), ("image_size_y", height), ("fov", fov), ("sensor_tick", 0.0)): bp.set_attribute(k, str(v))
        self._q = queue.Queue(); self.sensor = world.spawn_actor(bp, CAMERA_TRANSFORM, attach_to=parent); self.sensor.listen(self._on_image)
    def _on_image(self, image):
        from extract_carla_frames import FrameSample
        raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self._q.put(FrameSample(sim_frame=image.frame, sim_time=image.timestamp, image_rgb=raw[:, :, 2::-1].copy()))   # BGRA -> RGB

def render(client, a):
    lines = [json.loads(l) for l in open(Path(a.state_dir) / "states.jsonl")]; hdr = lines[0]["header"]; S = lines[1:]
    if a.limit: S = S[:a.limit]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    world = client.load_world(hdr["town"]); world.set_weather(weather(a.dose)); sync(world, None, hdr["dt"])
    first = {}
    for s in S:
        for x in s["actors"]: first.setdefault(x["rid"], x)
    actors = {rid: spawn_like(world, x["type"], x["color"], i) for i, (rid, x) in enumerate(first.items())}
    ego = spawn_like(world, S[0]["ego"]["type"], S[0]["ego"]["color"], len(actors) + 1, role="extractor_ego")
    cam = SyncCamera(world, ego, KITTI_IMAGE_WIDTH, KITTI_IMAGE_HEIGHT, KITTI_HFOV_DEG); K = CameraIntrinsics.from_attributes(KITTI_IMAGE_WIDTH, KITTI_IMAGE_HEIGHT, KITTI_HFOV_DEG)
    sem = SemCamera(world, ego, KITTI_IMAGE_WIDTH, KITTI_IMAGE_HEIGHT, KITTI_HFOV_DEG) if a.semantic else None
    inst = RawCamera(world, ego, KITTI_IMAGE_WIDTH, KITTI_IMAGE_HEIGHT, KITTI_HFOV_DEG, "sensor.camera.instance_segmentation") if a.instance else None
    dep = RawCamera(world, ego, KITTI_IMAGE_WIDTH, KITTI_IMAGE_HEIGHT, KITTI_HFOV_DEG, "sensor.camera.depth") if a.depth else None
    night = DOSES[a.dose].get("sun_altitude_angle", 45.0) < 0
    lights = carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam) if night else None
    for _ in range(5):
        fr0 = world.tick(); cam.get_for_frame(fr0)
        if sem: sem.get_for_frame(fr0)
        if inst: inst.get_for_frame(fr0)
        if dep: dep.get_for_frame(fr0)
    park = {rid: carla.Transform(carla.Location(5000 + 15 * (i % 60), 5000 + 15 * (i // 60), 500)) for i, rid in enumerate(actors)}
    with open(out / "manifest.jsonl", "w") as man:
        for s in S:
            present = {x["rid"]: x for x in s["actors"]}
            cmds = [carla.command.ApplyTransform(ego.id, l2T(s["ego"]["T"]))]
            for rid, act in actors.items():
                cmds.append(carla.command.ApplyTransform(act.id, l2T(present[rid]["T"]) if rid in present else park[rid]))
            client.apply_batch_sync(cmds, False)
            for rid, x in present.items():
                if x["type"].startswith("vehicle"):
                    actors[rid].set_light_state(lights if night else carla.VehicleLightState(x["light"]))
            fr = world.tick(); smp = cam.get_for_frame(fr)
            name = f"frame_{s['i']:06d}"; Image.fromarray(smp.image_rgb).save(out / f"{name}.jpg", "JPEG", quality=95)
            if sem: Image.fromarray(sem.get_for_frame(fr).image_rgb).save(out / f"{name}_sem.png")   # CARLA semantic tags (0.9.14+ table), uint8
            if inst: Image.fromarray(inst.get_for_frame(fr).image_rgb).save(out / f"{name}_inst.png")   # R = tag, G/B = instance id (lossless)
            if dep:
                d = dep.get_for_frame(fr).image_rgb.astype(np.float64); m = (d[..., 0] + 256 * d[..., 1] + 65536 * d[..., 2]) / (256 ** 3 - 1) * 1000.0
                Image.fromarray(np.clip(m * 100, 0, 65535).astype(np.uint16)).save(out / f"{name}_depth.png")   # depth in cm, 16-bit
            dets = project_actors(world=world, camera=cam.sensor, intrinsics=K, ego_actor_id=ego.id)
            (out / f"{name}.txt").write_text(detections_to_kitti_label_text(dets))
            man.write(json.dumps({"frame": f"{name}.jpg", "label": f"{name}.txt", "town": hdr["town"], "weather": a.dose, "cycle": s["cycle"],
                                  "image_width": KITTI_IMAGE_WIDTH, "image_height": KITTI_IMAGE_HEIGHT, "hfov_deg": KITTI_HFOV_DEG, "n_detections": len(dets)}) + "\n")
            if (s["i"] + 1) % 200 == 0: log(f"  render {hdr['town']}/{a.dose}: {s['i']+1}/{len(S)}")
    cam.destroy()
    if sem: sem.destroy()
    if inst: inst.destroy()
    if dep: dep.destroy()
    client.apply_batch([carla.command.DestroyActor(x.id) for x in list(actors.values()) + [ego]])
    s = world.get_settings(); s.synchronous_mode = False; world.apply_settings(s); log(f"rendered {len(S)} frames -> {out}")

def main():
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=["capture", "render"]); p.add_argument("--town", default="Town01")
    p.add_argument("--state-dir", required=True); p.add_argument("--out-dir"); p.add_argument("--dose", default="d00_ref", choices=list(DOSES))
    p.add_argument("--frames", type=int, default=1050); p.add_argument("--cycles", type=int, default=30); p.add_argument("--skip", type=int, default=10)
    p.add_argument("--warmup", type=int, default=15); p.add_argument("--dt", type=float, default=0.1); p.add_argument("--npc-vehicles", type=int, default=80)
    p.add_argument("--npc-walkers", type=int, default=60); p.add_argument("--tm-port", type=int, default=8010); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None); p.add_argument("--server-profile", default="local_package_largemap")
    p.add_argument("--semantic", action="store_true", help="also save the semantic-segmentation tags (frame_XXXXXX_sem.png); ledger §15.77")
    p.add_argument("--instance", action="store_true", help="also save instance segmentation (frame_XXXXXX_inst.png: R tag, G/B instance id)")
    p.add_argument("--depth", action="store_true", help="also save depth (frame_XXXXXX_depth.png, uint16 cm)")
    a = p.parse_args(); client, _ = ensure_server(load_server_profile(Path(DEFAULT_SERVER_CONFIG), a.server_profile))
    (capture if a.mode == "capture" else render)(client, a)
if __name__ == "__main__": raise SystemExit(main())
