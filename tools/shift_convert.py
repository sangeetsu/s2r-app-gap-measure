#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert SHIFT discrete/videos front-camera data (third-party HF mirror) into our CARLA cell layout (2026-10-15, plan replication).
Untrusted input: tar members extracted with the 'data' filter into their own directory; only .mp4/.json/.csv are read, nothing executed.
Per video: decode every 10th frame (1 fps, as SHIFT's own image subset), crop a KITTI-shaped band (rows 207:593 -> 1280x386, the 90-deg
SHIFT camera's horizon band; our CARLA rig is KITTI-matched 1242x375), clip boxes to the band, keep boxes with >=50% area inside and
height >= 10 px, map classes car/truck/bus->Car, pedestrian->Pedestrian, bicycle/motorcycle->Cyclist, write KITTI-format labels.
Output: OUT/videos/<video>/frame_<idx>.jpg + .txt, OUT/index.csv (video, town, weather/time coarse+fine, frames, boxes per class)."""
from __future__ import annotations
import argparse, csv, json, os, sys, tarfile
from multiprocessing import Pool
from pathlib import Path
import cv2
Y0, Y1 = 207, 593
CLS = {"car": "Car", "truck": "Car", "bus": "Car", "pedestrian": "Pedestrian", "bicycle": "Cyclist", "motorcycle": "Cyclist"}
def kitti_line(c, x1, y1, x2, y2): return f"{c} 0.00 0 0.00 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} -1 -1 -1 -1000 -1000 -1000 -10"
def convert(job):
    vid, mp4, frames, outdir, step = job
    out = Path(outdir) / vid; out.mkdir(parents=True, exist_ok=True); cap = cv2.VideoCapture(str(mp4)); i = 0; kept = 0; nb = {"Car": 0, "Pedestrian": 0, "Cyclist": 0}
    while True:
        ok, img = cap.read()
        if not ok: break
        if i % step == 0 and i in frames:
            lines = []
            for lab in frames[i]:
                c = CLS.get(lab["category"]); b = lab.get("box2d")
                if c is None or b is None: continue
                x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]; a0 = max(0, x2 - x1) * max(0, y2 - y1)
                cy1, cy2 = max(y1, Y0), min(y2, Y1)
                if a0 <= 0 or cy2 - cy1 < 10 or (cy2 - cy1) * (x2 - x1) < 0.5 * a0: continue
                lines.append(kitti_line(c, x1, cy1 - Y0, x2, cy2 - Y0)); nb[c] += 1
            cv2.imwrite(str(out / f"frame_{i:06d}.jpg"), img[Y0:Y1], [cv2.IMWRITE_JPEG_QUALITY, 95]); (out / f"frame_{i:06d}.txt").write_text("\n".join(lines) + ("\n" if lines else "")); kept += 1
        i += 1
    cap.release(); return vid, kept, nb, i
def main():
    p = argparse.ArgumentParser(); p.add_argument("--split", default="val"); p.add_argument("--src", default="/mnt/features/shift_hf/discrete/videos")
    p.add_argument("--out", default="/mnt/features/shift_conv"); p.add_argument("--step", type=int, default=10); p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(); S = Path(a.src) / a.split / "front"; OUT = Path(a.out) / a.split; mp4dir = OUT / "mp4"; mp4dir.mkdir(parents=True, exist_ok=True)
    if not any(mp4dir.glob("*.mp4")):
        with tarfile.open(S / "img.tar") as t: t.extractall(mp4dir, filter="data")
    print(f"[shift] mp4s: {len(list(mp4dir.glob('*.mp4')))}", flush=True)
    D = json.load(open(S / "det_2d.json"))["frames"]; per = {}; attrs = {}
    for f in D:
        per.setdefault(f["videoName"], {})[f["frameIndex"]] = f["labels"]; attrs.setdefault(f["videoName"], f["attributes"])
    del D
    jobs = [(v, mp4dir / f"{v}.mp4", {k: fr[k] for k in fr if k % a.step == 0}, str(OUT / "videos"), a.step) for v, fr in per.items() if (mp4dir / f"{v}.mp4").exists()]
    print(f"[shift] {len(jobs)} videos with labels", flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for k, (vid, kept, nb, nfr) in enumerate(pool.imap_unordered(convert, jobs)):
            at = attrs[vid]; rows.append({"video": vid, "town": at.get("town"), "weather_coarse": at.get("weather_coarse"), "weather_fine": at.get("weather_fine"),
                         "timeofday_coarse": at.get("timeofday_coarse"), "timeofday_fine": at.get("timeofday_fine"), "frames": kept, "video_frames": nfr, **{f"n_{c}": n for c, n in nb.items()}})
            if (k + 1) % 50 == 0: print(f"[shift] {k+1}/{len(jobs)} videos", flush=True)
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"[shift] done: {sum(r['frames'] for r in rows)} frames from {len(rows)} videos -> {OUT}", flush=True)
if __name__ == "__main__": raise SystemExit(main())
