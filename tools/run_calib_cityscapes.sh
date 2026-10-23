#!/usr/bin/env bash
# Second real target for the sim->real axis: every per-cell CARLA detector on Cityscapes val (ALDI boxes).
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.calibcs.lock; flock -n 9 || exit 0
PY=${PY:-python}
for cell in Town01_ClearNoon Town01_OvercastNoon Town01_SoftRainNoon Town01_ClearNight Town03_ClearNoon Town03_OvercastNoon Town03_SoftRainNoon Town03_ClearNight; do
  out="results/calib/src_${cell}_cityscapesval.json"; [ -f "$out" ] && continue
  $PY tools/eval_detector_on_cityscapes.py --ckpt checkpoints/calib_${cell}/last.pt --result "$out" 2>&1 | grep -E "metrics|FAIL" | sed "s/^/[$cell] /"
done; echo "[calibcs] done"
