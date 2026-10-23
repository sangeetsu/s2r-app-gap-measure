#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.gapchain.lock; flock -n 9 || exit 0
PY=${PY:-python}
for s in 42 17 100; do $PY tools/probe_transfer_matrix.py --seed $s --out results/calib/probe_transfer_matrix_dinov3_cs_s$s.json 2>&1 | grep -E "saved|Error|Traceback"; done
until [ -f results/calib/src_Town03_ClearNight_cityscapesval.json ] || ! pgrep -f "run_calib_[c]ityscapes" >/dev/null; do sleep 20; done
echo "[gapchain] inputs ready"
