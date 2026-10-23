#!/usr/bin/env bash
# After run_after_expansion.sh finishes (all-cell probe matrix + BoS/AoL): label-free detector-box gap on the
# full grid (tools/detector_box_gap.py, idempotent per source) + analysis. flock-guarded; safe to relaunch.
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.afterexp2.lock; flock -n 9 || exit 0
until grep -q "\[afterexp\] done" logs/after_expansion.log 2>/dev/null; do sleep 180; done   # start with BoS/AoL, not after it
PY=${PY:-python}; NEW=/mnt/features/carla_new
NC=$(for d in $NEW/*/; do [ "$(ls $d/frame_*.jpg 2>/dev/null | wc -l)" -ge 1000 ] && echo x; done | wc -l)
$PY tools/detector_box_gap.py --probe-seeds 42,17,100 --extra-features $NEW/spatial_features_dinov3_${NC}cells.npz --out results/calib_anchored/detector_box_gap_all_s42.json 2>&1 | grep -E "^\[dbg\]|Traceback|Error"
$PY tools/analyze_boxgap.py results/calib_anchored/detector_box_gap_all_s42.json 0.5
echo "[afterexp2] done ($NC new cells)"
