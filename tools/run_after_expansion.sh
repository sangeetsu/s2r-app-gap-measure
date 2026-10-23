#!/usr/bin/env bash
# When CARLA expansion completes: features -> per-cell detectors -> anchored evals (old+new) ->
# anchored probe matrix over ALL cells -> analysis. Idempotent end to end.
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.afterexp.lock; flock -n 9 || exit 0
until grep -q "expansion retry complete" logs/carla_expansion.log 2>/dev/null; do sleep 120; done
until grep -q "anchored grid complete" logs/anchored.log 2>/dev/null; do sleep 120; done   # GPU: let the 8-cell grid finish first
bash tools/run_newcells_pipeline.sh
PY=${PY:-python}; NEW=/mnt/features/carla_new
NC=$(for d in $NEW/*/; do [ "$(ls $d/frame_*.jpg 2>/dev/null | wc -l)" -ge 1000 ] && echo x; done | wc -l)
$PY tools/probe_transfer_matrix.py --anchored --seed 42 --extra-root $NEW --extra-features $NEW/spatial_features_dinov3_${NC}cells.npz \
   --out results/calib_anchored/probe_matrix_anchored_all_s42.json 2>&1 | grep -E "saved|Traceback"
$PY tools/analyze_anchored.py results/calib_anchored/probe_matrix_anchored_all_s42.json
echo "[afterexp] done ($NC new cells)"
# published references on the final grid (BoS ICLR'24, AoL NeurIPS'22), 100 imgs/target
NEWT=$(for d in $NEW/*/; do [ "$(ls $d/frame_*.jpg 2>/dev/null | wc -l)" -ge 1000 ] && basename $d; done | tr '\n' ' ')
$PY tools/bos_aol_baselines.py --targets KITTI Cityscapes Town01_ClearNoon Town01_OvercastNoon Town01_SoftRainNoon Town01_ClearNight Town03_ClearNoon Town03_OvercastNoon Town03_SoftRainNoon Town03_ClearNight $NEWT 2>&1 | grep -E "saved|Traceback|detectors"
echo "[afterexp] baselines done"
