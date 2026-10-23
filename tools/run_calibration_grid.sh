#!/usr/bin/env bash
# APPEARANCE-GAP CALIBRATION GRID (2026-08-28)
#
# New target: a measurement framework that PREDICTS downstream drop from feature-space
# distance. This builds the ground truth half — the measured dmAP the distances must predict.
#
# CARLA is a clean 2x4 grid (Town01/Town03 x ClearNoon/OvercastNoon/SoftRainNoon/ClearNight,
# 1000 frames each), which separates the two axes a gap metric must not confuse:
#     APPEARANCE  weather varied at FIXED town  (same geometry/layout, different lighting)
#     CONTENT     town varied at FIXED weather  (same rendering, different scenes)
#
# Train one detector per cell, then evaluate each on every cell + KITTI. With 8 sources the
# full matrix is 8 trainings and 8x9 evaluations, giving 72 measured (source,target) drops
# to calibrate against — and held-out axes (unseen town, unseen weather) for validation.
#
# Launch: setsid nohup bash tools/run_calibration_grid.sh > logs/calib.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
exec 9>/tmp/.calib.lock; flock -n 9 || { echo "[calib] locked, exiting"; exit 0; }
PY=${PY:-python}
mkdir -p logs results/calib
log(){ echo "[calib $(date '+%F %T')] $*"; }

CELLS="Town01_ClearNoon Town01_OvercastNoon Town01_SoftRainNoon Town01_ClearNight \
       Town03_ClearNoon Town03_OvercastNoon Town03_SoftRainNoon Town03_ClearNight"

# --- train one detector per cell (1000 frames each; ~15 min per run) ---
for cell in $CELLS; do
  ck="checkpoints/calib_${cell}"
  res="results/calib/src_${cell}_kittival.json"
  if [ -f "$res" ]; then log "SKIP train ${cell}"; continue; fi
  log "TRAIN ${cell}"
  rm -rf "$ck"
  $PY tools/train_detector.py --source-data carla --source-cells "$cell" \
      --epochs 12 --batch-size 4 --lr 0.005 --seed 42 \
      --out "$ck" --result "$res" 2>&1 | grep -E "^\[train\] (train:|eval:)" | tail -2
  [ -f "$res" ] && log "  ${cell} -> KITTI: $($PY -c "
import json;print(round(json.load(open('$res'))['metrics']['mAP@IoU=0.5']*100,2))")"
done

log "=== all per-cell detectors trained; now the cross-cell evaluation matrix ==="

# --- evaluate every per-cell detector on every OTHER cell (sim->sim, both axes) ---
for src in $CELLS; do
  ck="checkpoints/calib_${src}/last.pt"
  [ -f "$ck" ] || { log "MISSING $ck"; continue; }
  for tgt in $CELLS; do
    [ "$src" = "$tgt" ] && continue
    out="results/calib/eval_${src}__on__${tgt}.json"
    [ -f "$out" ] && continue
    $PY tools/eval_detector_on_carla_cell.py --ckpt "$ck" --cell "$tgt" --result "$out" \
        2>&1 | grep -E "RESULT" || log "  FAIL ${src} on ${tgt}"
  done
  log "  matrix row done: ${src}"
done

log "=== calibration grid complete ==="
