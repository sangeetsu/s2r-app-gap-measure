#!/usr/bin/env bash
# ANCHORED calibration grid (plan 2026-10-13, day 1-2): per-cell detectors on frames 0:800,
# every target measured on its held-out 800:1000 slice (identical protocol for source anchor
# and cross-cell targets), plus KITTI val and Cityscapes val. Idempotent; flock; relaunchable.
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.anchored.lock; flock -n 9 || exit 0
PY=${PY:-python}; mkdir -p results/calib_anchored logs
log(){ echo "[anch $(date '+%F %T')] $*"; }
CELLS="Town01_ClearNoon Town01_OvercastNoon Town01_SoftRainNoon Town01_ClearNight Town03_ClearNoon Town03_OvercastNoon Town03_SoftRainNoon Town03_ClearNight"
for c in $CELLS; do
  res=results/calib_anchored/src_${c}_kittival.json; [ -f "$res" ] && { log "SKIP train $c"; continue; }
  log "TRAIN $c (0:800)"; rm -rf checkpoints/anch_$c
  $PY tools/train_detector.py --source-data carla --source-cells $c --source-frames-slice 0:800 \
     --epochs 12 --batch-size 4 --lr 0.005 --seed 42 --out checkpoints/anch_$c --result $res 2>&1 | grep -E "^\[train\] eval" | stdbuf -oL sed "s/^/  /"
done
log "=== evaluation matrix on held-out 800:1000 slices + Cityscapes ==="
for s in $CELLS; do ck=checkpoints/anch_$s/last.pt; [ -f $ck ] || continue
  for t in $CELLS; do out=results/calib_anchored/eval_${s}__on__${t}.json; [ -f $out ] && continue
    $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $t --frames-slice 800:1000 --result $out 2>&1 | grep RESULT; done
  out=results/calib_anchored/src_${s}_cityscapesval.json; [ -f $out ] || $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $out 2>&1 | grep -c metrics >/dev/null
  log "row done: $s"
done; log "=== anchored grid complete ==="
