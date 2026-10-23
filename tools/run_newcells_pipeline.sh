#!/usr/bin/env bash
# Post-generation pipeline for /mnt/features/carla_new cells (plan 2026-10-13, days 3-6).
# For every cell with >=1000 frames: DINOv3 features (once, whole root) -> per-cell detector on
# 0:800 -> anchored evals on every cell's 800:1000 (old grid + new) + KITTI + Cityscapes.
# Idempotent; flock; safe to relaunch after reboot.
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.newcells.lock; flock -n 9 || exit 0
PY=${PY:-python}; NEW=/mnt/features/carla_new; OLD=~/datasets/carla_extracted
log(){ echo "[newcells $(date '+%F %T')] $*"; }
cells_ready(){ for d in $NEW/*/; do c=$(basename $d); [ "$(ls $d/frame_*.jpg 2>/dev/null | wc -l)" -ge 1000 ] && echo $c; done; }
NEWCELLS=$(cells_ready); [ -z "$NEWCELLS" ] && { log "no complete new cells yet"; exit 0; }
log "new cells: $NEWCELLS"
# features for the new root (re-extract when cell count grows: encode count in filename)
NC=$(echo $NEWCELLS | wc -w); FEAT=$NEW/spatial_features_dinov3_${NC}cells.npz
[ -f $FEAT ] || $PY tools/extract_dinov3_features.py --kind carla --root $NEW --img-h 352 --img-w 1216 --out $FEAT 2>&1 | tail -1
OLDCELLS="Town01_ClearNoon Town01_OvercastNoon Town01_SoftRainNoon Town01_ClearNight Town03_ClearNoon Town03_OvercastNoon Town03_SoftRainNoon Town03_ClearNight"
mkdir -p results/calib_anchored
for c in $NEWCELLS; do
  res=results/calib_anchored/src_${c}_kittival.json; [ -f $res ] && continue
  log "TRAIN $c"; rm -rf checkpoints/anch_$c
  $PY tools/train_detector.py --source-data carla --source-root $NEW --source-cells $c --source-frames-slice 0:800 \
     --epochs 12 --batch-size 4 --lr 0.005 --seed 42 --out checkpoints/anch_$c --result $res 2>&1 | grep -E "^\[train\] eval"
done
ALL="$OLDCELLS $NEWCELLS"
for s in $ALL; do ck=checkpoints/anch_$s/last.pt; [ -f $ck ] || continue
  for t in $ALL; do out=results/calib_anchored/eval_${s}__on__${t}.json; [ -f $out ] && continue
    root=$OLD; echo "$NEWCELLS" | grep -qw $t && root=$NEW
    $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $t --carla-root $root --frames-slice 800:1000 --result $out 2>&1 | grep RESULT; done
  out=results/calib_anchored/src_${s}_cityscapesval.json; [ -f $out ] || $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $out >/dev/null 2>&1
  log "row done: $s"
done; log "=== newcells pipeline complete ($NC new cells) ==="
