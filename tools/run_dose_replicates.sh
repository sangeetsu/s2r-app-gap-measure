#!/usr/bin/env bash
# Seed-17 replicates of all 40 dose-testbed detectors (ledger §15.50 caveat: single seed). Same recipe; checkpoints anchindr_*, results calib_dose_rep.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.doserep.lock; flock -n 9 || exit 0
source tools/grid_imnet.conf; PY=${PY:-python}; OUT=/mnt/features/carla_doses; R=results/calib_dose_rep; mkdir -p $R
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[doserep $(date '+%F %T')] $*"; }
TOWNS="${TOWNS:-Town01 Town02 Town05 Town10HD}"; DOSES="d00_ref d01_sun25 d02_sun10 d03_sun3 d04_night d05_fog25 d06_fog50 d07_fog75 d08_wet d09_rain"
train(){ local cell=$1 res ck; res=$R/src_${cell}_kittival.json; ck=checkpoints/anchindr_$cell/last.pt
  [ -f $res ] || { log "TRAIN $cell (seed 17)"; rm -rf checkpoints/anchindr_$cell; $J $PY tools/train_detector.py --init $INIT --source-data carla --source-root $OUT --source-cells $cell --source-frames-slice 0:800 \
     --epochs $EPOCHS --lr-steps $STEPS --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed 17 --out checkpoints/anchindr_$cell --result $res > logs/doserep_train_$cell.log 2>&1; }
  [ -f $R/src_${cell}_cityscapesval.json ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $R/src_${cell}_cityscapesval.json > /dev/null 2>&1; }
ev(){ local src=$1 tgt=$2 o; o=$R/eval_${src}__on__${tgt}.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/anchindr_$src/last.pt --cell $tgt --carla-root $OUT --frames-slice 800:1100 --result $o > /dev/null 2>&1; }
for town in $TOWNS; do ref=${town}_d00_ref; train $ref; for dose in $DOSES; do ev $ref ${town}_$dose; done; log "curve done: $town"; done
for town in $TOWNS; do for dose in $DOSES; do cell=${town}_$dose; [ $dose = d00_ref ] && continue; train $cell; ev $cell $cell; ev $cell ${town}_d00_ref; done; done
log "=== dose replicates complete ==="
