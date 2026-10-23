#!/usr/bin/env bash
# R-C: second detector family on the frame-matched testbed, 2 towns x 10 doses (+ asymmetry evals). Reads ARCH/LR from tools/grid_arch2.conf.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.dosearch2.lock; flock -n 9 || exit 0
source tools/grid_arch2.conf; PY=${PY:-python}; OUT=/mnt/features/carla_doses; R=results/calib_dose_arch2; mkdir -p $R
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[arch2 $(date '+%F %T')] $*"; }
TOWNS="${TOWNS:-Town01 Town10HD}"; DOSES="d00_ref d01_sun25 d02_sun10 d03_sun3 d04_night d05_fog25 d06_fog50 d07_fog75 d08_wet d09_rain"
train(){ local cell=$1 res ck; res=$R/src_${cell}_kittival.json; ck=checkpoints/anchina_$cell/last.pt
  [ -f $res ] || { log "TRAIN $cell ($ARCH lr $LR ep $EPOCHS wu $WARMUP)"; rm -rf checkpoints/anchina_$cell; $J $PY tools/train_detector.py --arch $ARCH --init imagenet --source-data carla --source-root $OUT --source-cells $cell --source-frames-slice 0:800 \
     --epochs $EPOCHS --lr-steps $STEPS --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed 42 --out checkpoints/anchina_$cell --result $res > logs/arch2_train_$cell.log 2>&1; }
  [ -f $R/src_${cell}_cityscapesval.json ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $R/src_${cell}_cityscapesval.json > /dev/null 2>&1; }
ev(){ local src=$1 tgt=$2 o; o=$R/eval_${src}__on__${tgt}.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/anchina_$src/last.pt --cell $tgt --carla-root $OUT --frames-slice 800:1100 --result $o > /dev/null 2>&1; }
for town in $TOWNS; do ref=${town}_d00_ref; train $ref; for dose in $DOSES; do ev $ref ${town}_$dose; done; log "curve done: $town"; done
for town in $TOWNS; do for dose in $DOSES; do cell=${town}_$dose; [ $dose = d00_ref ] && continue; train $cell; ev $cell $cell; ev $cell ${town}_d00_ref; done; done
log "=== arch2 testbed complete ==="; DOSE_DIR=$R $PY tools/analyze_dose.py results/calib_dose/dose_metrics.json > $R/analysis.txt 2>&1; log "analysis written"
