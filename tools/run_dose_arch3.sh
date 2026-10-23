#!/usr/bin/env bash
# Ledger §15.68 (E5): third detector family — Swin-T Faster R-CNN (candidate (b)) on Town01 + Town10HD x 10 doses, seed 42; gated on the smoke rule
# (own-cell mAP >= 70% of Faster R-CNN = 30.5). Collapse handling as §15.54: own-cell mAP < 8 -> retrain seeds 17 -> 100 -> 7.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.dosearch3.lock; flock -n 9 || exit 0
PY=${PY:-python}; OUT=/mnt/features/carla_doses; R=results/calib_dose_arch3; mkdir -p $R
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[arch3 $(date '+%F %T')] $*"; }
until [ -f results/smoke_arch3/eval_swin_own.json ]; do sleep 120; done
SM=$($PY -c "import json;print(json.load(open('results/smoke_arch3/eval_swin_own.json'))['metrics']['mAP@IoU=0.5']*100)")
awk -v o=$SM 'BEGIN{exit !(o>=30.5)}' || { log "Swin-T smoke own $SM < 30.5: candidate (b) does not train -> no third family (report)"; exit 0; }
log "Swin-T smoke own $SM >= 30.5 -> running the family"
REC="--arch frcnn_swin --optimizer adamw --lr 1e-4 --weight-decay 0.05 --epochs 12 --lr-steps 8,11 --warmup-iters 500"
TOWNS="Town01 Town10HD"; DOSES="d00_ref d01_sun25 d02_sun10 d03_sun3 d04_night d05_fog25 d06_fog50 d07_fog75 d08_wet d09_rain"
own(){ $PY -c "import json,sys;print(json.load(open(sys.argv[1]))['metrics']['mAP@IoU=0.5']*100)" $R/eval_${1}__on__${1}.json 2>/dev/null || echo 0; }
ev(){ local src=$1 tgt=$2 o; o=$R/eval_${src}__on__${tgt}.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/anchins_$src/last.pt --cell $tgt --carla-root $OUT --frames-slice 800:1100 --result $o > /dev/null 2>&1; }
train(){ local cell=$1 seed
  [ -f $R/src_${cell}_kittival.json ] && return
  for seed in 42 17 100 7; do rm -rf -- "/mnt/features/checkpoints/anchins_${cell:?}"; rm -f -- "$R/eval_${cell:?}__on__${cell:?}.json"; log "TRAIN $cell seed $seed"
    $J $PY tools/train_detector.py $REC --init imagenet --source-data carla --source-root $OUT --source-cells $cell --source-frames-slice 0:800 --batch-size 4 --seed $seed \
      --out checkpoints/anchins_$cell --result $R/src_${cell}_kittival.json > logs/arch3_train_${cell}_s$seed.log 2>&1
    ev $cell $cell; o=$(own $cell); log "  $cell seed $seed own $o"; awk -v o=$o 'BEGIN{exit !(o>=8)}' && break; rm -f -- "$R/src_${cell:?}_kittival.json"; done
  [ -f $R/src_${cell}_cityscapesval.json ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt checkpoints/anchins_$cell/last.pt --result $R/src_${cell}_cityscapesval.json > /dev/null 2>&1; }
for t in $TOWNS; do train ${t}_d00_ref; for d in $DOSES; do ev ${t}_d00_ref ${t}_$d; done; log "curve done: $t"; done
for t in $TOWNS; do for d in $DOSES; do [ $d = d00_ref ] && continue; train ${t}_$d; ev ${t}_$d ${t}_d00_ref; done; done
DOSE_DIR=$R $PY tools/analyze_dose.py results/calib_dose/dose_metrics.json > $R/analysis.txt 2>&1; log "=== arch3 complete; analysis written ==="
