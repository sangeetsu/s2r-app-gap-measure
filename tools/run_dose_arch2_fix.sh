#!/usr/bin/env bash
# Retrain collapsed FCOS detectors (own-cell mAP < 8) with alternative seeds, recompute their evals, rescore (ledger §15.53 run 2).
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.dosearch2fix.lock; flock -n 9 || exit 0
source tools/grid_arch2.conf; PY=${PY:-python}; OUT=/mnt/features/carla_doses; R=results/calib_dose_arch2
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[arch2fix $(date '+%F %T')] $*"; }
own(){ $PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('metrics',d)['mAP@IoU=0.5']*100)" $R/eval_${1}__on__${1}.json 2>/dev/null || echo 0; }
evalcell(){ local c=$1 t=${1%%_d*}; $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/anchina_$c/last.pt --cell $c --carla-root $OUT --frames-slice 800:1100 --result $R/eval_${c}__on__${c}.json > /dev/null 2>&1; }
for d in checkpoints/anchina_*/; do c=$(basename $d | sed 's/^anchina_//'); t=${c%%_d*}; o=$(own $c); awk -v o=$o 'BEGIN{exit !(o<8)}' || continue
  log "RETRAIN $c (own $o)"
  for seed in 17 100 7; do rm -rf checkpoints/anchina_$c $R/src_${c}_kittival.json $R/src_${c}_cityscapesval.json $R/eval_${c}__on__*.json
    $J $PY tools/train_detector.py --arch $ARCH --init imagenet --source-data carla --source-root $OUT --source-cells $c --source-frames-slice 0:800 \
       --epochs $EPOCHS --lr-steps $STEPS --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed $seed --out checkpoints/anchina_$c --result $R/src_${c}_kittival.json > logs/arch2_train_${c}_s$seed.log 2>&1
    evalcell $c; o=$(own $c); log "  $c seed $seed -> own $o"; awk -v o=$o 'BEGIN{exit !(o>=8)}' && break; done
  $J $PY tools/eval_detector_on_cityscapes.py --ckpt checkpoints/anchina_$c/last.pt --result $R/src_${c}_cityscapesval.json > /dev/null 2>&1
  rm -f $R/eval_${t}_d00_ref__on__${c}.json $R/eval_${c}__on__${t}_d00_ref.json   # evals involving this cell are recomputed below
done
DOSES="d00_ref d01_sun25 d02_sun10 d03_sun3 d04_night d05_fog25 d06_fog50 d07_fog75 d08_wet d09_rain"
ev(){ local src=$1 tgt=$2 o; o=$R/eval_${src}__on__${tgt}.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/anchina_$src/last.pt --cell $tgt --carla-root $OUT --frames-slice 800:1100 --result $o > /dev/null 2>&1; }
for town in Town01 Town10HD; do for dose in $DOSES; do ev ${town}_d00_ref ${town}_$dose; [ $dose = d00_ref ] || { ev ${town}_$dose ${town}_$dose; ev ${town}_$dose ${town}_d00_ref; }; done; done
DOSE_DIR=$R $PY tools/analyze_dose.py results/calib_dose/dose_metrics.json > $R/analysis.txt 2>&1; log "=== arch2 fix complete; analysis written ==="
