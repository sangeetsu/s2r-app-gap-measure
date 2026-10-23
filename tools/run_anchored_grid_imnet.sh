#!/usr/bin/env bash
# 24-cell anchored grid with NON-detection-pretrained detectors (ImageNet backbone; 2026-10-14, ledger §15.29-15.30).
# Same protocol as the COCO grid: per-cell detector on frames 0:800; every cell's 800:1000 + KITTI val (in training) + Cityscapes val.
# Two queues inside research.slice: trainer (GPU, serial) and evaluator (follows finished checkpoints). Idempotent; flock'd.
# glibc: send allocations >128 KB to mmap so freed image buffers return to the OS (dataset loops otherwise fragment the heap:
# 1.3 MB retained per decoded frame -> 23.7 GB OOM on 2026-10-14; ledger §15.31)
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.gridimnet.lock; flock -n 9 || exit 0
source tools/grid_imnet.conf    # INIT LR EPOCHS STEPS WARMUP (chosen from the smoke tests)
PY=${PY:-python}; OLD=~/datasets/carla_extracted; NEW=/mnt/features/carla_new
R=results/calib_imnet; CK=checkpoints/anchin; mkdir -p $R
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[imnet $(date '+%F %T')] $*"; }
CELLS="$(ls -d $OLD/Town*/ | xargs -n1 basename) $(ls -d $NEW/Town*/ | xargs -n1 basename)"
root_of(){ [ -d $OLD/$1 ] && echo $OLD || echo $NEW; }
evaluator(){ while true; do pending=0
  for s in $CELLS; do ck=${CK}_$s/last.pt; [ -f $R/src_${s}_kittival.json ] || { pending=1; continue; }
    for t in $CELLS; do out=$R/eval_${s}__on__${t}.json; [ -f $out ] && continue
      $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $t --carla-root $(root_of $t) --frames-slice 800:1000 --result $out > /dev/null 2>&1; done
    out=$R/src_${s}_cityscapesval.json; [ -f $out ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $out > /dev/null 2>&1
  done; [ $pending = 0 ] && break; sleep 60; done; log "evaluator done"; }
evaluator > logs/grid_imnet_eval.log 2>&1 &
for c in $CELLS; do res=$R/src_${c}_kittival.json; [ -f $res ] && continue
  log "TRAIN $c ($INIT lr=$LR ep=$EPOCHS steps=$STEPS warmup=$WARMUP)"; rm -rf ${CK}_$c
  $J $PY tools/train_detector.py --init $INIT --source-data carla --source-root $(root_of $c) --source-cells $c --source-frames-slice 0:800 \
     --epochs $EPOCHS --lr-steps $STEPS --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed 42 --out ${CK}_$c --result $res 2>&1 | grep -E "^\[train\] eval" | cut -c1-200
done
wait; log "=== imnet grid complete ==="
CALIB_DIR=$R $PY tools/analyze_decomposition.py
