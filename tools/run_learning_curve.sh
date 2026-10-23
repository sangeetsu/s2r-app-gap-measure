#!/usr/bin/env bash
# Within-cell learning curve (2026-10-14, ledger §15.38): content and appearance fixed, only the amount of labeled data varies.
# 6 ClearNoon cells (one per grid town) x frame prefixes 100/200/400 (800 = main ImageNet grid), fixed 2400 iterations, same schedule shape.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.lcurve.lock; flock -n 9 || exit 0
source tools/grid_imnet.conf; PY=${PY:-python}; OLD=~/datasets/carla_extracted; NEW=/mnt/features/carla_new; R=results/calib_imnet_lc; mkdir -p $R
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[lc $(date '+%F %T')] $*"; }
root_of(){ [ -d $OLD/$1 ] && echo $OLD || echo $NEW; }
for k in 100 200 400; do for c in Town01_ClearNoon Town02_ClearNoon Town03_ClearNoon Town04_ClearNoon Town05_ClearNoon Town10HD_ClearNoon; do
  tag=${c}_n$k; res=$R/src_${tag}_kittival.json; ck=checkpoints/anchinlc_$tag/last.pt
  E=$(( 2400 * 4 / k )); s1=$(( E * 8 / 12 )); s2=$(( E * 11 / 12 ))
  [ -f $res ] || { log "TRAIN $tag (epochs $E, steps $s1,$s2)"; rm -rf checkpoints/anchinlc_$tag
    $J $PY tools/train_detector.py --init $INIT --source-data carla --source-root $(root_of $c) --source-cells $c --source-frames-slice 0:$k \
      --epochs $E --lr-steps $s1,$s2 --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed 42 --out checkpoints/anchinlc_$tag --result $res 2>&1 | grep -E "^\[train\] eval" | cut -c1-110; }
  out=$R/eval_${tag}__on__${c}.json; [ -f $out ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $c --carla-root $(root_of $c) --frames-slice 800:1000 --result $out 2>&1 | grep RESULT
  out=$R/src_${tag}_cityscapesval.json; [ -f $out ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $out > /dev/null 2>&1
done; done; log "=== learning curve complete ==="
