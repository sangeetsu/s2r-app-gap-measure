#!/usr/bin/env bash
# FCOS stability smoke on the failing town (Town10HD_d00_ref): seed luck vs schedule. 3 configs, sequential.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; PY=${PY:-python}; OUT=/mnt/features/carla_doses; R=results/smoke_arch2; C=Town10HD_d00_ref
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"
run(){ local tag=$1 lr=$2 ep=$3 wu=$4 seed=$5 s1=$(( $3*8/12 )) s2=$(( $3*11/12 ))
  [ -f $R/${tag}_kittival.json ] || $J $PY tools/train_detector.py --arch fcos --init imagenet --source-data carla --source-root $OUT --source-cells $C --source-frames-slice 0:800 \
     --epochs $ep --lr-steps $s1,$s2 --warmup-iters $wu --batch-size 4 --lr $lr --seed $seed --out checkpoints/smoke_$tag --result $R/${tag}_kittival.json > logs/smoke_$tag.log 2>&1
  $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/smoke_$tag/last.pt --cell $C --carla-root $OUT --frames-slice 800:1100 --result $R/${tag}_own.json 2>&1 | grep RESULT; echo "[stab $(date +%T)] done $tag"; }
run t10_lr005_e12_s17 0.005 12 500 17; run t10_lr005_e24_wu1000 0.005 24 1000 42; run t10_lr003_e24_wu1000 0.003 24 1000 42; echo "=== fcos stab complete ==="
