#!/usr/bin/env bash
# 2026-10-14 smoke 2: ImageNet-init with a real schedule (legacy StepLR(3) froze LR after epoch 6). Same cell/protocol as smoke_init.sh.
set -u; cd "$(dirname "$0")/.."; PY=${PY:-python}; C=Town01_ClearNoon; ROOT=~/datasets/carla_extracted
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"
run(){ tag=$1; lr=$2; ep=$3; steps=$4; $J $PY tools/train_detector.py --init imagenet --source-data carla --source-root $ROOT --source-cells $C \
   --source-frames-slice 0:800 --epochs $ep --lr-steps $steps --warmup-iters 500 --batch-size 4 --lr $lr --seed 42 \
   --out checkpoints/smoke_$tag --result results/smoke_init/${tag}_kittival.json > logs/smoke_$tag.log 2>&1; echo "[smoke2 $(date +%T)] trained $tag"; }
run ms005e24 0.005 24 16,22 & run ms01e24 0.01 24 16,22 & run ms01e12 0.01 12 8,11 & wait
for tag in ms005e24 ms01e24 ms01e12; do ck=checkpoints/smoke_$tag/last.pt; [ -f $ck ] || continue
  $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $C --carla-root $ROOT --frames-slice 800:1000 --result results/smoke_init/${tag}_own.json 2>&1 | grep RESULT
  $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell Town03_ClearNight --carla-root $ROOT --frames-slice 800:1000 --result results/smoke_init/${tag}_t03night.json 2>&1 | grep RESULT
  $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result results/smoke_init/${tag}_cityscapes.json > /dev/null 2>&1; done
echo "=== smoke sched complete ==="
