#!/usr/bin/env bash
# 2026-10-14: which non-detection-pretrained init gives a usable CARLA detector? One cell (Town01_ClearNoon, frames 0:800),
# 3 runs in parallel inside research.slice; KITTI val eval is built into train_detector.py; then own-cell 800:1000 + Cityscapes.
set -u; cd "$(dirname "$0")/.."; PY=${PY:-python}; C=Town01_ClearNoon; ROOT=~/datasets/carla_extracted
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; mkdir -p results/smoke_init
run(){ tag=$1; init=$2; ep=$3; $J $PY tools/train_detector.py --init $init --source-data carla --source-root $ROOT --source-cells $C \
   --source-frames-slice 0:800 --epochs $ep --batch-size 4 --lr 0.005 --seed 42 --out checkpoints/smoke_$tag --result results/smoke_init/${tag}_kittival.json \
   > logs/smoke_$tag.log 2>&1; echo "[smoke $(date +%T)] trained $tag: $(grep -E '^\[train\] eval' logs/smoke_$tag.log | tail -1 | grep -oE "mAP@IoU=0.5': [0-9.]+")"; }
run in12 imagenet 12 & run in24 imagenet 24 & run sc24 scratch 24 & wait
for tag in in12 in24 sc24; do ck=checkpoints/smoke_$tag/last.pt; [ -f $ck ] || continue
  $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $C --carla-root $ROOT --frames-slice 800:1000 --result results/smoke_init/${tag}_own.json 2>&1 | grep RESULT
  $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell Town03_ClearNight --carla-root $ROOT --frames-slice 800:1000 --result results/smoke_init/${tag}_t03night.json 2>&1 | grep RESULT
  $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result results/smoke_init/${tag}_cityscapes.json > /dev/null 2>&1; echo "[smoke] $tag cityscapes done"; done
echo "=== smoke init complete ==="
