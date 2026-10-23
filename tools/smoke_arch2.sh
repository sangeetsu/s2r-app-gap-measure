#!/usr/bin/env bash
# Second detector family smoke (plan 2026-10-16 R-C): FCOS at lr .0025/.005 and RetinaNet at .005, ImageNet backbone, Town01_d00_ref 0:800, 12 ep.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; PY=${PY:-python}; OUT=/mnt/features/carla_doses; R=results/smoke_arch2; mkdir -p $R
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; C=Town01_d00_ref
run(){ local tag=$1 arch=$2 lr=$3 ck=checkpoints/smoke_$1/last.pt
  [ -f $R/${tag}_kittival.json ] || $J $PY tools/train_detector.py --arch $arch --init imagenet --source-data carla --source-root $OUT --source-cells $C --source-frames-slice 0:800 \
     --epochs 12 --lr-steps 8,11 --warmup-iters 500 --batch-size 4 --lr $lr --seed 42 --out checkpoints/smoke_$tag --result $R/${tag}_kittival.json > logs/smoke_$tag.log 2>&1
  $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $C --carla-root $OUT --frames-slice 800:1100 --result $R/${tag}_own.json 2>&1 | grep RESULT
  $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell Town01_d07_fog75 --carla-root $OUT --frames-slice 800:1100 --result $R/${tag}_fog75.json 2>&1 | grep RESULT
  $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $R/${tag}_cityscapes.json > /dev/null 2>&1; echo "[smoke-arch2 $(date +%T)] done $tag"; }
run fcos_lr0025 fcos 0.0025; run fcos_lr005 fcos 0.005; run retina_lr005 retinanet 0.005; echo "=== smoke arch2 complete ==="
