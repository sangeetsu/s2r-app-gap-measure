#!/usr/bin/env bash
# Ledger §15.68 (E5) smoke: Deformable DETR and Swin-T Faster R-CNN on Town10HD_d00_ref (frames 0:800, seed 42); own-cell 800:1100 + KITTI val.
export MALLOC_MMAP_THRESHOLD_=131072 HF_HOME=/mnt/features/hf_cache
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.smoke3.lock; flock -n 9 || exit 0
PY=${PY:-python}; OUT=/mnt/features/carla_doses; R=results/smoke_arch3; mkdir -p $R; C=Town10HD_d00_ref
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[smoke3 $(date '+%F %T')] $*"; }
run(){ local tag=$1; shift; [ -f $R/src_${tag}_kittival.json ] || { log "TRAIN $tag"; $J $PY tools/train_detector.py --init imagenet --source-data carla --source-root $OUT --source-cells $C \
    --source-frames-slice 0:800 --batch-size 4 --seed 42 --out checkpoints/smoke3_$tag --result $R/src_${tag}_kittival.json "$@" > logs/smoke3_$tag.log 2>&1; }
  [ -f $R/eval_${tag}_own.json ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/smoke3_$tag/last.pt --cell $C --carla-root $OUT --frames-slice 800:1100 --result $R/eval_${tag}_own.json > /dev/null 2>&1
  log "$tag own $(python3 -c "import json;print(round(json.load(open('$R/eval_${tag}_own.json'))['metrics']['mAP@IoU=0.5']*100,1))" 2>/dev/null) KITTI $(python3 -c "import json;print(round(json.load(open('$R/src_${tag}_kittival.json'))['metrics']['mAP@IoU=0.5']*100,1))" 2>/dev/null)"; }
run ddetr --arch ddetr --optimizer adamw --lr 2e-4 --backbone-lr-mult 0.1 --weight-decay 1e-4 --clip-grad 0.1 --epochs 50 --lr-steps 40 --warmup-iters 500
run swin --arch frcnn_swin --optimizer adamw --lr 1e-4 --weight-decay 0.05 --epochs 12 --lr-steps 8,11 --warmup-iters 500
log "=== smoke3 done ==="
