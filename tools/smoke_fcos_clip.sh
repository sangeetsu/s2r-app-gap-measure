#!/usr/bin/env bash
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; PY=${PY:-python}; OUT=/mnt/features/carla_doses; R=results/smoke_arch2; C=Town10HD_d00_ref
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"
while pgrep -u $USER -f "^bash tools/smoke_fcos_sta[b].sh" >/dev/null; do sleep 60; done
for cfg in "t10_lr005_e12_wu1000_clip1 0.005 12 1000 1.0" "t10_lr005_e24_wu1000_clip1 0.005 24 1000 1.0"; do set -- $cfg; tag=$1; lr=$2; ep=$3; wu=$4; cl=$5; s1=$(( ep*8/12 )); s2=$(( ep*11/12 ))
  [ -f $R/${tag}_kittival.json ] || $J $PY tools/train_detector.py --arch fcos --init imagenet --source-data carla --source-root $OUT --source-cells $C --source-frames-slice 0:800 \
     --epochs $ep --lr-steps $s1,$s2 --warmup-iters $wu --clip-grad $cl --batch-size 4 --lr $lr --seed 42 --out checkpoints/smoke_$tag --result $R/${tag}_kittival.json > logs/smoke_$tag.log 2>&1
  $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/smoke_$tag/last.pt --cell $C --carla-root $OUT --frames-slice 800:1100 --result $R/${tag}_own.json 2>&1 | grep RESULT; echo "[clip $(date +%T)] done $tag"; done
echo "=== fcos clip smoke complete ==="
