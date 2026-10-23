#!/usr/bin/env bash
# Ledger §15.67 (E2): Virtual KITTI 2 frame-matched replication. Stage A (this script): features, clone detectors (seeds 42/17), evals on the
# weather variants, metrics, dose-trained detectors (seed 42) + asymmetry evals. E2b (sim->real) runs in tools/run_vkitti_real.sh.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.vkitti.lock; flock -n 9 || exit 0
PY=${PY:-python}; OUT=/mnt/features/vkitti2; R=results/calib_vkitti; mkdir -p $R ${R}_rep
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[vkitti $(date '+%F %T')] $*"; }
SCENES="Scene01 Scene02 Scene06 Scene18 Scene20"; VARS="d20_morning d21_sunset d22_overcast d23_fog d24_rain"
nfr(){ ls $OUT/$1/frame_*.jpg | wc -l; }
F=$OUT/spatial_features_dinov3.npz
[ -f $F ] || { log "features"; $J $PY tools/extract_dinov3_features.py --kind carla --root $OUT --img-h 352 --img-w 1216 --out $F > logs/vkitti_features.log 2>&1 || { log "FEATURES FAILED"; exit 1; }; }
train(){ local cell=$1 seed=$2 out=$3 ck n ntr E s1 s2; n=$(nfr $cell); ntr=$(( n * 3 / 4 )); E=$(python3 -c "print(round(9600/$ntr))"); s1=$(python3 -c "print(round($E*16/24))"); s2=$(python3 -c "print(round($E*22/24))")
  ck=checkpoints/vk_${cell}_s$seed
  [ -f $out/src_${cell}_kittival.json ] || { log "TRAIN $cell seed $seed ($ntr frames, $E epochs, steps $s1,$s2)"; rm -rf -- "/mnt/features/checkpoints/vk_${cell:?}_s${seed:?}"
    $J $PY tools/train_detector.py --init imagenet --source-data carla --source-root $OUT --source-cells $cell --source-frames-slice 0:$ntr --epochs $E --lr-steps $s1,$s2 \
      --warmup-iters 500 --batch-size 4 --lr 0.01 --seed $seed --out $ck --result $out/src_${cell}_kittival.json > logs/vkitti_train_${cell}_s$seed.log 2>&1; }
  [ -f $out/src_${cell}_cityscapesval.json ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck/last.pt --result $out/src_${cell}_cityscapesval.json > /dev/null 2>&1; }
ev(){ local src=$1 seed=$2 tgt=$3 out=$4 n ntr o; n=$(nfr $tgt); ntr=$(( n * 3 / 4 )); o=$out/eval_${src}__on__${tgt}.json
  [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/vk_${src}_s$seed/last.pt --cell $tgt --carla-root $OUT --frames-slice $ntr:$n --result $o > /dev/null 2>&1; }
for s in $SCENES; do ref=${s}_d00_ref
  train $ref 42 $R; train $ref 17 ${R}_rep
  for c in $ref $(for v in $VARS; do echo ${s}_$v; done); do ev $ref 42 $c $R; ev $ref 17 $c ${R}_rep; done; log "clone curve done: $s"; done
log "dose metrics"; $J $PY tools/dose_metrics.py --root $OUT --features $F --out $R/dose_metrics.json > logs/vkitti_metrics.log 2>&1 || log "METRICS FAILED"
log "low-level"; systemd-run --user --scope -q --slice=research.slice -p MemoryMax=12G nice -n 10 $PY tools/dose_lowlevel.py --root $OUT --out $R/dose_lowlevel.json > logs/vkitti_lowlevel.log 2>&1 || log "LOWLEVEL FAILED"
for s in $SCENES; do for v in $VARS; do c=${s}_$v; train $c 42 $R; ev $c 42 $c $R; ev $c 42 ${s}_d00_ref $R; done; log "dose-trained done: $s"; done
log "=== VKITTI stage A complete ==="
