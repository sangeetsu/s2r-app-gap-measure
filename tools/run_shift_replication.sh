#!/usr/bin/env bash
# SHIFT replication (ledger §15.48): 15 condition cells + learning curve, ImageNet-backbone detectors, real evals, then features + FID.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.shiftrep.lock; flock -n 9 || exit 0
source tools/grid_imnet.conf; PY=${PY:-python}; ROOT=/mnt/features/shift_cells; R=results/calib_shift; mkdir -p $R
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[shift-rep $(date '+%F %T')] $*"; }
run(){ c=$1; n=$2; tag=$3; E=$(( 2400*4/n )); s1=$(( E*2/3 )); s2=$(( E*11/12 )); res=$R/src_${tag}_kittival.json; ck=checkpoints/anchis_$tag/last.pt
  [ -f $res ] || { log "TRAIN $tag ($n frames, $E epochs)"; rm -rf checkpoints/anchis_$tag; $J $PY tools/train_detector.py --init $INIT --source-data carla --source-root $ROOT --source-cells $c \
     --source-frames-slice 0:$n --epochs $E --lr-steps $s1,$s2 --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed 42 --out checkpoints/anchis_$tag --result $res > logs/shift_train_$tag.log 2>&1; }
  [ -f $R/src_${tag}_cityscapesval.json ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $R/src_${tag}_cityscapesval.json > /dev/null 2>&1
  o=$R/eval_${tag}__on__${c}.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $c --carla-root $ROOT --frames-slice 400:550 --result $o > /dev/null 2>&1; }
for d in $ROOT/SHIFT_*/; do c=$(basename $d); run $c 400 $c; done
for c in SHIFT_daytime_clear SHIFT_night_rainy SHIFT_dawn-dusk_foggy; do for n in 100 200; do run $c $n ${c}_n$n; done; done
log "training complete"
F=$ROOT/spatial_features_dinov3_15cells.npz; [ -f $F ] || $J $PY tools/extract_dinov3_features.py --kind carla --root $ROOT --img-h 352 --img-w 1216 --out $F > logs/shift_features.log 2>&1
systemd-run --user --scope -q --slice=research.slice -p MemoryMax=10G nice -n 10 $PY tools/fid_to_real.py --root $ROOT --features $F --out $R/fid_to_real.json 2>&1 | grep fid2real | tail -16
log "=== SHIFT replication complete ==="
