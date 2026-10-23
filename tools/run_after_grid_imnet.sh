#!/usr/bin/env bash
# After the ImageNet-backbone grid: (1) seed replicates on 8 spread-out cells -> noise floor; (2) label-free detector-box signals
# (3-probe ensemble); (3) BoS/AoL; (4) decomposition tables for both grids. Sequential, research.slice, allocator fix. flock'd.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.aftergridin.lock; flock -n 9 || exit 0
until grep -q "imnet grid complete" logs/grid_imnet.log 2>/dev/null; do sleep 300; done
source tools/grid_imnet.conf; PY=${PY:-python}; OLD=~/datasets/carla_extracted; NEW=/mnt/features/carla_new
FEAT=$NEW/spatial_features_dinov3_16cells.npz; R=results/calib_imnet; RR=results/calib_imnet_rep; mkdir -p $RR
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[aftergrid $(date '+%F %T')] $*"; }
root_of(){ [ -d $OLD/$1 ] && echo $OLD || echo $NEW; }
CELLS="$(ls -d $OLD/Town*/ | xargs -n1 basename) $(ls -d $NEW/Town*/ | xargs -n1 basename)"
for c in Town01_ClearNoon Town01_ClearNight Town02_SoftRainNoon Town03_ClearNight Town04_OvercastNoon Town05_SoftRainNoon Town10HD_OvercastNoon Town10HD_ClearNight; do
  res=$RR/src_${c}_kittival.json; ck=checkpoints/anchinrep_$c/last.pt
  [ -f $res ] || { log "REPLICATE seed 17: $c"; $J $PY tools/train_detector.py --init $INIT --source-data carla --source-root $(root_of $c) --source-cells $c \
     --source-frames-slice 0:800 --epochs $EPOCHS --lr-steps $STEPS --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed 17 \
     --out checkpoints/anchinrep_$c --result $res 2>&1 | grep -E "^\[train\] eval" | cut -c1-160; }
  for t in $CELLS; do out=$RR/eval_${c}__on__${t}.json; [ -f $out ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $t --carla-root $(root_of $t) --frames-slice 800:1000 --result $out > /dev/null 2>&1; done
  out=$RR/src_${c}_cityscapesval.json; [ -f $out ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $out > /dev/null 2>&1
done
log "noise floor"; $PY tools/analyze_noise_floor.py | tee $RR/noise_floor.txt
log "detector-box ensemble (imnet)"; $J $PY tools/detector_box_gap.py --probe-seeds 42,17,100 --extra-features $FEAT --ckpt-glob "checkpoints/anchin_*/last.pt" \
   --out $R/detector_box_gap_all_s42.json > logs/detbox_imnet.log 2>&1; tail -2 logs/detbox_imnet.log
NEWT=$(ls -d $NEW/Town*/ | xargs -n1 basename | tr '\n' ' ')
[ -f $R/bos_aol.json ] || { log "BoS/AoL (imnet)"; $J $PY tools/bos_aol_baselines.py --ckpt-glob "checkpoints/anchin_*/last.pt" --targets KITTI Cityscapes \
   $(ls -d $OLD/Town*/ | xargs -n1 basename | tr '\n' ' ') $NEWT --out $R/bos_aol.json > logs/bos_imnet.log 2>&1; tail -1 logs/bos_imnet.log; }
log "decomposition"; CALIB_DIR=$R $PY tools/analyze_decomposition.py > $R/decomposition.txt 2>&1; $PY tools/analyze_decomposition.py > results/calib_anchored/decomposition_24.txt 2>&1
cat $R/decomposition.txt; log "=== after-grid (imnet) complete ==="
