#!/usr/bin/env bash
# Ledger §15.67 E2b: KITTI tracking real frames for the five VKITTI2 sequences; distances -> PREDICTIONS -> clone-on-real evals; real-trained detectors.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.vkreal.lock; flock -n 9 || exit 0
PY=${PY:-python}; RO=/mnt/features/vkitti2_real; SO=/mnt/features/vkitti2; R=results/calib_vkitti; Z=/mnt/features/downloads/kitti_tracking/data_tracking_image_2.zip
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[vkreal $(date '+%F %T')] $*"; }
SCENES="Scene01 Scene02 Scene06 Scene18 Scene20"; VARS="d20_morning d21_sunset d22_overcast d23_fog d24_rain"
until [ "$(stat -c %s $Z 2>/dev/null)" = "15813146295" ]; do sleep 60; done; log "tracking zip complete"
[ -d $RO/Scene20_d00_ref ] || $PY -I tools/vkitti_convert.py --mode tracking --out $RO > logs/vkreal_convert.log 2>&1 || { log "CONVERT FAILED"; exit 1; }
for s in $SCENES; do log "  $s real frames $(ls $RO/${s}_d00_ref/*.jpg | wc -l) vs clone $(ls $SO/${s}_d00_ref/*.jpg | wc -l)"; done
F=$RO/spatial_features_dinov3.npz; [ -f $F ] || $J $PY tools/extract_dinov3_features.py --kind carla --root $RO --img-h 352 --img-w 1216 --out $F > logs/vkreal_features.log 2>&1 || { log "FEATURES FAILED"; exit 1; }
need(){ for s in $SCENES; do for v in d00_ref $VARS; do [ -f $R/eval_${s}_d00_ref__on__${s}_$v.json ] && [ -f ${R}_rep/eval_${s}_d00_ref__on__${s}_$v.json ] || return 1; done; done; [ -f $R/dose_metrics.json ] && python3 -c "import json,sys;d=json.load(open('$R/dose_metrics.json'));sys.exit(0 if len(d)>=30 else 1)"; }
until need; do sleep 120; done; log "stage-A clone evals + weather metrics present"
[ -f $R/sim2real_distances.json ] || $J $PY tools/vkitti_sim2real.py --mode distances > logs/vkreal_distances.log 2>&1 || { log "DISTANCES FAILED"; exit 1; }
[ -f $R/predictions_sim2real.json ] || { $PY tools/vkitti_sim2real.py --mode predict 2>&1 | tee logs/vkreal_predict.log; [ -f $R/predictions_sim2real.json ] || { log "PREDICT FAILED"; exit 1; }; }
nfr(){ ls $1/$2/frame_*.jpg | wc -l; }
for s in $SCENES; do n=$(nfr $RO ${s}_d00_ref); ntr=$(( n * 3 / 4 ))
  for seed in 42 17; do o=$R/eval_${s}_d00_ref__on__${s}_real.json; [ $seed = 17 ] && o=${R}_rep/eval_${s}_d00_ref__on__${s}_real.json
    [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/vk_${s}_d00_ref_s$seed/last.pt --cell ${s}_d00_ref --carla-root $RO --frames-slice $ntr:$n --result $o > /dev/null 2>&1; done; done
log "clone-on-real evals done"
for s in $SCENES; do n=$(nfr $RO ${s}_d00_ref); ntr=$(( n * 3 / 4 )); E=$(python3 -c "print(round(9600/$ntr))"); s1=$(python3 -c "print(round($E*16/24))"); s2=$(python3 -c "print(round($E*22/24))"); ck=checkpoints/vkreal_${s}_s42
  [ -f $R/src_${s}_real_kittival.json ] || $J $PY tools/train_detector.py --init imagenet --source-data carla --source-root $RO --source-cells ${s}_d00_ref --source-frames-slice 0:$ntr --epochs $E --lr-steps $s1,$s2 \
     --warmup-iters 500 --batch-size 4 --lr 0.01 --seed 42 --out $ck --result $R/src_${s}_real_kittival.json > logs/vkreal_train_$s.log 2>&1
  [ -f $R/eval_${s}_real__on__${s}_real.json ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck/last.pt --cell ${s}_d00_ref --carla-root $RO --frames-slice $ntr:$n --result $R/eval_${s}_real__on__${s}_real.json > /dev/null 2>&1
  [ -f $R/eval_${s}_real__on__${s}_d00_ref.json ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck/last.pt --cell ${s}_d00_ref --carla-root $SO --frames-slice $ntr:$n --result $R/eval_${s}_real__on__${s}_d00_ref.json > /dev/null 2>&1
  log "real-trained done: $s"; done
log "=== E2b complete ==="
