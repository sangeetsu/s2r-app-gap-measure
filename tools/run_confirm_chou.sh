#!/usr/bin/env bash
# Pre-registered confirmation of the geometry model (ledger §15.43): 12 fresh cells -> ImageNet-backbone detectors -> real evals -> markers -> score.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.confirmchou.lock; flock -n 9 || exit 0
source tools/grid_imnet.conf; PY=${PY:-python}; OUT=/mnt/features/carla_confirm; R=results/calib_imnet_confirm; mkdir -p $OUT $R
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[confirm $(date '+%F %T')] $*"; }
nf(){ ls $OUT/$1/frame_*.jpg 2>/dev/null | wc -l; }
CELLS="Town01_ClearSunset Town02_ClearSunset Town03_ClearSunset Town04_ClearSunset Town05_ClearSunset Town10HD_ClearSunset Town01_WetNoon Town02_WetNoon Town03_WetNoon Town04_WetNoon Town05_WetNoon Town10HD_WetNoon"
start_carla(){ pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null; sleep 5
  (cd ~/CARLA/CARLA_0.9.16 && setsid nohup systemd-run --user --scope -q --slice=research.slice ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 > /dev/null 2>&1 < /dev/null &)
  sleep 60; log "CARLA (Epic) started"; }
trainer(){ while true; do left=0
  for c in $CELLS; do res=$R/src_${c}_kittival.json; ck=checkpoints/anchinc_$c/last.pt
    [ -f $R/src_${c}_cityscapesval.json ] && continue
    [ -f $OUT/.failed_$c ] && continue                      # generation failed twice: do not wait for it
    [ "$(nf $c)" -ge 1000 ] && [ ! -f $OUT/$c/.generating ] || { left=1; continue; }
    [ -f $res ] || { log "TRAIN $c"; rm -rf checkpoints/anchinc_$c; $J $PY tools/train_detector.py --init $INIT --source-data carla --source-root $OUT --source-cells $c \
       --source-frames-slice 0:800 --epochs $EPOCHS --lr-steps $STEPS --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed 42 --out checkpoints/anchinc_$c --result $res > /dev/null 2>&1; }
    o=$R/eval_${c}__on__${c}.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $c --carla-root $OUT --frames-slice 800:1000 --result $o > /dev/null 2>&1
    $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $R/src_${c}_cityscapesval.json > /dev/null 2>&1; log "  trained+evaluated $c"
  done; [ $left = 0 ] && grep -q "generation done" logs/confirm_chou.log && break; sleep 120; done; log "trainer done"; }
trainer &
start_carla
for c in $CELLS; do town=${c%%_*}; w=${c#*_}
  for attempt in 1 2; do [ "$(nf $c)" -ge 1000 ] && break; mkdir -p $OUT/$c; touch $OUT/$c/.generating; t0=$(date +%s)
    timeout 7200 $J $PY tools/extract_carla_frames.py --server-profile local_package_largemap --towns $town --weathers $w --frames-per-cell 1000 --seed 42 --out-dir $OUT > logs/confirm_gen_$c.log 2>&1 < /dev/null
    rm -f $OUT/$c/.generating; log "  $c: $(nf $c) frames in $(( ($(date +%s)-t0)/60 )) min"
    [ "$(nf $c)" -ge 1000 ] || { rm -rf $OUT/$c; start_carla; }; done
  [ "$(nf $c)" -ge 1000 ] || { touch $OUT/.failed_$c; log "  SKIP $c (failed twice)"; }; done
pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null; log "generation done"; wait
$J $PY tools/manifold_markers.py --globs "checkpoints/anchinc_*/last.pt" --out results/manifold_markers/markers_confirm.json 2>&1 | grep -E "^\[mm\]" | tail -14
$PY tools/score_confirm_chou.py | tee results/manifold_markers/confirm_score.txt; log "=== confirmation complete ==="
