#!/usr/bin/env bash
# Held-out towns for the pre-registered FID test (2026-10-14, ledger §15.34). Order: generate -> features -> PREDICT (timestamped,
# before any detector exists) -> train ImageNet-backbone detectors -> evaluate -> score. research.slice; allocator fix; flock'd.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.towns2.lock; flock -n 9 || exit 0
source tools/grid_imnet.conf; PY=${PY:-python}; OUT=/mnt/features/carla_new2; R=results/calib_imnet_towns2; mkdir -p $OUT $R
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[towns2 $(date '+%F %T')] $*"; }
nf(){ ls $OUT/$1/frame_*.jpg 2>/dev/null | wc -l; }
# CARLA server: started ONCE, detached, stdout/stderr to /dev/null, inside research.slice. If the extractor starts its own server,
# the server inherits the extractor's stdout pipe and the pipeline never ends (hung 08:28-12:50 on 2026-10-14).
start_carla(){ pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null; sleep 5
  (cd ~/CARLA/CARLA_0.9.16 && setsid nohup systemd-run --user --scope -q --slice=research.slice ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 > /dev/null 2>&1 < /dev/null &)
  sleep 45; log "CARLA server (re)started: $(pgrep -u $USER -c -f 'CarlaUE4-Linux-Shippin[g]') proc"; }
start_carla
for town in Town06 Town07 Town12 Town13 Town15 Town11; do for w in ClearNoon ClearNight; do c=${town}_$w
  for attempt in 1 2; do [ "$(nf $c)" -ge 1000 ] && break
    log "GEN $c (attempt $attempt)"; t0=$(date +%s)
    timeout 5400 $J $PY tools/extract_carla_frames.py --server-profile local_package --towns $town --weathers $w --frames-per-cell 1000 --out-dir $OUT \
       > logs/towns2_gen_$c.log 2>&1 < /dev/null
    grep -E "saved 1000|Error|Traceback|not found|failed" logs/towns2_gen_$c.log | tail -2
    log "  $c: $(nf $c) frames in $(( ($(date +%s)-t0)/60 )) min"
    [ "$(nf $c)" -ge 1000 ] || start_carla; done
  [ "$(nf $c)" -ge 1000 ] || { log "  SKIP $c (generation failed twice)"; rm -rf $OUT/$c; }
done; done
pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null
CELLS=$(for d in $OUT/Town*/; do [ "$(nf $(basename $d))" -ge 1000 ] && basename $d; done | tr '\n' ' '); NC=$(echo $CELLS | wc -w); FEAT=$OUT/spatial_features_dinov3_${NC}cells.npz
log "complete cells ($NC): $CELLS"
[ -f $FEAT ] || $J $PY tools/extract_dinov3_features.py --kind carla --root $OUT --img-h 352 --img-w 1216 --out $FEAT 2>&1 | tail -1
PRED=results/calib_imnet/prereg_predictions_towns2.json
[ -f $PRED ] || { $J $PY tools/fid_to_real.py --root $OUT --features $FEAT --cells $CELLS --predict --out $PRED 2>&1 | grep fid2real; log "PREDICTIONS WRITTEN $(sha256sum $PRED | cut -c1-16)"; }
for c in $CELLS; do res=$R/src_${c}_kittival.json; ck=checkpoints/anchin_$c/last.pt
  [ -f $res ] || { log "TRAIN $c"; rm -rf checkpoints/anchin_$c; $J $PY tools/train_detector.py --init $INIT --source-data carla --source-root $OUT --source-cells $c \
     --source-frames-slice 0:800 --epochs $EPOCHS --lr-steps $STEPS --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed 42 --out checkpoints/anchin_$c --result $res 2>&1 | grep -E "^\[train\] eval" | cut -c1-120; }
  out=$R/eval_${c}__on__${c}.json; [ -f $out ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $c --carla-root $OUT --frames-slice 800:1000 --result $out 2>&1 | grep RESULT
  out=$R/src_${c}_cityscapesval.json; [ -f $out ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $out > /dev/null 2>&1
done
log "score"; $PY tools/analyze_prereg.py | tee $R/prereg_score.txt; log "=== towns2 complete ==="
