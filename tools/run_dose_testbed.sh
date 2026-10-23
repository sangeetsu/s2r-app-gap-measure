#!/usr/bin/env bash
# Frame-matched appearance dose-response testbed (ledger §15.47), v2 2026-10-15 15:00: function-local variables (v1 clobbered the loop
# variable -> garbled cell names), success = rendered frames == captured states, per-attempt logs, hardened capture (1050 states).
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.dosetb.lock; flock -n 9 || exit 0
source tools/grid_imnet.conf; PY=${PY:-python}; ST=/mnt/features/carla_dose_states; OUT=/mnt/features/carla_doses; R=results/calib_dose; mkdir -p $ST $OUT $R
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[dose $(date '+%F %T')] $*"; }
TOWNS="${TOWNS:-Town01 Town02 Town05 Town10HD}"; DOSES="d00_ref d01_sun25 d02_sun10 d03_sun3 d04_night d05_fog25 d06_fog50 d07_fog75 d08_wet d09_rain"
nf(){ ls $OUT/$1/frame_*.jpg 2>/dev/null | wc -l; }
ns(){ local n; n=$(wc -l < $ST/$1/states.jsonl 2>/dev/null || echo 1); echo $(( n - 1 )); }
start_carla(){ pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null; sleep 5
  (cd ~/CARLA/CARLA_0.9.16 && setsid nohup systemd-run --user --scope -q --slice=research.slice ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 > /dev/null 2>&1 < /dev/null &)
  until (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null; do sleep 5; done; sleep 30; log "CARLA started"; }
need=0; for town in $TOWNS; do for dose in $DOSES; do [ "$(ns $town)" -ge 1000 ] && [ "$(nf ${town}_$dose)" -eq "$(ns $town)" ] || need=1; done; done
if [ $need = 1 ]; then start_carla
  for town in $TOWNS; do
    for attempt in 1 2 3; do [ "$(ns $town)" -ge 1000 ] && break
      log "CAPTURE $town (attempt $attempt)"; rm -rf $ST/$town
      timeout 3600 $J $PY tools/carla_state_replay.py capture --town $town --state-dir $ST/$town > logs/dose_capture_${town}_a$attempt.log 2>&1 < /dev/null
      log "  $town: $(ns $town) states"; [ "$(ns $town)" -ge 1000 ] || start_carla; done
    [ "$(ns $town)" -ge 1000 ] || { log "  SKIP $town: capture failed 3 times"; continue; }
    for dose in $DOSES; do cell=${town}_$dose
      for attempt in 1 2; do [ "$(nf $cell)" -eq "$(ns $town)" ] && break; rm -rf $OUT/$cell; log "RENDER $cell (attempt $attempt)"
        timeout 1800 $J $PY tools/carla_state_replay.py render --state-dir $ST/$town --out-dir $OUT/$cell --dose $dose > logs/dose_render_${cell}_a$attempt.log 2>&1 < /dev/null
        [ "$(nf $cell)" -eq "$(ns $town)" ] || start_carla; done
      log "  $cell: $(nf $cell)/$(ns $town) frames"; done; done
  pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null; fi
CELLS=$(for town in $TOWNS; do for dose in $DOSES; do [ "$(ns $town)" -ge 1000 ] && [ "$(nf ${town}_$dose)" -eq "$(ns $town)" ] && echo ${town}_$dose; done; done)
log "rendering done: $(echo $CELLS | wc -w) complete cells"
featq(){ local nc f allc; allc=$(for d in $OUT/Town*/; do b=$(basename $d); t=${b%%_d*}; [ "$(ns $t)" -ge 1000 ] && [ "$(nf $b)" -eq "$(ns $t)" ] && echo $b; done); nc=$(echo $allc | wc -w); f=$OUT/spatial_features_dinov3_${nc}cells.npz; [ -f $f ] || $J $PY tools/extract_dinov3_features.py --kind carla --root $OUT --img-h 352 --img-w 1216 --out $f > logs/dose_features.log 2>&1; log "features done ($f)"; }
featq &
train(){ local cell=$1 res ck; res=$R/src_${cell}_kittival.json; ck=checkpoints/anchind_$cell/last.pt
  [ -f $res ] || { log "TRAIN $cell"; rm -rf checkpoints/anchind_$cell; $J $PY tools/train_detector.py --init $INIT --source-data carla --source-root $OUT --source-cells $cell --source-frames-slice 0:800 \
     --epochs $EPOCHS --lr-steps $STEPS --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed 42 --out checkpoints/anchind_$cell --result $res > logs/dose_train_$cell.log 2>&1; }
  [ -f $R/src_${cell}_cityscapesval.json ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $R/src_${cell}_cityscapesval.json > /dev/null 2>&1; }
ev(){ local src=$1 tgt=$2 o; o=$R/eval_${src}__on__${tgt}.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/anchind_$src/last.pt --cell $tgt --carla-root $OUT --frames-slice 800:1100 --result $o > /dev/null 2>&1; }
for town in $TOWNS; do echo "$CELLS" | grep -qw ${town}_d00_ref || continue; ref=${town}_d00_ref; train $ref
  for dose in $DOSES; do echo "$CELLS" | grep -qw ${town}_$dose && ev $ref ${town}_$dose; done; log "within-CARLA curve done: $town"; done
for town in $TOWNS; do for dose in $DOSES; do cell=${town}_$dose; [ $dose = d00_ref ] && continue; echo "$CELLS" | grep -qw $cell || continue
  train $cell; ev $cell $cell; ev $cell ${town}_d00_ref; done; done
wait; log "=== dose testbed complete ==="
