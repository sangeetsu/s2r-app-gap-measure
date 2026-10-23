#!/usr/bin/env bash
# Content-fixed appearance intervention: CARLA -quality-level=Low vs the existing Epic cells (ledger §15.40). research.slice; allocator fix.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.lowq.lock; flock -n 9 || exit 0
source tools/grid_imnet.conf; PY=${PY:-python}; OUT=/mnt/features/carla_lowq; R=results/calib_imnet_lowq; mkdir -p $OUT $R
OLD=~/datasets/carla_extracted; NEW=/mnt/features/carla_new; epic_root(){ [ -d $OLD/$1 ] && echo $OLD || echo $NEW; }
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[lowq $(date '+%F %T')] $*"; }
nf(){ ls $OUT/$1/frame_*.jpg 2>/dev/null | wc -l; }
start_carla(){ pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null; sleep 5
  (cd ~/CARLA/CARLA_0.9.16 && setsid nohup systemd-run --user --scope -q --slice=research.slice ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Low > /dev/null 2>&1 < /dev/null &)
  sleep 60; log "CARLA (quality Low) started: $(pgrep -u $USER -f 'CarlaUE4-Linux-Shippin[g]' | head -1)"; }
CELLS="Town01_ClearNoon Town02_ClearNoon Town03_ClearNoon Town04_ClearNoon Town05_ClearNoon Town10HD_ClearNoon"
need=0; for c in $CELLS; do [ "$(nf $c)" -ge 1000 ] || need=1; done
[ $need = 1 ] && start_carla
for c in $CELLS; do town=${c%%_*}; w=${c#*_}
  for attempt in 1 2; do [ "$(nf $c)" -ge 1000 ] && break; start_carla; log "GEN $c low quality (attempt $attempt)"; t0=$(date +%s)
    timeout 3600 $J $PY tools/extract_carla_frames.py --server-profile local_package_largemap --towns $town --weathers $w --frames-per-cell 1000 --seed $((42+attempt)) --out-dir $OUT > logs/lowq_gen_$c.log 2>&1 < /dev/null
    log "  $c: $(nf $c) frames in $(( ($(date +%s)-t0)/60 )) min (seed $((42+attempt)))"; [ "$(nf $c)" -ge 1000 ] || rm -rf $OUT/$c; done; done
pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null
for c in $CELLS; do [ "$(nf $c)" -ge 1000 ] || continue; res=$R/src_${c}_kittival.json; ck=checkpoints/anchinlq_$c/last.pt
  [ -f $res ] || { log "TRAIN $c (low)"; rm -rf checkpoints/anchinlq_$c; $J $PY tools/train_detector.py --init $INIT --source-data carla --source-root $OUT --source-cells $c \
     --source-frames-slice 0:800 --epochs $EPOCHS --lr-steps $STEPS --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed 42 --out checkpoints/anchinlq_$c --result $res 2>&1 | grep -E "^\[train\] eval" | cut -c1-110; }
  o=$R/eval_${c}__on__${c}_low.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $c --carla-root $OUT --frames-slice 800:1000 --result $o 2>&1 | grep RESULT
  o=$R/eval_${c}__on__${c}_epic.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $c --carla-root $(epic_root $c) --frames-slice 800:1000 --result $o 2>&1 | grep RESULT
  o=$R/eval_epic_${c}__on__${c}_low.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/anchin_$c/last.pt --cell $c --carla-root $OUT --frames-slice 800:1000 --result $o 2>&1 | grep RESULT
  o=$R/src_${c}_cityscapesval.json; [ -f $o ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck --result $o > /dev/null 2>&1
done
NC=$(for c in $CELLS; do [ "$(nf $c)" -ge 1000 ] && echo x; done | wc -l); FEAT=$OUT/spatial_features_dinov3_${NC}cells.npz
[ -f $FEAT ] || $J $PY tools/extract_dinov3_features.py --kind carla --root $OUT --img-h 352 --img-w 1216 --out $FEAT 2>&1 | tail -1
[ -f $R/fid_to_real_low.json ] || $J $PY tools/fid_to_real.py --root $OUT --features $FEAT --out $R/fid_to_real_low.json 2>&1 | grep fid2real
$PY tools/analyze_lowq.py | tee $R/lowq_result.txt; log "=== lowq complete ==="
