#!/usr/bin/env bash
# Ledger §15.100: re-render the clear-noon reference of every town with CARLA ground-truth instance segmentation + depth (same state captures).
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; PY=${PY:-python}; ST=/mnt/features/carla_dose_states; OUT=/mnt/features/carla_gt
J="systemd-run --user --scope -q --slice=research.slice -p MemoryMax=12G -p MemorySwapMax=0"; log(){ echo "[gt $(date '+%F %T')] $*"; }
pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null; sleep 5
(cd ~/CARLA/CARLA_0.9.16 && setsid nohup systemd-run --user --scope -q --slice=research.slice ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 > /dev/null 2>&1 < /dev/null &)
until (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null; do sleep 5; done; sleep 30; log "CARLA started"
for t in ${TOWNS:-Town03 Town05 Town01 Town02 Town04 Town06 Town07 Town10HD}; do
  timeout 1800 $J $PY tools/carla_state_replay.py render --state-dir $ST/$t --out-dir $OUT/${t}_d00_ref --dose d00_ref --instance --depth > logs/gt_render_$t.log 2>&1 < /dev/null
  n=$(ls $OUT/${t}_d00_ref/frame_*_inst.png 2>/dev/null | wc -l); same=$(cd $OUT/${t}_d00_ref && for f in frame_*.txt; do cmp -s $f /mnt/features/carla_doses/${t}_d00_ref/$f && echo 1; done | wc -l)
  log "$t: $n inst/depth frames; labels identical to the reference cell: $same/$(ls /mnt/features/carla_doses/${t}_d00_ref/*.txt | wc -l)"
done
pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null; log "GT render done"
