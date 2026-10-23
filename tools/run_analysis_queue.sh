#!/usr/bin/env bash
# CPU/IO analysis queue in research.slice (2026-10-14). Detector-independent analyses first (shared by the COCO and ImageNet
# grids). The FID matrix gets a per-job 10 GB cap and a memory sampler: the 24-cell run on 2026-10-13 grew to 23.7 GB anon.
# glibc: send allocations >128 KB to mmap so freed image buffers return to the OS (dataset loops otherwise fragment the heap:
# 1.3 MB retained per decoded frame -> 23.7 GB OOM on 2026-10-14; ledger §15.31)
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.analysisq.lock; flock -n 9 || exit 0
PY=${PY:-python}; NEW=/mnt/features/carla_new; FEAT=$NEW/spatial_features_dinov3_16cells.npz
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[aq $(date '+%F %T')] $*"; }
while pgrep -u $USER -f "tools/probe_transfer_matrix" >/dev/null; do sleep 30; done
for s in 17 100; do out=results/calib_anchored/probe_matrix_anchored_all_s$s.json; [ -f $out ] && continue
  log "probe matrix seed $s"; $J $PY tools/probe_transfer_matrix.py --anchored --seed $s --extra-root $NEW --extra-features $FEAT --out $out > logs/pm_all_s$s.log 2>&1; done
if [ ! -f results/domain_distance_dinov3_all.json ]; then
  log "FID matrix (per-job cap 10G, sampler -> logs/dist_all_mem.log)"
  systemd-run --user --scope -q --slice=research.slice -p MemoryMax=10G nice -n 10 ionice -c2 -n7 $PY tools/domain_distance_matrix.py \
     --extra-root $NEW --extra-features $FEAT --out results/domain_distance_dinov3_all.json > logs/dist_all.log 2>&1 &
  sleep 5; while pgrep -u $USER -f "tools/domain_distance" >/dev/null; do
    p=$(pgrep -u $USER -f "tools/domain_distance" | head -1)
    echo "$(date +%T) anon=$(awk '/RssAnon/{printf "%.2fGB",$2/1048576}' /proc/$p/status 2>/dev/null) last=$(tail -1 logs/dist_all.log | cut -c1-80)" >> logs/dist_all_mem.log; sleep 60; done
  wait; log "FID matrix exit; result: $(ls results/domain_distance_dinov3_all.json 2>&1 | tail -1)"; fi
log "=== shared analyses complete ==="
