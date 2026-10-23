#!/usr/bin/env bash
# GRID EXPANSION (plan 2026-10-13): new towns x the SAME 4 weathers as the existing grid,
# 1000 frames/cell, to /mnt/features/carla_new. Sequential (one CARLA server). Idempotent:
# a cell with >=1000 frames is skipped; a town that fails to load is logged and skipped.
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.carlaexp.lock; flock -n 9 || exit 0
PY=${PY:-python}; OUT=/mnt/features/carla_new; mkdir -p $OUT logs
log(){ echo "[carla-exp $(date '+%F %T')] $*"; }
for town in Town02 Town04 Town05 Town10HD; do
  for w in ClearNoon OvercastNoon SoftRainNoon ClearNight; do
    n=$(ls $OUT/${town}_${w}/frame_*.jpg 2>/dev/null | wc -l)
    [ "$n" -ge 1000 ] && { log "SKIP ${town}_${w} ($n frames)"; continue; }
    log "GEN ${town}_${w}"; t0=$(date +%s)
    $PY tools/extract_carla_frames.py --server-profile local_package --towns $town --weathers $w \
       --frames-per-cell 1000 --out-dir $OUT 2>&1 | grep -E "saved 1000|Error|error|Traceback|not found|failed" | tail -3
    n=$(ls $OUT/${town}_${w}/frame_*.jpg 2>/dev/null | wc -l)
    log "  ${town}_${w}: $n frames in $(( ($(date +%s)-t0)/60 )) min"
  done
done
log "=== expansion complete ==="
