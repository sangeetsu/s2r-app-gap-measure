#!/usr/bin/env bash
# 2026-10-14: memory-safe SEQUENTIAL runner for the remaining 24-cell analyses. Every job runs in its own user cgroup
# with a hard RAM cap and NO swap, low CPU/IO priority, one at a time. A job that exceeds the cap is killed by the
# kernel inside its cgroup instead of taking the desktop/tailscale down. Idempotent; stop everything with:
#   pkill -f run_remaining_safe; pkill -f "tools/(probe_transfer_matrix|detector_box_gap|domain_distance_matrix|bos_aol_baselines)\.py"
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.remaining.lock; flock -n 9 || exit 0
PY=${PY:-python}; NEW=/mnt/features/carla_new; FEAT=$NEW/spatial_features_dinov3_16cells.npz
CAP="systemd-run --user --scope -q -p MemoryMax=12G -p MemoryHigh=10G -p MemorySwapMax=0 nice -n 10 ionice -c2 -n7"
log(){ echo "[safe $(date '+%F %T')] $*"; }
for s in 17 100; do out=results/calib_anchored/probe_matrix_anchored_all_s$s.json; [ -f $out ] && continue
  log "probe matrix seed $s"; $CAP $PY tools/probe_transfer_matrix.py --anchored --seed $s --extra-root $NEW --extra-features $FEAT --out $out 2>&1 | grep -E "saved|Traceback|Error|Killed"; done
log "detector-box ensemble (resumes per source)"; $CAP $PY tools/detector_box_gap.py --probe-seeds 42,17,100 --extra-features $FEAT --out results/calib_anchored/detector_box_gap_all_s42.json 2>&1 | grep -E "^\[dbg\]|Traceback|Error|Killed"
[ -f results/domain_distance_dinov3_all.json ] || { log "FID matrix"; $CAP $PY tools/domain_distance_matrix.py --extra-root $NEW --extra-features $FEAT --out results/domain_distance_dinov3_all.json 2>&1 | grep -E "saved|Traceback|Error|Killed|cells:"; }
log "decomposition (without BoS/AoL)"; $PY tools/analyze_decomposition.py
NEWT=$(for d in $NEW/*/; do basename $d; done | tr '\n' ' ')
[ -f results/calib_anchored/bos_aol.json ] || { log "BoS/AoL (last; longest)"; $CAP $PY tools/bos_aol_baselines.py --targets KITTI Cityscapes Town01_ClearNoon Town01_OvercastNoon Town01_SoftRainNoon Town01_ClearNight Town03_ClearNoon Town03_OvercastNoon Town03_SoftRainNoon Town03_ClearNight $NEWT --out results/calib_anchored/bos_aol.json 2>&1 | grep -E "saved|Traceback|Error|Killed"; }
log "decomposition (final)"; $PY tools/analyze_decomposition.py; log "=== remaining analyses complete ==="
