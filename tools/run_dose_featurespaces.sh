#!/usr/bin/env bash
# R-B feature-space robustness (plan 2026-10-16): DINOv2-L and CLIP features on the 40 dose cells, then the within-CARLA dose metrics
# and R2 scoring per feature space. research.slice; allocator fix; flock'd.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.dosefs.lock; flock -n 9 || exit 0
PY=${PY:-python}; OUT=/mnt/features/carla_doses; J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[dosefs $(date '+%F %T')] $*"; }
for fs in dinov2 clip; do F=$OUT/spatial_features_${fs}_40cells.npz
  [ -f $F ] || { log "extract $fs"; $J $PY tools/extract_${fs}_features.py --kind carla --root $OUT --out $F > logs/dose_feat_$fs.log 2>&1; }
  log "metrics $fs"; systemd-run --user --scope -q --slice=research.slice -p MemoryMax=12G nice -n 10 $PY tools/dose_metrics.py --features $F --skip-real --out results/calib_dose/dose_metrics_$fs.json > logs/dose_metrics_$fs.log 2>&1
  $PY tools/analyze_dose.py results/calib_dose/dose_metrics_$fs.json > results/calib_dose/analysis_$fs.txt 2>&1; log "analysis_$fs.txt written"; done
log "=== feature-space robustness complete ==="
