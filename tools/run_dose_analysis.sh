#!/usr/bin/env bash
# After dose features exist: metrics; after the testbed completes: analysis. research.slice; allocator fix.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.doseana.lock; flock -n 9 || exit 0; PY=${PY:-python}
until grep -q "features done" logs/dose_testbed.log 2>/dev/null; do sleep 300; done
F=$(ls /mnt/features/carla_doses/spatial_features_dinov3_*cells.npz | tail -1)
systemd-run --user --scope -q --slice=research.slice -p MemoryMax=12G nice -n 10 $PY tools/dose_metrics.py --features $F > logs/dose_metrics.log 2>&1
until grep -q "dose testbed complete" logs/dose_testbed.log 2>/dev/null; do sleep 300; done
$PY tools/analyze_dose.py | tee results/calib_dose/analysis.txt; echo "=== dose analysis complete ==="
