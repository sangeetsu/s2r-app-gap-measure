#!/usr/bin/env bash
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; PY=${PY:-python}; F=/mnt/features/carla_doses/spatial_features_dinov3_80cells.npz
J="systemd-run --user --scope -q --slice=research.slice -p MemoryMax=12G nice -n 10 ionice -c2 -n7"
$J $PY tools/dose_metrics.py --features $F > logs/dose_metrics_8towns.log 2>&1; echo "[8towns $(date +%T)] metrics done"
$J $PY tools/dose_metrics_detbox.py --features $F > logs/dose_metrics_detbox_8towns.log 2>&1; echo "[8towns $(date +%T)] detbox metrics done"
$PY tools/analyze_dose.py > results/calib_dose/analysis_8towns.txt 2>&1; $PY tools/analyze_dose.py results/calib_dose/dose_metrics_detbox.json > results/calib_dose/analysis_8towns_detbox.txt 2>&1
echo "=== 8-town metrics complete ==="
