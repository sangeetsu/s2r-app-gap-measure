#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.afteranch.lock; flock -n 9 || exit 0
until grep -q "anchored grid complete" logs/anchored.log 2>/dev/null; do sleep 60; done
PY=${PY:-python}
$PY tools/probe_transfer_matrix.py --anchored --seed 42 --out results/calib_anchored/probe_matrix_anchored_s42.json 2>&1 | grep -E "saved|Traceback"
$PY tools/analyze_anchored.py results/calib_anchored/probe_matrix_anchored_s42.json
echo "[afteranch] done"
