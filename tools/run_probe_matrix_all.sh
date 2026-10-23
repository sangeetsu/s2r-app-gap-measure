#!/usr/bin/env bash
# All-cell anchored probe matrix, 3 probe seeds (the 2026-10-13 chain run crashed on a list-aliasing bug; fixed).
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.pmall.lock; flock -n 9 || exit 0
PY=${PY:-python}; NEW=/mnt/features/carla_new
for s in 42 17 100; do out=results/calib_anchored/probe_matrix_anchored_all_s$s.json; [ -f $out ] && continue
  $PY tools/probe_transfer_matrix.py --anchored --seed $s --extra-root $NEW --extra-features $NEW/spatial_features_dinov3_16cells.npz --out $out 2>&1 | grep -E "saved|Traceback|Error" ; done
echo "=== probe matrix all seeds complete ==="
