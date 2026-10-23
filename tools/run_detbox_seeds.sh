#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."; PY=${PY:-python}
for s in 17 100; do [ -f results/calib_anchored/detector_box_gap_s$s.json ] && grep -q Town03_SoftRainNoon results/calib_anchored/detector_box_gap_s$s.json && continue
  $PY tools/detector_box_gap.py --seed $s --out results/calib_anchored/detector_box_gap_s$s.json 2>&1 | grep -E "^\[dbg\]|Traceback|Error"; done; echo "=== detbox seeds complete ==="
