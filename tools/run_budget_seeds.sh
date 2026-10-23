#!/usr/bin/env bash
set -u; cd "$(dirname "$0")/.."; PY=${PY:-python}
for s in 42 17 100; do [ -f results/calib_anchored/probe_budget_curve_s$s.json ] && [ $s != 42 ] && continue
  $PY tools/probe_budget_curve.py --seed $s 2>&1 | grep -v "^\[matrix\]"; done; echo "=== budget seeds complete ==="
