#!/usr/bin/env bash
# Writes the competing-model predictions for towns3 the moment the FID predictions appear (before its first detector finishes, ~10 min).
cd "$(dirname "$0")/.."; P=results/calib_imnet/prereg_predictions_towns3.json
until [ -f $P ]; do sleep 30; done; sleep 5
${PY:-python} tools/prereg_competing.py $P ${P%.json}_competing.json >> logs/towns3.log 2>&1
