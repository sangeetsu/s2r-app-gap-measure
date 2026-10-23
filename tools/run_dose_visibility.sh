#!/usr/bin/env bash
# Visibility control (reviewer item 3): reference detectors re-evaluated on all doses with GT/pred boxes shorter than 25 / 40 px removed.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; PY=${PY:-python}; OUT=/mnt/features/carla_doses; J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"
DOSES="d00_ref d01_sun25 d02_sun10 d03_sun3 d04_night d05_fog25 d06_fog50 d07_fog75 d08_wet d09_rain"
for h in 25 40; do R=results/calib_dose_h$h; mkdir -p $R
  for town in Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10HD; do ref=${town}_d00_ref
    for dose in $DOSES; do o=$R/eval_${ref}__on__${town}_$dose.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/anchind_$ref/last.pt --cell ${town}_$dose --carla-root $OUT --frames-slice 800:1100 --min-box-height $h --result $o > /dev/null 2>&1; done
    echo "[vis $(date +%T)] h>=$h $town done"; done
  DOSE_DIR=$R $PY tools/analyze_dose.py results/calib_dose/dose_metrics.json > $R/analysis.txt 2>&1; done
echo "=== visibility control complete ==="
