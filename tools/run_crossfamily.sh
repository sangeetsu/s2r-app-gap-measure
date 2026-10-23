#!/usr/bin/env bash
# Ledger §15.79: predictions -> evals of the FCOS and Swin-T reference detectors on the round-1 held-out renders -> analysis.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.crossfam.lock; flock -n 9 || exit 0
PY=${PY:-python}; OUT=results/calib_crossfamily; mkdir -p $OUT; J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"
[ -f $OUT/predictions.json ] || $PY tools/predict_crossfamily.py | tee logs/crossfam_predict.log
[ -f $OUT/predictions.json ] || exit 1
for f in fcos:anchina swin:anchins; do fam=${f%%:*}; pre=${f##*:}
  for t in Town01 Town10HD; do for d in d10_fog35 d11_fog65 d12_fog90 d13_sun6 d14_night_rain d15_night_fog50 d16_sun10_wet d17_cloud90 d18_dust50; do o=$OUT/eval_${fam}_${t}_$d.json
    [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/${pre}_${t}_d00_ref/last.pt --cell ${t}_$d --carla-root /mnt/features/carla_doses_ho --frames-slice 800:1100 --result $o > /dev/null 2>&1; done; done; done
$PY tools/analyze_crossfamily.py > $OUT/analysis.txt 2>&1; echo "[crossfam $(date '+%F %T')] done" 
