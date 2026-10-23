#!/usr/bin/env bash
# Ledger §15.92: evaluate FCOS / Swin-T reference detectors on round-2 and round-3 renders (after predictions.json exists), then score.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; OUT=results/calib_formula_families; PY=${PY:-python}
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"
[ -f $OUT/predictions.json ] || exit 1
for f in fcos:anchina swin:anchins; do fam=${f%%:*}; pre=${f##*:}
  for t in Town01 Town10HD; do
    for d in d19_rain_fog25 d20_rain_fog50 d21_fog15 d22_fog45 d23_sun0 d24_night_wet; do o=$OUT/eval_${fam}_${t}_$d.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/${pre}_${t}_d00_ref/last.pt --cell ${t}_$d --carla-root /mnt/features/carla_doses_ho2 --frames-slice 800:1100 --result $o > /dev/null 2>&1; done
    for d in d25_fog5 d26_fog30 d27_fog55 d28_fog80 d29_sun15 d30_sun_m5 d31_night_fog25 d32_dust90 d33_rain_fog75; do o=$OUT/eval_${fam}_${t}_$d.json; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/${pre}_${t}_d00_ref/last.pt --cell ${t}_$d --carla-root /mnt/features/carla_doses_ho3 --frames-slice 800:1100 --result $o > /dev/null 2>&1; done
  done; done
echo "[formfam $(date '+%F %T')] evals done: $(ls $OUT/eval_*.json | wc -l)"
