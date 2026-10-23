#!/usr/bin/env bash
# Ledger §15.87 (round 3, calibration formula). Derived from run_dose_heldout.sh;: render held-out doses from the existing state captures, compute metrics, WRITE PREDICTIONS, then evaluate the existing
# reference detectors (seeds 42 + 17). Order is enforced: predict_heldout_doses.py refuses to run if any held-out eval exists, and evals
# start only after predictions.json exists.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.doseho3.lock; flock -n 9 || exit 0
PY=${PY:-python}; ST=/mnt/features/carla_dose_states; SRC=/mnt/features/carla_doses; OUT=/mnt/features/carla_doses_ho3; R=results/calib_dose_ho3; mkdir -p $OUT $R ${R}_rep
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[doseho3 $(date '+%F %T')] $*"; }
TOWNS="Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10HD"; NEW="d25_fog5 d26_fog30 d27_fog55 d28_fog80 d29_sun15 d30_sun_m5 d31_night_fog25 d32_dust90 d33_rain_fog75"
nf(){ ls $OUT/$1/frame_*.jpg 2>/dev/null | wc -l; }; ns(){ echo $(( $(wc -l < $ST/$1/states.jsonl) - 1 )); }
clear_cell(){ rm -rf -- "/mnt/features/carla_doses_ho3/${1:?}"; }
start_carla(){ pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null; sleep 5
  (cd ~/CARLA/CARLA_0.9.16 && setsid nohup systemd-run --user --scope -q --slice=research.slice ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 > /dev/null 2>&1 < /dev/null &)
  until (echo > /dev/tcp/127.0.0.1/2000) 2>/dev/null; do sleep 5; done; sleep 30; log "CARLA started"; }
for t in $TOWNS; do for c in ${t}_d00_ref ${t}_d06_fog50; do [ -e $OUT/$c ] || ln -s $SRC/$c $OUT/$c; done; done
need=0; for t in $TOWNS; do for d in $NEW; do [ "$(nf ${t}_$d)" -eq "$(ns $t)" ] || need=1; done; done
if [ $need = 1 ]; then start_carla
  for t in $TOWNS; do for d in $NEW; do c=${t}_$d
    for attempt in 1 2; do [ "$(nf $c)" -eq "$(ns $t)" ] && break; clear_cell $c; log "RENDER $c (attempt $attempt)"
      timeout 1800 $J $PY tools/carla_state_replay.py render --state-dir $ST/$t --out-dir $OUT/$c --dose $d > logs/doseho3_render_${c}_a$attempt.log 2>&1 < /dev/null
      [ "$(nf $c)" -eq "$(ns $t)" ] || start_carla; done
    log "  $c: $(nf $c)/$(ns $t) frames"; done; done
  pkill -u $USER -f "CarlaUE4-Linux-Shippin[g]" 2>/dev/null; fi
for t in $TOWNS; do for d in $NEW; do c=${t}_$d; [ "$(nf $c)" -eq "$(ns $t)" ] || { log "EXCLUDE $c (render incomplete: $(nf $c)/$(ns $t))"; clear_cell $c; }; done; done
# Features per town (2026-10-17 19:50, after the all-cells sidecar (314 GB, preallocated) filled /mnt/features): one town's 11 cells at a time
# (~40 GB), dose metrics appended to the same JSON, the town's feature cache deleted afterwards. Same extractor, size and metric code.
TT=/mnt/features/carla_doses_ho3_town; mkdir -p $TT
for t in $TOWNS; do
  ndone=$(python3 -c "import json,os;f='$R/dose_metrics.json';d=json.load(open(f)) if os.path.exists(f) else {};print(sum(1 for c in d if c.startswith('${t}_d')))")
  ncell=$(ls -d $OUT/${t}_d* | wc -l); [ "$ndone" -ge "$ncell" ] && { log "metrics already done: $t"; continue; }
  mkdir -p $TT/$t; for c in $(ls -d $OUT/${t}_d* | xargs -n1 basename); do [ -e $TT/$t/$c ] || ln -s "$(readlink -f $OUT/$c)" $TT/$t/$c; done
  log "features $t ($ncell cells)"; $J $PY tools/extract_dinov3_features.py --kind carla --root $TT/$t --img-h 352 --img-w 1216 --out $TT/$t/feat.npz > logs/doseho3_features_$t.log 2>&1 || { log "FEATURES FAILED $t"; exit 1; }
  log "dose metrics $t"; $J $PY tools/dose_metrics.py --root $TT/$t --features $TT/$t/feat.npz --out $R/dose_metrics.json > logs/doseho3_metrics_$t.log 2>&1 || { log "METRICS FAILED $t"; exit 1; }
  rm -f -- "/mnt/features/carla_doses_ho3_town/${t:?}/feat.npz" "/mnt/features/carla_doses_ho3_town/${t:?}/feat.z_spatial.npy"; log "town $t done; cache removed"
done
[ -f $R/predictions.json ] || { $PY tools/predict_heldout3.py 2>&1 | tee logs/doseho3_predict.log; [ -f $R/predictions.json ] || { log "PREDICTION FAILED"; exit 1; }; }
until [ -L checkpoints ]; do log "waiting for the checkpoint move"; sleep 120; done
ev(){ local ck=$1 cell=$2 o=$3; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $cell --carla-root $OUT --frames-slice 800:1100 --result $o > /dev/null 2>&1; }
for t in $TOWNS; do for d in $NEW d06_fog50; do c=${t}_$d; [ -d $OUT/$c ] || continue
  ev checkpoints/anchind_${t}_d00_ref/last.pt $c $R/eval_${t}_d00_ref__on__$c.json; ev checkpoints/anchindr_${t}_d00_ref/last.pt $c ${R}_rep/eval_${t}_d00_ref__on__$c.json; done; log "evals done: $t"; done
$PY tools/analyze_heldout3.py > $R/analysis.txt 2>&1; log "=== held-out doses complete ==="
