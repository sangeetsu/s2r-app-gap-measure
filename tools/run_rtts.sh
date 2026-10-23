#!/usr/bin/env bash
# Ledger §15.69 (E3): RTTS real-fog target. Waits for the HF mirror download; count check; DINOv3 features; evals of 80 + 32 dose detectors; FID; analysis.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.rtts.lock; flock -n 9 || exit 0
PY=${PY:-python}; RT=/mnt/features/downloads/rtts; R=results/calib_dose_rtts; mkdir -p $R ${R}_rep /mnt/features/rtts_features
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[rtts $(date '+%F %T')] $*"; }
until [ "$(ls $RT/VOCdevkit/VOC2007/JPEGImages 2>/dev/null | wc -l)" -ge 4322 ]; do sleep 60; done
log "download complete: $($PY -I tools/eval_detector_on_rtts.py --ckpt none --root $RT --result /dev/null --count-only)"
IMG=$RT/VOCdevkit/VOC2007/JPEGImages; F=/mnt/features/rtts_features/spatial_features_dinov3_rtts.npz
[ -f $F ] || $J $PY tools/extract_dinov3_features.py --kind generic --root $IMG --img-h 480 --img-w 640 --out $F > logs/rtts_features.log 2>&1 || { log "FEATURES FAILED"; exit 1; }
TOWNS="Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10HD"; DOSES="d00_ref d01_sun25 d02_sun10 d03_sun3 d04_night d05_fog25 d06_fog50 d07_fog75 d08_wet d09_rain"
ev(){ local ck=$1 o=$2; [ -f $o ] || $J $PY tools/eval_detector_on_rtts.py --ckpt $ck --root $RT --result $o > /dev/null 2>&1; }
for t in $TOWNS; do for d in $DOSES; do ev checkpoints/anchind_${t}_$d/last.pt $R/src_${t}_${d}_rtts.json; done
  for d in d00_ref d05_fog25 d06_fog50 d07_fog75; do ev checkpoints/anchindr_${t}_$d/last.pt ${R}_rep/src_${t}_${d}_rtts.json; done; log "evals done: $t"; done
[ -f $R/fid_to_rtts.json ] || systemd-run --user --scope -q --slice=research.slice -p MemoryMax=10G nice -n 10 $PY tools/rtts_fid_analyze.py fid > logs/rtts_fid.log 2>&1
$PY tools/rtts_fid_analyze.py analyze | tee $R/analysis.txt; log "=== RTTS complete ==="
