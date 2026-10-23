#!/usr/bin/env bash
# Foggy Cityscapes third real target (ledger §15.59): links -> DINOv3 features per beta -> evals of dose detectors -> FID-to-foggy -> analysis.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.foggycs.lock; flock -n 9 || exit 0
until grep -q "foggy complete" logs/foggy_download.log 2>/dev/null; do sleep 300; done
PY=${PY:-python}; FR=/mnt/features/cityscapes_foggy; R=results/calib_dose_foggy; mkdir -p $R
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[foggycs $(date '+%F %T')] $*"; }
$PY tools/foggy_setup_and_fid.py links
HW=$($PY -c "import numpy as np; t=np.load('$HOME/datasets/cityscapes/spatial_features_dinov3_val.npz',allow_pickle=True)['target_hw']; print(int(t[0])*16, int(t[1])*16)"); set -- $HW; IH=$1; IW=$2; log "extraction size ${IH}x${IW} (matches the clear CS val cache)"
for b in 0.005 0.01 0.02; do F=$FR/spatial_features_dinov3_foggy_b$b.npz; [ -f $F ] || $J $PY tools/extract_dinov3_features.py --kind cityscapes --root $FR/b$b/leftImg8bit/val --img-h $IH --img-w $IW --out $F > logs/foggy_feat_b$b.log 2>&1; done; log "features done"
TOWNS="Town01 Town02 Town03 Town04 Town05 Town06 Town07 Town10HD"; DOSES="d00_ref d01_sun25 d02_sun10 d03_sun3 d04_night d05_fog25 d06_fog50 d07_fog75 d08_wet d09_rain"
ev(){ local c=$1 b=$2 o; o=$R/src_${c}_foggy$b.json; [ -f $o ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt checkpoints/anchind_$c/last.pt --cityscapes-root $FR/b$b --result $o > /dev/null 2>&1; }
for t in $TOWNS; do for d in $DOSES; do ev ${t}_$d 0.02; done; for d in d00_ref d05_fog25 d06_fog50 d07_fog75; do ev ${t}_$d 0.005; ev ${t}_$d 0.01; done; log "evals done: $t"; done
systemd-run --user --scope -q --slice=research.slice -p MemoryMax=10G nice -n 10 $PY tools/foggy_setup_and_fid.py fid > logs/foggy_fid.log 2>&1; log "fid-to-foggy done"
$PY tools/analyze_foggy.py | tee $R/analysis.txt; log "=== foggy cs complete ==="
