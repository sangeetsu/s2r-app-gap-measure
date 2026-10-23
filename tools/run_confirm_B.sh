#!/usr/bin/env bash
# Candidate B on the fresh confirmation cells: features -> nnd -> predictions (before outcomes) -> score when detectors are done (ledger §15.45).
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.confirmB.lock; flock -n 9 || exit 0
PY=${PY:-python}; OUT=/mnt/features/carla_confirm; J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"
until grep -q "generation done" logs/confirm_chou.log 2>/dev/null; do sleep 300; done
CELLS=$(for d in $OUT/Town*/; do [ "$(ls $d/frame_*.jpg 2>/dev/null | wc -l)" -ge 1000 ] && basename $d; done); NC=$(echo $CELLS | wc -w); FEAT=$OUT/spatial_features_dinov3_${NC}cells.npz
[ -f $FEAT ] || $J $PY tools/extract_dinov3_features.py --kind carla --root $OUT --img-h 352 --img-w 1216 --out $FEAT 2>&1 | tail -1
SRC=$(for c in $CELLS; do echo -n "$OUT:$FEAT:$c "; done)
$J -p MemoryMax=8G $PY tools/nnd_to_real.py --sources $SRC --out results/manifold_markers/nnd_confirm.json 2>&1 | grep "\[nnd\]"
until grep -q "confirmation complete" logs/confirm_chou.log 2>/dev/null; do sleep 300; done
$PY tools/score_confirm_B.py results/manifold_markers/nnd_confirm.json $OUT results/calib_imnet_confirm results/manifold_markers/confirm_predictions_B.json 0.025 | tee results/manifold_markers/confirm_score_B.txt
echo "=== candidate B complete ==="
