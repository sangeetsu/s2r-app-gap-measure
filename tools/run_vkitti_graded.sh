#!/usr/bin/env bash
# Ledger §15.93: VKITTI2 graded doses. Waits for the §15.92 evaluations to finish.
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.vkgraded.lock; flock -n 9 || exit 0
until grep -q 'evals done' logs/formula_families.log 2>/dev/null; do sleep 60; done
PY=${PY:-python}; OUT=results/calib_vkitti_graded; mkdir -p $OUT ${OUT}_rep; G=/mnt/features/vkitti2_graded
J="systemd-run --user --scope -q --slice=research.slice nice -n 10 ionice -c2 -n7"; log(){ echo "[vkgraded $(date '+%F %T')] $*"; }
log "blending"; $J $PY tools/vkitti_blend.py > logs/vkgraded_blend.log 2>&1 || { log "BLEND FAILED"; exit 1; }
for s in Scene01 Scene02 Scene18 Scene20; do
  ndone=$(python3 -c "import json,os;f='$OUT/dose_metrics.json';d=json.load(open(f)) if os.path.exists(f) else {};print(sum(1 for c in d if c.startswith('${s}_d')))")
  [ "$ndone" -ge 21 ] && continue
  log "features $s"; $J $PY tools/extract_dinov3_features.py --kind carla --root $G/$s --img-h 352 --img-w 1216 --out $G/$s/feat.npz > logs/vkgraded_feat_$s.log 2>&1 || { log "FEATURES FAILED $s"; exit 1; }
  $J $PY tools/dose_metrics.py --root $G/$s --features $G/$s/feat.npz --out $OUT/dose_metrics.json > logs/vkgraded_metrics_$s.log 2>&1 || { log "METRICS FAILED $s"; exit 1; }
  rm -f -- "/mnt/features/vkitti2_graded/${s:?}/feat.npz" "/mnt/features/vkitti2_graded/${s:?}/feat.z_spatial.npy"; log "metrics done $s"; done
nfr(){ ls $G/$1/$2/frame_*.jpg | wc -l; }
ev(){ local s=$1 c=$2 seed=$3 out=$4 n ntr o; n=$(nfr $s $c); ntr=$(( n * 3 / 4 )); o=$out/eval_${s}_d00_ref__on__$c.json
  [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt checkpoints/vk_${s}_d00_ref_s$seed/last.pt --cell $c --carla-root $G/$s --frames-slice $ntr:$n --result $o > /dev/null 2>&1; }
for s in Scene01 Scene02 Scene18 Scene20; do for c in $(ls $G/$s | grep '_a50$'); do ev $s $c 42 $OUT; ev $s $c 17 ${OUT}_rep; done; done; log "alpha 0.5 evals done"
[ -f $OUT/predictions.json ] || { $PY tools/predict_vkitti_graded.py 2>&1 | tee logs/vkgraded_predict.log; [ -f $OUT/predictions.json ] || { log "PREDICT FAILED"; exit 1; }; }
for s in Scene01 Scene02 Scene18 Scene20; do for c in $(ls $G/$s | grep '_a25$\|_a75$'); do ev $s $c 42 $OUT; ev $s $c 17 ${OUT}_rep; done; done
$PY tools/analyze_vkitti_graded.py > $OUT/analysis.txt 2>&1; log "=== VKITTI graded complete ==="
