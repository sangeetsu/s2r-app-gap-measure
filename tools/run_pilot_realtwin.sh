#!/usr/bin/env bash
# Ledger §15.98 real-twin pilot, evaluation stage. Waits for the Cosmos generation (/mnt/features/cosmos/pilot/run_pilot_gen.sh), then in order:
# d51 control cells -> survival -> per-town features + dose metrics -> PREDICTIONS (hashed; refuses if evals exist) -> reference-detector evals -> train d50/d51
# detectors (seeds 42, 17) + real-target and held-out evals -> analysis. Sequential, inside research.slice (31 GB box, §feedback_resource_limits).
export MALLOC_MMAP_THRESHOLD_=131072
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.pilotrt.lock; flock -n 9 || exit 0
source tools/grid_imnet.conf; PY=${PY:-python}; P=/mnt/features/cosmos/pilot; R=results/pilot_realtwin; mkdir -p $R ${R}_rep logs
J="systemd-run --user --scope -q --slice=research.slice -p MemoryMax=20G -p MemorySwapMax=0 nice -n 10 ionice -c2 -n7"; log(){ echo "[pilotrt $(date '+%F %T')] $*"; }
TOWNS="Town03 Town05"
until grep -q "PILOT GENERATION COMPLETE" $P/pilot_gen.log 2>/dev/null; do sleep 300; done
for t in $TOWNS; do n=$(ls $P/${t}_d50_realtwin/frame_*.jpg 2>/dev/null | wc -l); [ "$n" -eq 1050 ] || { log "INCOMPLETE $t d50: $n/1050 — stopping for the §15.98 seed-3 / drop rule"; exit 1; }; done
log "generation complete; building d51 control cells"
$J $PY - <<'EOF'
import os, shutil
from PIL import Image
P = "/mnt/features/cosmos/pilot"
for t in ("Town03", "Town05"):
    src = f"/mnt/features/carla_doses/{t}_d00_ref"; dst = f"{P}/{t}_d51_resample"; os.makedirs(dst, exist_ok=True)
    for f in sorted(os.listdir(src)):
        if f.endswith(".jpg") and not os.path.exists(f"{dst}/{f}"):
            im = Image.open(f"{src}/{f}").convert("RGB"); assert im.size == (1242, 375), im.size
            c = Image.new("RGB", (1242, 684)); c.paste(im, (0, 154))
            c = c.resize((1280, 704), Image.LANCZOS).resize((1242, 684), Image.LANCZOS).crop((0, 154, 1242, 529))
            c.save(f"{dst}/{f}", quality=95); shutil.copy(f"{src}/{f[:-4]}.txt", f"{dst}/{f[:-4]}.txt")
    print(t, len([x for x in os.listdir(dst) if x.endswith(".jpg")]))
EOF
NAMES=$(for k in $(seq 800 1049); do printf "frame_%06d " $k; done)
for t in $TOWNS; do [ -f $R/survival_$t.json ] || $J $PY tools/cosmos_survival.py --orig $P/${t}_d00_ref --restyled $P/${t}_d50_realtwin --names $NAMES --control-size 1280x387 --out $R/survival_$t.json > logs/pilotrt_survival_$t.log 2>&1 || { log "SURVIVAL FAILED $t"; exit 1; }; done
python3 -c "import json; json.dump({t: json.load(open('$R/survival_'+t+'.json')) for t in '$TOWNS'.split()}, open('$R/survival.json','w'))"; log "survival done"
TT=/mnt/features/cosmos/pilot_feat; mkdir -p $TT
for t in $TOWNS; do
  ndone=$(python3 -c "import json,os;f='$R/dose_metrics.json';d=json.load(open(f)) if os.path.exists(f) else {};print(sum(1 for c in d if c.startswith('${t}_d')))"); [ "$ndone" -ge 3 ] && continue
  mkdir -p $TT/$t; for c in ${t}_d00_ref ${t}_d50_realtwin ${t}_d51_resample; do [ -e $TT/$t/$c ] || ln -s "$(readlink -f $P/$c)" $TT/$t/$c; done
  log "features $t"; $J $PY tools/extract_dinov3_features.py --kind carla --root $TT/$t --img-h 352 --img-w 1216 --out $TT/$t/feat.npz > logs/pilotrt_features_$t.log 2>&1 || { log "FEATURES FAILED $t"; exit 1; }
  log "dose metrics $t"; $J $PY tools/dose_metrics.py --root $TT/$t --features $TT/$t/feat.npz --out $R/dose_metrics.json > logs/pilotrt_metrics_$t.log 2>&1 || { log "METRICS FAILED $t"; exit 1; }
  rm -f -- "/mnt/features/cosmos/pilot_feat/${t:?}/feat.npz" "/mnt/features/cosmos/pilot_feat/${t:?}/feat.z_spatial.npy"; log "town $t metrics done; cache removed"
done
[ -f $R/predictions.json ] || { $PY tools/predict_pilot_realtwin.py > logs/pilotrt_predict.log 2>&1; [ -f $R/predictions.json ] || { log "PREDICTION FAILED"; exit 1; }; }
log "predictions written ($(cat $R/predictions.sha256))"
ev(){ local ck=$1 cell=$2 o=$3; [ -f $o ] || $J $PY tools/eval_detector_on_carla_cell.py --ckpt $ck --cell $cell --carla-root $P --frames-slice 800:1100 --result $o > /dev/null 2>&1; }
for t in $TOWNS; do for d in d50_realtwin d51_resample; do
  ev checkpoints/anchind_${t}_d00_ref/last.pt ${t}_$d $R/eval_${t}_d00_ref__on__${t}_$d.json; ev checkpoints/anchindr_${t}_d00_ref/last.pt ${t}_$d ${R}_rep/eval_${t}_d00_ref__on__${t}_$d.json; done; done
log "reference-detector evals done"
for t in $TOWNS; do for d in d50_realtwin d51_resample; do c=${t}_$d
  for s in 42 17; do [ $s = 42 ] && { RR=$R; ck=checkpoints/pilotrt_${c}; } || { RR=${R}_rep; ck=checkpoints/pilotrtr_${c}; }
    [ -f $RR/src_${c}_kittival.json ] || { log "TRAIN $c seed $s"; rm -rf -- "checkpoints/$(basename ${ck:?})"; $J $PY tools/train_detector.py --init $INIT --source-data carla --source-root $P --source-cells $c --source-frames-slice 0:800 \
      --epochs $EPOCHS --lr-steps $STEPS --warmup-iters $WARMUP --batch-size 4 --lr $LR --seed $s --out $ck --result $RR/src_${c}_kittival.json > logs/pilotrt_train_${c}_s$s.log 2>&1 || { log "TRAIN FAILED $c $s"; exit 1; }; }
    [ -f $RR/src_${c}_cityscapesval.json ] || $J $PY tools/eval_detector_on_cityscapes.py --ckpt $ck/last.pt --result $RR/src_${c}_cityscapesval.json > /dev/null 2>&1
    ev $ck/last.pt $c $RR/eval_${c}__on__${c}.json; ev $ck/last.pt ${t}_d00_ref $RR/eval_${c}__on__${t}_d00_ref.json
  done; done; done
$PY tools/analyze_pilot_realtwin.py > $R/analysis.txt 2>&1; log "=== real-twin pilot complete ==="; cat $R/analysis.txt
