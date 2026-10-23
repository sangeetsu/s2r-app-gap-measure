#!/usr/bin/env bash
# Stage 3 after the 24-cell chain: FID/MMD/CORAL/domain-acc distance matrix over ALL cells (+KITTI, Cityscapes) as the
# label-free control, then the consolidated decomposition analysis (ledger §15.27). flock-guarded; idempotent.
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.afterexp3.lock; flock -n 9 || exit 0
until grep -q "\[afterexp\] done" logs/after_expansion.log 2>/dev/null; do sleep 300; done   # FID matrix runs alongside BoS/AoL and stage 2
PY=${PY:-python}; NEW=/mnt/features/carla_new
NC=$(for d in $NEW/*/; do [ "$(ls $d/frame_*.jpg 2>/dev/null | wc -l)" -ge 1000 ] && echo x; done | wc -l)
[ -f results/domain_distance_dinov3_all.json ] || $PY tools/domain_distance_matrix.py --extra-root $NEW --extra-features $NEW/spatial_features_dinov3_${NC}cells.npz \
   --out results/domain_distance_dinov3_all.json 2>&1 | grep -E "saved|Traceback|Error|cells:"
until grep -q "\[afterexp2\] done" logs/after_expansion2.log 2>/dev/null && grep -q "baselines done" logs/after_expansion.log 2>/dev/null; do sleep 300; done
echo "=== decomposition (24-cell grid) ==="; $PY tools/analyze_decomposition.py
echo "[afterexp3] done ($NC new cells)"
