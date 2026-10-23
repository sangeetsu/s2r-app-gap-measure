#!/usr/bin/env bash
# Distance matrix, queued behind the fairness chain. Both mmap huge caches; running them
# together pushed a 31 GB box to 0 available and got the previous run OOM-killed.
set -u
cd "$(dirname "$0")/.."
exec 9>/tmp/.distq.lock; flock -n 9 || exit 0
while pgrep -f "run_fairness_chain|pretrain_dense_head_coco|train_dense_detector" > /dev/null; do sleep 120; done
echo "[distq $(date '+%F %T')] fairness chain done; starting distance matrix"
${PY:-python} tools/domain_distance_matrix.py \
  --out results/domain_distance_dinov3.json
