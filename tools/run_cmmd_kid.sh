#!/usr/bin/env bash
# E4: CMMD + KID baselines on the 80-cell frame-matched testbed (results/calib_dose/dose_cmmd_kid.json). Model cache on the data disk.
export MALLOC_MMAP_THRESHOLD_=131072 HF_HOME=/mnt/features/hf_cache HF_HUB_CACHE=/mnt/features/hf_cache
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.cmmd.lock; flock -n 9 || exit 0
systemd-run --user --scope -q --slice=research.slice -p MemoryMax=12G nice -n 10 ionice -c2 -n7 ${PY:-python} tools/dose_cmmd_kid.py > logs/cmmd_kid.log 2>&1
echo "[cmmd $(date '+%F %T')] exit $?" >> logs/cmmd_kid.log
