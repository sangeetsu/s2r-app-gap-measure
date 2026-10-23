#!/usr/bin/env bash
# Re-run VKITTI2 stage A after the first pass ends, to retrain the clone detectors that failed when /mnt/features filled (2026-10-17 19:43).
cd "$(dirname "$0")/.."; until grep -q "VKITTI stage A complete" logs/vkitti.log; do sleep 120; done
echo "[refill $(date '+%F %T')] second pass" >> logs/vkitti.log; bash tools/run_vkitti.sh >> logs/vkitti.log 2>&1
