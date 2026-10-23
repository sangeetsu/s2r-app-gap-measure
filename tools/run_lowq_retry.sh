#!/usr/bin/env bash
# After run_lowq.sh exits: regenerate the Low cells that hit CARLA's 10 s client timeout (Town02/03/05), with the 180 s profile and a fresh
# server per cell, then rerun the (idempotent) chain so training/evals/features/FID/analysis cover all 6 towns.
cd "$(dirname "$0")/.."
while pgrep -u $USER -f "^bash tools/run_low[q].sh" >/dev/null; do sleep 60; done
sed -i 's/--server-profile local_package --towns/--server-profile local_package_largemap --towns/' tools/run_lowq.sh
sed -i 's/^  for attempt in 1 2; do \[ "\$(nf \$c)" -ge 1000 \] \&\& break; log "GEN \$c low quality (attempt \$attempt)"; t0=\$(date +%s)$/  for attempt in 1 2; do [ "$(nf $c)" -ge 1000 ] \&\& break; start_carla; log "GEN $c low quality (attempt $attempt)"; t0=$(date +%s)/' tools/run_lowq.sh
grep -c "largemap" tools/run_lowq.sh >> logs/lowq.log; grep -c "start_carla; log \"GEN" tools/run_lowq.sh >> logs/lowq.log
for c in Town02_ClearNoon Town03_ClearNoon Town05_ClearNoon; do n=$(ls /mnt/features/carla_lowq/$c/frame_*.jpg 2>/dev/null | wc -l); [ "$n" -lt 1000 ] && rm -rf /mnt/features/carla_lowq/$c; done
rm -f results/calib_imnet_lowq/fid_to_real_low.json
echo "[lowq-retry $(date '+%F %T')] relaunching chain for missing cells" >> logs/lowq.log
bash tools/run_lowq.sh >> logs/lowq.log 2>&1
