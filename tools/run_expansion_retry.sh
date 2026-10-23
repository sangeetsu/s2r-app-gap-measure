#!/usr/bin/env bash
# Retry pass for transient CARLA RPC timeouts (Town04_OvercastNoon 2026-10-13 16:01): waits for the first expansion
# pass, removes the one-off smoke-cell cleanup from run_carla_expansion.sh, reruns it (skips complete cells),
# then writes the marker run_after_expansion.sh waits for.
set -u; cd "$(dirname "$0")/.."; exec 9>/tmp/.expretry.lock; flock -n 9 || exit 0
until grep -q "=== expansion complete ===" logs/carla_expansion.log 2>/dev/null; do sleep 120; done
sed -i '/rm -rf \$OUT\/Town02_ClearNoon/d' tools/run_carla_expansion.sh
for pass in 1 2; do
  missing=$(for t in Town02 Town04 Town05 Town10HD; do for w in ClearNoon OvercastNoon SoftRainNoon ClearNight; do
    [ "$(ls /mnt/features/carla_new/${t}_${w}/frame_*.jpg 2>/dev/null | wc -l)" -ge 1000 ] || echo ${t}_${w}; done; done)
  [ -z "$missing" ] && break
  echo "[retry $(date '+%F %T')] pass $pass, missing: $missing" >> logs/carla_expansion.log
  pgrep -f "CarlaUE4-Linux-[S]hipping" >/dev/null || { echo "[retry] CARLA server down; starting" >> logs/carla_expansion.log;
    (cd ~/CARLA/CARLA_0.9.16 && setsid nohup ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 > /dev/null 2>&1 < /dev/null &); sleep 60; }
  sleep 30; bash tools/run_carla_expansion.sh >> logs/carla_expansion.log 2>&1
done
echo "=== expansion retry complete ===" >> logs/carla_expansion.log
