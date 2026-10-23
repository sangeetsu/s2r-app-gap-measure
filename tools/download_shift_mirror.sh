#!/usr/bin/env bash
# SHIFT discrete/videos front-camera subset from the third-party HF mirror KevinConnorLee/SHIFT (official dl.cv.ethz.ch retired: NXDOMAIN).
# Untrusted data: downloaded into its own directory; nothing in it is executed. Resumable (curl -C -).
set -u; D=/mnt/features/shift_hf; B="https://huggingface.co/datasets/KevinConnorLee/SHIFT/resolve/main"; cd $D
for f in discrete/videos/val/front/det_2d.json discrete/videos/val/front/img.tar discrete/videos/train/front/det_2d.json discrete/videos/train/front/img.tar; do
  mkdir -p "$(dirname $f)"; echo "[shift $(date '+%F %T')] GET $f"
  for try in 1 2 3; do nice -n 10 curl -sSL --retry 5 --retry-delay 10 -C - -o "$f" "$B/$f" && break; echo "  retry $try"; sleep 30; done
  echo "[shift $(date '+%F %T')]   $(stat -c %s "$f") bytes"; done
echo "[shift $(date '+%F %T')] === download complete ==="
