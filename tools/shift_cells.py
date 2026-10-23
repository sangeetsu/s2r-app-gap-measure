#!/usr/bin/env python3
"""Build SHIFT replication cells (ledger §15.48) from tools/shift_convert.py output: one cell per (timeofday_coarse x weather_coarse),
videos shuffled with a fixed seed, held-out = whole videos until >=150 frames, train = 400 frames sampled from the remaining videos.
Cell frames 0:400 = train, 400:550 = held-out (no video in both). Images symlinked, labels copied, manifest.jsonl written."""
import csv, json, random, shutil, sys
from pathlib import Path
SRC=Path("/mnt/features/shift_conv/val"); OUT=Path("/mnt/features/shift_cells"); NTR, NHO = 400, 150
R=list(csv.DictReader(open(SRC/"index.csv"))); groups={}
for r in R: groups.setdefault((r["timeofday_coarse"].replace("/","-"), r["weather_coarse"]), []).append(r["video"])
meta={}
for (tod,w),vids in sorted(groups.items()):
    cell=f"SHIFT_{tod}_{w}"; rng=random.Random(f"{cell}-42"); vids=sorted(vids); rng.shuffle(vids)
    frames=lambda v: sorted((SRC/"videos"/v).glob("frame_*.jpg"))
    ho,tr=[],[]
    for v in vids:
        (ho if len(ho)<NHO else tr).extend([(v,f) for f in frames(v)])
    ho=ho[:NHO]; rng.shuffle(tr); tr=sorted(tr[:NTR])
    if len(tr)<NTR: print(f"skip {cell}: only {len(tr)} train frames"); continue
    d=OUT/cell; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    with open(d/"manifest.jsonl","w") as man:
        for k,(v,f) in enumerate(tr+ho):
            (d/f"frame_{k:06d}.jpg").symlink_to(f); shutil.copy(f.with_suffix(".txt"), d/f"frame_{k:06d}.txt")
            man.write(json.dumps({"frame":f"frame_{k:06d}.jpg","label":f"frame_{k:06d}.txt","video":v,"src":f.name,"split":"train" if k<NTR else "heldout"})+"\n")
    nb=sum(1 for k in range(NTR) for l in open(d/f"frame_{k:06d}.txt") if l.strip())
    meta[cell]={"timeofday":tod,"weather":w,"train_videos":len({v for v,_ in tr}),"heldout_videos":len({v for v,_ in ho}),"train_boxes":nb}
    print(f"{cell:<32} train {len(tr)} frames / {meta[cell]['train_videos']} videos, {nb} boxes | held-out {len(ho)} frames / {meta[cell]['heldout_videos']} videos")
json.dump(meta,open(OUT/"cells_meta.json","w"),indent=1)
