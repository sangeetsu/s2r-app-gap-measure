#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Foggy Cityscapes (ledger §15.59). mode=links: symlink trees b{beta}/leftImg8bit/val/<city>/<name>_leftImg8bit.png -> foggy file, so the
unchanged loader/evaluator/extractor work. mode=fid: image-level DINOv3 FID from every dose cell (80-cell lookup) to foggy val per beta."""
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
BETAS=["0.005","0.01","0.02"]; ROOT=Path("/mnt/features/cityscapes_foggy")
def links():
    src=ROOT/"leftImg8bit_foggy"/"val"; n=0
    for b in BETAS:
        for f in sorted(src.rglob(f"*_leftImg8bit_foggy_beta_{b}.png")):
            dst=ROOT/f"b{b}"/"leftImg8bit"/"val"/f.parent.name/f.name.replace(f"_foggy_beta_{b}",""); dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists(): dst.symlink_to(f); n+=1
        print(f"[foggy] beta {b}: {len(list((ROOT/f'b{b}'/'leftImg8bit'/'val').rglob('*.png')))} links")
def fid_mode(a):
    from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
    from s2rgap.data.carla import CARLADetection
    from domain_distance_matrix import fid, _cap
    from fid_to_real import img_feats
    lk=SpatialFeatureLookup.from_npz(a.features); out={}
    cells=sorted(d.name for d in Path(a.root).iterdir() if d.is_dir() and any(d.glob("frame_*.jpg")))
    F={}
    for b in BETAS:
        lb=SpatialFeatureLookup.from_npz(ROOT/f"spatial_features_dinov3_foggy_b{b}.npz")
        F[b]=np.array([np.array(lb.z_mmap[i],dtype=np.float32).reshape(-1,lb.z_mmap.shape[-1]).mean(0) for i in range(len(lb.z_mmap))],np.float32); print(f"[foggy] beta {b}: {len(F[b])} frames",flush=True)
    for c in cells:
        ds=CARLADetection(root=a.root,cells=[c]); C=img_feats(lk,[r[2] for r in ds.records])
        out[c]={f"fid_foggy_{b}":float(fid(_cap(F[b],1),_cap(C,2))) for b in BETAS}; print(f"[foggy] {c:<22} "+" ".join(f"b{b}={out[c][f'fid_foggy_{b}']:.2f}" for b in BETAS),flush=True)
    json.dump(out,open(a.out,"w"),indent=1); print("[foggy] saved",a.out)
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=["links","fid"]); p.add_argument("--root",default="/mnt/features/carla_doses"); p.add_argument("--features",default="/mnt/features/carla_doses/spatial_features_dinov3_80cells.npz"); p.add_argument("--out",default="results/calib_dose_foggy/fid_to_foggy.json")
    a=p.parse_args(); links() if a.mode=="links" else fid_mode(a)
