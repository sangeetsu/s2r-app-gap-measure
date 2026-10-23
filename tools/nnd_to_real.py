#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Candidate B (ledger §15.44-15.45): directed object-level distance from real targets to a source's object manifold, frozen DINOv3.
nnd_T_to_S = median_{z in T} min_{s in S} ||z - s|| / median_{s in S} r_k(s), features standardised with S statistics, k=3, cap 3000.
Fixed seeds per domain so values are reproducible and independent of processing order."""
from __future__ import annotations
import argparse, json, os, sys, zlib
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection
from latent_geometry import obj_feats, knn_radius
def seed_of(name): return zlib.crc32(name.encode()) & 0xffffffff
def nnd(Xs,Xt,dev):
    mu,sd=Xs.mean(0),Xs.std(0)+1e-6; S=torch.from_numpy((Xs-mu)/sd).to(dev); T=torch.from_numpy((Xt-mu)/sd).to(dev)
    rS=knn_radius(S); return float((torch.cdist(T,S).min(1).values.median()/rS.median()).item())
def main():
    p=argparse.ArgumentParser(); p.add_argument("--sources",nargs="+",required=True,help="root:features_npz:cell"); p.add_argument("--out",required=True)
    a=p.parse_args(); dev=torch.device("cuda"); out=json.load(open(a.out)) if os.path.exists(a.out) else {}
    lk=SpatialFeatureLookup.from_npz(Path.home()/"datasets/kitti/spatial_features_dinov3.npz")
    kv=KITTIDetection(kitti_root=str(Path.home()/"datasets/kitti/extracted/training"),split_json=str(Path.home()/"datasets/kitti/splits.json"),split="val")
    tg={"KITTI":obj_feats(lk,kv,range(0,len(kv),max(1,len(kv)//1000)),np.random.default_rng(seed_of("KITTI")))}
    from s2rgap.data.cityscapes_coco import CityscapesCOCO
    lc=SpatialFeatureLookup.from_npz(Path.home()/"datasets/cityscapes/spatial_features_dinov3_val.npz")
    cs=CityscapesCOCO(Path.home()/"datasets/cityscapes/extracted/leftImg8bit",REPO/"nvme/datasets/aldi_labels/cityscapes_val_instances.json",split="val")
    tg["Cityscapes"]=obj_feats(lc,cs,range(len(cs)),np.random.default_rng(seed_of("Cityscapes"))); print(f"[nnd] targets KITTI {len(tg['KITTI'])} CS {len(tg['Cityscapes'])}",flush=True)
    lookups={}
    for spec in a.sources:
        root,feat,cell=spec.split(":")
        if cell in out: continue
        lookups.setdefault(feat,SpatialFeatureLookup.from_npz(feat)); ds=CARLADetection(root=root,cells=[cell])
        Xs=obj_feats(lookups[feat],ds,range(0,min(800,len(ds))),np.random.default_rng(seed_of(cell)))
        out[cell]={T:nnd(Xs,Xt,dev) for T,Xt in tg.items()}; out[cell]["n_src"]=int(len(Xs)); json.dump(out,open(a.out,"w"),indent=1)
        print(f"[nnd] {cell:<24} KITTI {out[cell]['KITTI']:.3f} CS {out[cell]['Cityscapes']:.3f} (n_src {len(Xs)})",flush=True)
if __name__=="__main__": raise SystemExit(main())
