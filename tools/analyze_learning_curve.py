#!/usr/bin/env python3
"""Within-cell (causal) effect of labeled-data quantity on real-target mAP vs the between-town (correlational) slope (§15.38)."""
import json, glob, os, numpy as np
from pathlib import Path
from scipy import stats
def met(f): d=json.load(open(f)); return d.get("metrics",d)
def root(c):
    for r in (f"{Path.home()}/datasets/carla_extracted","/mnt/features/carla_new","/mnt/features/carla_new2"):
        if os.path.isdir(f"{r}/{c}"): return r
def nbox(c,k): return sum(1 for f in sorted(glob.glob(f"{root(c)}/{c}/frame_*.txt"))[:k] for l in open(f) if l.split() and l.split()[0] in ("Car","Pedestrian","Cyclist"))
L="results/calib_imnet_lc"; G="results/calib_imnet"; H="results/calib_imnet_towns2"
cells=["Town01_ClearNoon","Town02_ClearNoon","Town03_ClearNoon","Town04_ClearNoon","Town05_ClearNoon","Town10HD_ClearNoon"]
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    print(f"=== {T}: real mAP by training frames (100/200/400/800), fixed 2400 iterations; within-cell slope per log(boxes)")
    slopes=[]; X=[]; Y=[]; grp=[]
    for c in cells:
        pts=[]
        for k in (100,200,400,800):
            f=f"{G}/src_{c}_{key}.json" if k==800 else f"{L}/src_{c}_n{k}_{key}.json"
            if os.path.exists(f): pts.append((k,np.log(nbox(c,k)),met(f)["mAP@IoU=0.5"]*100))
        if len(pts)>=3:
            b=stats.linregress([p[1] for p in pts],[p[2] for p in pts]).slope; slopes.append(b); X+= [p[1] for p in pts]; Y+=[p[2] for p in pts]; grp+=[c]*len(pts)
        print(f"  {c:<20} "+"  ".join(f"n{k}: {m:5.1f} ({int(np.exp(lb))} boxes)" for k,lb,m in pts)+(f"  slope {slopes[-1]:+.2f}/log-box" if len(pts)>=3 else ""))
    if slopes:
        X,Y,grp=np.array(X),np.array(Y),np.array(grp); Xc=X-np.array([X[grp==g].mean() for g in grp]); Yc=Y-np.array([Y[grp==g].mean() for g in grp])
        fe=float((Xc*Yc).sum()/(Xc*Xc).sum()); print(f"  pooled within-cell (fixed-effects) slope {fe:+.2f} pp per log(boxes); per-cell {np.mean(slopes):+.2f} ± {np.std(slopes):.2f}")
        allc=sorted(f.split("src_")[1].replace(f"_{key}.json","") for f in glob.glob(f"{G}/src_*_{key}.json"))+sorted(f.split("src_")[1].replace(f"_{key}.json","") for f in glob.glob(f"{H}/src_*_{key}.json"))
        ya=[met(f"{G if os.path.exists(f'{G}/src_{c}_{key}.json') else H}/src_{c}_{key}.json")["mAP@IoU=0.5"]*100 for c in allc]; xa=[np.log(nbox(c,800)) for c in allc]
        bb=stats.linregress(xa,ya); print(f"  between-source slope (30 sources, 800 frames) {bb.slope:+.2f} pp per log(boxes) (r={bb.rvalue:+.3f})")
