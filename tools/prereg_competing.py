#!/usr/bin/env python3
"""Competing pre-registered models for held-out cells (2026-10-14): mAP ~ log(training boxes) and mAP ~ FID + log(boxes), fit on the
24 ImageNet-grid sources; FID-only predictions come from the fid_to_real --predict file. Usage: prereg_competing.py PRED_IN OUT"""
import json, glob, os, sys, time, hashlib, numpy as np
from pathlib import Path
A="results/calib_imnet"; PIN,POUT=sys.argv[1],sys.argv[2]
def met(f): d=json.load(open(f)); return d.get("metrics",d)
src=sorted(f.split("src_")[1].replace("_kittival.json","") for f in glob.glob(f"{A}/src_*_kittival.json") if "Town06" not in f and "Town07" not in f and "Town1" not in f or f.split("src_")[1].startswith("Town10HD"))
src=[s for s in src if s.split("_")[0] in ("Town01","Town02","Town03","Town04","Town05","Town10HD")]; assert len(src)==24, len(src)
def root(c):
    for r in (f"{Path.home()}/datasets/carla_extracted","/mnt/features/carla_new","/mnt/features/carla_new2","/mnt/features/carla_new3"):
        if os.path.isdir(f"{r}/{c}"): return r
def nbox(c): return sum(1 for f in sorted(glob.glob(f"{root(c)}/{c}/frame_*.txt"))[:800] for l in open(f) if l.split() and l.split()[0] in ("Car","Pedestrian","Cyclist"))
D=json.load(open("results/domain_distance_dinov3_all.json"))["metrics"]["image"]; Q=json.load(open(PIN))
out={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"from":PIN,"models":{},"cells":{}}
held={c:{"lb":float(np.log(max(1,nbox(c)))),"fid":{T:Q["cells"][c][T]["fid"] for T in ("KITTI","Cityscapes")}} for c in Q["cells"]}
lb=np.array([np.log(nbox(s)) for s in src])
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    y=np.array([met(f"{A}/src_{s}_{key}.json")["mAP@IoU=0.5"]*100 for s in src]); x=np.array([(D.get(f"{s}||{T}") or D.get(f"{T}||{s}"))["fid"] for s in src])
    for name,X,hx in [("boxes",np.column_stack([np.ones(24),lb]),lambda c:[1,held[c]["lb"]]),("fid+boxes",np.column_stack([np.ones(24),x,lb]),lambda c:[1,held[c]["fid"][T],held[c]["lb"]])]:
        b=np.linalg.lstsq(X,y,rcond=None)[0]; out["models"][f"{T}:{name}"]={"coef":b.tolist(),"resid_sd":float(np.std(y-X@b,ddof=X.shape[1]))}
        for c in held: out["cells"].setdefault(c,{"n_boxes":int(round(np.exp(held[c]["lb"])))})[f"{T}:{name}"]=float(np.dot(b,hx(c)))
    for c in held: out["cells"][c][f"{T}:fid"]=Q["cells"][c][T]["pred_mAP"]
Path(POUT).write_text(json.dumps(out,indent=1)); print("saved",POUT,"sha256",hashlib.sha256(open(POUT,'rb').read()).hexdigest()[:16])
