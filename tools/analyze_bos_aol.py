#!/usr/bin/env python3
"""Published label-free references (BoS ICLR'24 box stability; Agreement-on-the-Line NeurIPS'22) vs anchored drops,
raw and anchored (in-domain minus target), alongside the FM veto gap and labeled dIV_fg."""
import json, glob, sys, numpy as np
from scipy import stats
R="results/calib_anchored"; F=sys.argv[1] if len(sys.argv)>1 else f"{R}/bos_aol_8cells.json"
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
cells=sorted({f.split("eval_")[1].split("__on__")[0] for f in glob.glob(f"{R}/eval_*.json")}); anchor={s:m(f"{R}/eval_{s}__on__{s}.json") for s in cells if glob.glob(f"{R}/eval_{s}__on__{s}.json")}
cells=[c for c in cells if c in anchor]; B=json.load(open(F)); srcs=[s for s in cells if s in B]
E=json.load(open(f"{R}/detector_box_gap_ens.json")) if glob.glob(f"{R}/detector_box_gap_ens.json") else None
mats=[json.load(open(f))["matrix"] for f in sorted(glob.glob(f"{R}/probe_matrix_anchored_s*.json"))]
keys=[k for k in next(v for v in B[srcs[0]].values() if isinstance(v,dict)) if isinstance(B[srcs[0]][next(iter(B[srcs[0]]))].get(k,0.0),float)]
def rep(name,x,y):
    x,y=np.array(x,float),np.array(y,float); ok=~np.isnan(x)&~np.isnan(y)
    if ok.sum()<4: return
    r,p=stats.pearsonr(x[ok],y[ok]); rho,_=stats.spearmanr(x[ok],y[ok]); print(f"  {name:<44} r={r:+.3f} (p={p:.1e}) rho={rho:+.3f} n={ok.sum()}")
def block(P,y,title):
    print(f"=== {title}: {len(P)} pairs ===")
    if mats: rep("dIV_fg (LABELED)",[np.mean([M[S][T]["dIV_fg"] for M in mats if S in M and T in M[S]]) for S,T in P],y)
    if E: rep("soft FM veto gap (ens, tau.7, label-free)",[E[S][T]["0.7"]["probe_pbg"]-E[S]["in@0.7"]["probe_pbg"] if S in E and T in E[S] else np.nan for S,T in P],y)
    for k in keys:
        rep(f"{k} raw (target value)",[B[S][T][k] if T in B[S] else np.nan for S,T in P],y)
        rep(f"{k} anchored (in-domain minus target)",[B[S][S][k]-B[S][T][k] if (S in B[S] and T in B[S]) else np.nan for S,T in P],y)
P=[(S,T) for S in srcs for T in cells if S!=T and glob.glob(f"{R}/eval_{S}__on__{T}.json")]; block(P,[anchor[S]-m(f"{R}/eval_{S}__on__{T}.json") for S,T in P],"CARLA pairs, anchored drop")
for Tgt,key in [("KITTI","kittival"),("Cityscapes","cityscapesval")]:
    fs=[s for s in srcs if glob.glob(f"{R}/src_{s}_{key}.json") and Tgt in B[s]]
    if len(fs)>=6: block([(s,Tgt) for s in fs],[anchor[s]-m(f"{R}/src_{s}_{key}.json") for s in fs],f"{Tgt} (n={len(fs)} sources), anchored drop")
