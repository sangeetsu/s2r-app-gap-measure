#!/usr/bin/env python3
"""Anchored framework analysis: DROP = anchor mAP(S on own 800:1000) - mAP(S on T) vs predictors."""
import json, glob, sys, numpy as np
from scipy import stats
R="results/calib_anchored"; cells=sorted({f.split("eval_")[1].split("__on__")[0] for f in glob.glob(f"{R}/eval_*.json")})
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
anchor={s:m(f"{R}/eval_{s}__on__{s}.json") for s in cells if glob.glob(f"{R}/eval_{s}__on__{s}.json")}
cells=[c for c in cells if c in anchor]; print(f"{len(cells)} anchored sources; anchors: "+", ".join(f"{c.split('_')[0][-2:]}{c.split('_')[1][:3]}={anchor[c]:.0f}" for c in cells))
pm=sys.argv[1] if len(sys.argv)>1 else None; P=json.load(open(pm))["matrix"] if pm else None
pairs=[(S,T) for S in cells for T in cells if S!=T and glob.glob(f"{R}/eval_{S}__on__{T}.json")]
drop=np.array([anchor[S]-m(f"{R}/eval_{S}__on__{T}.json") for S,T in pairs])
print(f"\n{len(pairs)} directional pairs; drop mean {drop.mean():.1f} sd {drop.std():.1f}")
def rep(name,x):
    x=np.array(x,float); ok=~np.isnan(x); r,p=stats.pearsonr(x[ok],drop[ok]); rho,ps=stats.spearmanr(x[ok],drop[ok])
    errs=[]
    for h in cells:
        tr=np.array([h not in pr for pr in pairs])&ok; te=np.array([h in pr for pr in pairs])&ok
        if tr.sum()<3 or te.sum()==0: continue
        sl,ic,*_=stats.linregress(x[tr],drop[tr]); errs+=list(np.abs(drop[te]-(ic+sl*x[te])))
    errs=np.array(errs); print(f"  {name:<28} r={r:+.3f} (p={p:.1e})  rho={rho:+.3f}  LOO|err| {errs.mean():.2f}  <=4pp {100*(errs<=4).mean():.0f}%")
if P: rep("dIV_fg (probe, anchored)", [P[S][T]["dIV_fg"] if S in P and T in P[S] else np.nan for S,T in pairs])
D=json.load(open("results/domain_distance_dinov3.json"))["metrics"]["image"]
rep("FID image", [ (D.get(f"{S}||{T}") or D.get(f"{T}||{S}") or {"fid":np.nan})["fid"] for S,T in pairs])
for Tgt,key in [("KITTI","kittival"),("Cityscapes","cityscapesval")]:
    fs={s:m(f"{R}/src_{s}_{key}.json") for s in cells if glob.glob(f"{R}/src_{s}_{key}.json")}
    if len(fs)<6: continue
    y=np.array([anchor[s]-fs[s] for s in fs]); print(f"\n{Tgt}: n={len(fs)} sources, anchored drop vs predictor")
    if P: x=np.array([P[s][Tgt]["dIV_fg"] for s in fs]); r,p=stats.pearsonr(x,y); rho,ps=stats.spearmanr(x,y); print(f"  dIV_fg   r={r:+.3f} (p={p:.3f}) rho={rho:+.3f} (p={ps:.3f})")
    x=np.array([(D.get(f"{s}||{Tgt}") or D.get(f"{Tgt}||{s}") or {"fid":np.nan})["fid"] for s in fs]); ok=~np.isnan(x)
    if ok.sum()>5: r,p=stats.pearsonr(x[ok],y[ok]); print(f"  FID      r={r:+.3f} (p={p:.3f})")
