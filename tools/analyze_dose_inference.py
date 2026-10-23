#!/usr/bin/env python3
"""Reviewer item 2: town-clustered permutation p for R2 (metric vs appearance-only drop) and calibration-vs-number-of-towns curves.
Usage: analyze_dose_inference.py [metrics.json ...] (any number of metric files; keys merged)."""
import json, glob, os, sys, numpy as np
from scipy import stats
R=os.environ.get("DOSE_DIR","results/calib_dose"); files=sys.argv[1:] or [f"{R}/dose_metrics.json"]
M={}
for f in files:
    for c,v in json.load(open(f)).items(): M.setdefault(c,{}).update({k:x for k,x in v.items() if isinstance(x,(int,float))})
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
T=sorted({c.split("_d")[0] for c in M}); D=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
pairs=[(t,d) for t in T for d in D if f"{t}_{d}" in M]; y=np.array([m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{R}/eval_{t}_d00_ref__on__{t}_{d}.json") for t,d in pairs]); tw=np.array([t for t,_ in pairs])
skip={"n_obj","n_boxes","n_paired"}; keys=[k for k in M[f"{pairs[0][0]}_{pairs[0][1]}"] if k not in skip and not k.startswith("fid_") and not k.startswith("incfid_K") and not k.startswith("incfid_C")]
rng=np.random.default_rng(0)
print(f"{len(pairs)} pairs, {len(T)} towns. Clustered permutation: metric values shuffled within town (2000 draws).")
print(f"  {'metric':<20}{'Spearman':>10}{'clustered p':>13}   LOTO-k abs error (pp) for k train towns = 1,2,4,7")
for k in keys:
    x=np.array([M[f"{t}_{d}"][k] for t,d in pairs],float); ok=~np.isnan(x)
    rho=stats.spearmanr(x[ok],y[ok])[0]; null=[]
    for _ in range(2000):
        xp=x.copy()
        for t in T: idx=np.where(tw==t)[0]; xp[idx]=rng.permutation(x[idx])
        null.append(stats.spearmanr(xp[ok],y[ok])[0])
    p=(1+np.sum(np.abs(null)>=abs(rho)))/2001
    errs={}
    for kk in (1,2,4,7):
        if kk>=len(T): continue
        e=[]
        for _ in range(200):
            tr=set(rng.choice(T,kk,replace=False)); itr=np.array([t in tr for t in tw])&ok; ite=(~np.array([t in tr for t in tw]))&ok
            if itr.sum()<3: continue
            b=np.polyfit(x[itr],y[itr],1); e+=list(np.abs(y[ite]-np.polyval(b,x[ite])))
        errs[kk]=np.mean(e)
    print(f"  {k:<20}{rho:+10.3f}{p:13.4f}   "+"  ".join(f"k={kk}: {v:4.1f}" for kk,v in errs.items())+f"   (predict-the-mean: {np.abs(y-y.mean()).mean():.1f})")
