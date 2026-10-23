#!/usr/bin/env python3
"""Ledger §15.92 descriptive, post hoc: the calibration formula on Virtual KITTI 2 weather variants (Car AP; 4 reliable scenes x 5 variants), vs a straight line;
leave-one-scene-out and leave-one-variant-out."""
import json, numpy as np
from scipy import optimize, stats
R="results/calib_vkitti"; S=["Scene01","Scene02","Scene18","Scene20"]; V=["d20_morning","d21_sunset","d22_overcast","d23_fog","d24_rain"]
ap=lambda f: json.load(open(f))["metrics"]["AP_Car"]*100; M=json.load(open(f"{R}/dose_metrics.json"))
A={s:np.mean([ap(f"{Rr}/eval_{s}_d00_ref__on__{s}_d00_ref.json") for Rr in (R,R+"_rep")]) for s in S}
P=[(s,v) for s in S for v in V]; x=np.array([M[f"{s}_{v}"]["obj_fid_ref"] for s,v in P]); y=np.array([np.mean([ap(f"{Rr}/eval_{s}_d00_ref__on__{s}_d00_ref.json")-ap(f"{Rr}/eval_{s}_d00_ref__on__{s}_{v}.json") for Rr in (R,R+"_rep")]) for s,v in P])
Av=np.array([A[s] for s,_ in P]); gs=np.array([s for s,_ in P]); gv=np.array([v for _,v in P])
def fitF(i): return optimize.minimize_scalar(lambda t: np.sum((y[i]-Av[i]*(1-np.exp(-x[i]/t)))**2),bounds=(10,1e5),method="bounded").x
for name,groups,G in (("leave-one-scene-out",S,gs),("leave-one-variant-out",V,gv)):
    pf=np.zeros_like(y); pl=np.zeros_like(y)
    for g in groups:
        te=G==g; tr=~te; t=fitF(tr); pf[te]=Av[te]*(1-np.exp(-x[te]/t)); b,a=np.polyfit(x[tr],y[tr],1); pl[te]=a+b*x[te]
    print(f"{name}: formula MAE {np.abs(pf-y).mean():5.2f} rho {stats.spearmanr(pf,y)[0]:+.2f} | line MAE {np.abs(pl-y).mean():5.2f} rho {stats.spearmanr(pl,y)[0]:+.2f} | mean-pred MAE {np.abs(y-y.mean()).mean():.2f}")
print(f"tau fitted on all 20 pairs: {fitF(np.ones(len(y),bool)):.0f} (VKITTI2's own object-FID scale)")
