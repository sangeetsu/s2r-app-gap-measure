#!/usr/bin/env python3
"""Score ledger §15.93."""
import json, glob, os, numpy as np
from scipy import stats
OUT="results/calib_vkitti_graded"; P=json.load(open(f"{OUT}/predictions.json")); ap=lambda f: json.load(open(f))["metrics"]["AP_Car"]*100
ev=sorted([f for f in glob.glob(f"{OUT}/eval_*.json")+glob.glob(f"{OUT}_rep/eval_*.json") if "_a25" in f or "_a75" in f],key=os.path.getmtime)
print(f"predictions {P['created']} (tau {P['tau']:.0f}); order {'OK' if os.path.getmtime(ev[0])>os.path.getmtime(OUT+'/predictions.json') else 'VIOLATED'}")
cells=sorted(P["cells"]); o=np.array([np.mean([ap(f"results/calib_vkitti{x}/eval_{P['cells'][c]['scene']}_d00_ref__on__{P['cells'][c]['scene']}_d00_ref.json")-ap(f"{OUT}{x}/eval_{P['cells'][c]['scene']}_d00_ref__on__{c}.json") for x in ("","_rep")]) for c in cells])
F=np.array([P["cells"][c]["formula"] for c in cells]); L=np.array([P["cells"][c]["line"] for c in cells]); eF=np.abs(F-o); eL=np.abs(L-o); w=stats.wilcoxon(eF,eL,alternative="less")
print(f"40 test pairs; observed {o.min():.1f}-{o.max():.1f}; predict-the-cal-mean MAE {np.abs(P['cal_mean']-o).mean():.2f}")
print(f"  formula MAE {eF.mean():.2f} rho {stats.spearmanr(F,o)[0]:+.3f} within ±4 {100*(eF<=4).mean():.0f}% | line MAE {eL.mean():.2f} rho {stats.spearmanr(L,o)[0]:+.3f} within ±4 {100*(eL<=4).mean():.0f}%")
print(f"VG1: Wilcoxon one-sided p={w.pvalue:.4f} -> {'PASS' if eF.mean()<eL.mean() and w.pvalue<.05 else 'FAIL'}")
rF=stats.spearmanr(F,o)[0]; print(f"VG2: formula rho {rF:+.3f} (>= .8) MAE {eF.mean():.2f} (<= 5.0) -> {'PASS' if rF>=.8 and eF.mean()<=5.0 else 'FAIL'}")
for v in ("morning","sunset","overcast","fog","rain"):
    i=np.array([P["cells"][c]["variant"]==v for c in cells]); print(f"  {v:<9} observed {o[i].mean():5.1f}  formula {F[i].mean():5.1f}  line {L[i].mean():5.1f}")
M=json.load(open(f"{OUT}/dose_metrics.json")); allx=[];ally=[]
for c,v in P["cells"].items(): allx.append(v["x"]); ally.append(o[cells.index(c)])
print(f"object FID vs drop over the 40 test pairs: rho {stats.spearmanr(allx,ally)[0]:+.3f}")
