#!/usr/bin/env python3
"""Score ledger §15.87: F1 M2 vs M0 (Wilcoxon), F2 M2 rho/MAE, F3 within-dose rank; M1 reported."""
import json, glob, os, numpy as np
from scipy import stats
H3="results/calib_dose_ho3"; P=json.load(open(f"{H3}/predictions.json")); m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
ev=sorted(glob.glob(f"{H3}/eval_*.json")+glob.glob(f"{H3}_rep/eval_*.json"),key=os.path.getmtime)
print(f"predictions {P['created']} (tau {P['M2']['tau']:.1f}); order {'OK' if os.path.getmtime(ev[0])>os.path.getmtime(H3+'/predictions.json') else 'VIOLATED'}")
cells=sorted(P["cells"]); o=[]
for c in cells:
    t=P["cells"][c]["town"]; o.append(np.mean([m(f"results/calib_dose/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{H3}/eval_{t}_d00_ref__on__{c}.json"),m(f"results/calib_dose_rep/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{H3}_rep/eval_{t}_d00_ref__on__{c}.json")]))
o=np.array(o); d=np.array([P["cells"][c]["dose"] for c in cells]); pr={k:np.array([P["cells"][c][k] for c in cells]) for k in ("M0","M1","M2")}; e={k:np.abs(v-o) for k,v in pr.items()}
w=stats.wilcoxon(e["M2"],e["M0"],alternative="less"); mae0=np.abs(P["fit_mean"]-o).mean()
print(f"{len(o)} pairs; observed drop range {o.min():.1f}-{o.max():.1f}, sd {o.std():.1f}; predict-the-fit-mean MAE {mae0:.2f}")
for k in ("M0","M1","M2"):
    wd=np.nanmean([stats.spearmanr(pr[k][d==x],o[d==x])[0] for x in sorted(set(d))])
    print(f"  {k}: MAE {e[k].mean():5.2f}  pair rho {stats.spearmanr(pr[k],o)[0]:+.3f}  within ±4 pp {100*(e[k]<=4).mean():3.0f}%  mean within-dose rho {wd:+.2f}")
r2=stats.spearmanr(pr["M2"],o)[0]
print(f"F1: M2 MAE {e['M2'].mean():.2f} vs M0 {e['M0'].mean():.2f}, Wilcoxon one-sided p={w.pvalue:.4f} -> {'PASS' if e['M2'].mean()<e['M0'].mean() and w.pvalue<.05 else 'FAIL'}")
print(f"F2: M2 rho {r2:+.3f} (>= .8) and MAE {e['M2'].mean():.2f} (<= 5.0) -> {'PASS' if r2>=.8 and e['M2'].mean()<=5.0 else 'FAIL'}")
for x in sorted(set(d)):
    i=d==x; print(f"  {x:<16} observed {o[i].mean():5.1f}  M0 {pr['M0'][i].mean():5.1f}  M1 {pr['M1'][i].mean():5.1f}  M2 {pr['M2'][i].mean():5.1f}")
