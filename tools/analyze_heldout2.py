#!/usr/bin/env python3
"""Score ledger §15.76: B1 isotonic vs linear (Wilcoxon on |errors|), B2 isotonic rank/MAE, B3 share within ±4 pp."""
import json, glob, os, numpy as np
from scipy import stats
H2="results/calib_dose_ho2"; P=json.load(open(f"{H2}/predictions.json")); m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
ev=sorted(glob.glob(f"{H2}/eval_*.json")+glob.glob(f"{H2}_rep/eval_*.json"),key=os.path.getmtime)
print(f"predictions {P['created']}; order {'OK' if os.path.getmtime(ev[0])>os.path.getmtime(H2+'/predictions.json') else 'VIOLATED'}")
cells=sorted(P["cells"]); o=[]
for c in cells:
    t=P["cells"][c]["town"]; o.append(np.mean([m(f"results/calib_dose/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{H2}/eval_{t}_d00_ref__on__{c}.json"),m(f"results/calib_dose_rep/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{H2}_rep/eval_{t}_d00_ref__on__{c}.json")]))
o=np.array(o); L=np.array([P["cells"][c]["pred_linear"] for c in cells]); I=np.array([P["cells"][c]["pred_isotonic"] for c in cells]); eL=np.abs(L-o); eI=np.abs(I-o)
w=stats.wilcoxon(eI,eL,alternative="less"); mae0=np.abs(P["cal_mean"]-o).mean()
print(f"{len(o)} pairs. B1: MAE isotonic {eI.mean():.2f} vs linear {eL.mean():.2f}; Wilcoxon one-sided p={w.pvalue:.4f} -> {'PASS' if eI.mean()<eL.mean() and w.pvalue<.05 else 'FAIL'}")
rI=stats.spearmanr(I,o)[0]; rL=stats.spearmanr(L,o)[0]
print(f"B2: isotonic rho {rI:+.3f}, MAE {eI.mean():.2f} (<= 7.0 and < mean-pred {mae0:.2f}) -> {'PASS' if rI>=.7 and eI.mean()<=7 and eI.mean()<mae0 else 'FAIL'}; linear rho {rL:+.3f}")
print(f"B3: within ±4 pp: isotonic {100*(eI<=4).mean():.0f}%, linear {100*(eL<=4).mean():.0f}% (seed-noise ceiling ≈ 88%)")
for d in sorted({P['cells'][c]['dose'] for c in cells}):
    i=np.array([P["cells"][c]["dose"]==d for c in cells]); print(f"  {d:<16} observed {o[i].mean():5.1f}  linear {L[i].mean():5.1f}  isotonic {I[i].mean():5.1f}")
