#!/usr/bin/env python3
"""Score ledger §15.92: X1 F-own vs L-own (Wilcoxon, pooled 60 pairs), X2 F-shared (tau 239.3 unchanged) rho >= .8 and MAE <= 5.0."""
import json, glob, os, numpy as np
from scipy import stats
OUT="results/calib_formula_families"; P=json.load(open(f"{OUT}/predictions.json")); m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
ev=sorted(glob.glob(f"{OUT}/eval_*.json"),key=os.path.getmtime); print(f"predictions {P['created']}; order {'OK' if os.path.getmtime(ev[0])>os.path.getmtime(OUT+'/predictions.json') else 'VIOLATED'}; {len(ev)} evals")
allo=[]; allp={k:[] for k in ("F_own","F_shared","L_own")}
for f,F in P["families"].items():
    cells=sorted(F["cells"]); o=np.array([F["A"][c.split("_d")[0]]-m(f"{OUT}/eval_{f}_{c}.json") for c in cells]); allo+=list(o)
    print(f"\n=== {f}: tau_own {F['tau_own']:.1f} vs shared {P['tau_shared']:.1f}; 30 fresh pairs; observed {o.min():.1f}-{o.max():.1f}; predict-the-cal-mean MAE {np.abs(F['cal_mean']-o).mean():.2f}")
    for k in ("F_own","F_shared","L_own"):
        pr=np.array([F["cells"][c][k] for c in cells]); allp[k]+=list(pr); e=np.abs(pr-o)
        print(f"  {k:<9} MAE {e.mean():5.2f}  rho {stats.spearmanr(pr,o)[0]:+.3f}  within ±4 pp {100*(e<=4).mean():3.0f}%")
o=np.array(allo); e={k:np.abs(np.array(v)-o) for k,v in allp.items()}; w=stats.wilcoxon(e["F_own"],e["L_own"],alternative="less"); rs=stats.spearmanr(allp["F_shared"],o)[0]
print(f"\nPOOLED 60 pairs: F_own MAE {e['F_own'].mean():.2f} | F_shared {e['F_shared'].mean():.2f} (rho {rs:+.3f}, within ±4 pp {100*(e['F_shared']<=4).mean():.0f}%) | L_own {e['L_own'].mean():.2f}")
print(f"X1: F_own vs L_own Wilcoxon one-sided p={w.pvalue:.4f} -> {'PASS' if e['F_own'].mean()<e['L_own'].mean() and w.pvalue<.05 else 'FAIL'}")
print(f"X2: F_shared rho {rs:+.3f} (>= .8) and MAE {e['F_shared'].mean():.2f} (<= 5.0) -> {'PASS' if rs>=.8 and e['F_shared'].mean()<=5.0 else 'FAIL'}")
