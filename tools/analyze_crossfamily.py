#!/usr/bin/env python3
"""Score ledger §15.79."""
import json, glob, os, numpy as np
from scipy import stats
OUT="results/calib_crossfamily"; P=json.load(open(f"{OUT}/predictions.json")); m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
FAM={"fcos":"results/calib_dose_arch2","swin":"results/calib_dose_arch3"}
ev=sorted(glob.glob(f"{OUT}/eval_*.json"),key=os.path.getmtime); print(f"predictions {P['created']}; order {'OK' if os.path.getmtime(ev[0])>os.path.getmtime(OUT+'/predictions.json') else 'VIOLATED'}")
for f,R in FAM.items():
    F=P["families"][f]; cells=sorted(F["cells"]); o=np.array([m(f"{R}/eval_{c.split('_d')[0]}_d00_ref__on__{c.split('_d')[0]}_d00_ref.json")-m(f"{OUT}/eval_{f}_{c}.json") for c in cells])
    print(f"\n=== {f}: 18 held-out pairs (Town01 + Town10HD x 9 round-1 doses); predict-the-mean MAE {np.abs(F['cal_mean']-o).mean():.2f}")
    for k in ("pred_linear","pred_isotonic","pred_frcnn_line"):
        pr=np.array([F["cells"][c][k] for c in cells]); rho=stats.spearmanr(pr,o)[0]; mae=np.abs(pr-o).mean()
        print(f"  {k:<16} rho {rho:+.3f}  MAE {mae:5.2f}  within 4 pp {100*(np.abs(pr-o)<=4).mean():3.0f}%"+(f"  -> {'PASS' if rho>=.7 and mae<=7 and mae<np.abs(F['cal_mean']-o).mean() else 'FAIL'}" if k=="pred_linear" else ""))
    for c,ob in zip(cells,o): print(f"    {c:<22} obs {ob:5.1f}  lin {F['cells'][c]['pred_linear']:5.1f}")
