#!/usr/bin/env python3
"""Run-to-run noise floor of the anchored drop: same cell, same data, different training seed (2026-10-14).
Compares results/calib_imnet (seed 42) with results/calib_imnet_rep (seed 17) for the replicated sources."""
import json, glob, numpy as np
A,B="results/calib_imnet","results/calib_imnet_rep"
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
reps=sorted({f.split("eval_")[1].split("__on__")[0] for f in glob.glob(f"{B}/eval_*.json")})
dd,dm=[],[]
for s in reps:
    a0,b0=m(f"{A}/eval_{s}__on__{s}.json"),m(f"{B}/eval_{s}__on__{s}.json")
    for f in glob.glob(f"{B}/eval_{s}__on__*.json"):
        t=f.split("__on__")[1][:-5]
        if t==s or not glob.glob(f"{A}/eval_{s}__on__{t}.json"): continue
        ma,mb=m(f"{A}/eval_{s}__on__{t}.json"),m(f)
        dm.append(mb-ma); dd.append((b0-mb)-(a0-ma))
    for key in ("kittival","cityscapesval"):
        if glob.glob(f"{B}/src_{s}_{key}.json"): print(f"  {s:<22} {key:<14} seed42 {m(f'{A}/src_{s}_{key}.json'):5.1f}  seed17 {m(f'{B}/src_{s}_{key}.json'):5.1f}")
    print(f"  {s:<22} anchor (own 800:1000) seed42 {a0:5.1f}  seed17 {b0:5.1f}")
dd,dm=np.array(dd),np.array(dm)
print(f"\n{len(reps)} replicated sources, {len(dd)} (S,T) pairs")
print(f"  target mAP seed-to-seed diff: sd {dm.std():.2f} pp, mean |diff| {np.abs(dm).mean():.2f} pp")
print(f"  anchored DROP seed-to-seed diff: sd {dd.std():.2f} pp, mean |diff| {np.abs(dd).mean():.2f} pp, within 4 pp {100*(np.abs(dd)<=4).mean():.0f}%")
print(f"  -> a perfect predictor of the seed-42 drop is within 4 pp of a re-trained detector's drop only ~{100*(np.abs(dd)<=4).mean():.0f}% of the time")
