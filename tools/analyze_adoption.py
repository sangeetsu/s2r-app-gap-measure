#!/usr/bin/env python3
"""Post hoc, descriptive (ledger §15.80): (a) how many calibration doses are needed — calibrate object FID -> drop on k of the 18 doses observed so far,
predict the other doses (all 8 towns), MAE vs k; (b) metric-guided choice of training appearance — per town and real target, pick the dose whose
cell is closest to the target (smallest FID-to-target) and compare its real mAP with clear training, a random dose (mean) and the oracle."""
import json, os, numpy as np
from itertools import combinations
T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]
CAL=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
HO1=["d10_fog35","d11_fog65","d12_fog90","d13_sun6","d14_night_rain","d15_night_fog50","d16_sun10_wet","d17_cloud90","d18_dust50"]
m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
own={R:{t:m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json") for t in T} for R in ("results/calib_dose","results/calib_dose_rep")}
M0=json.load(open("results/calib_dose/dose_metrics.json")); M1=json.load(open("results/calib_dose_ho/dose_metrics.json"))
def drop(t,d):
    if d in CAL: return np.mean([own[R][t]-m(f"{R}/eval_{t}_d00_ref__on__{t}_{d}.json") for R in own])
    return np.mean([own["results/calib_dose"][t]-m(f"results/calib_dose_ho/eval_{t}_d00_ref__on__{t}_{d}.json"),own["results/calib_dose_rep"][t]-m(f"results/calib_dose_ho_rep/eval_{t}_d00_ref__on__{t}_{d}.json")])
ALL=CAL+HO1; X={d:np.array([(M0 if d in CAL else M1)[f"{t}_{d}"]["obj_fid_ref"] for t in T]) for d in ALL}; Y={d:np.array([drop(t,d) for t in T]) for d in ALL}
rng=np.random.default_rng(0); print("(a) calibration doses needed: calibrate on k doses (all 8 towns), predict the other doses; MAE (pp), 300 random dose subsets per k")
for k in (2,3,4,6,9,12):
    e=[]
    for _ in range(300):
        tr=list(rng.choice(ALL,k,replace=False)); te=[d for d in ALL if d not in tr]
        x=np.concatenate([X[d] for d in tr]); y=np.concatenate([Y[d] for d in tr])
        if np.ptp(x)==0: continue
        b,a=np.polyfit(x,y,1); e.append(np.mean(np.concatenate([np.abs(Y[d]-(a+b*X[d])) for d in te])))
    print(f"  k={k:2d} doses: MAE {np.mean(e):5.2f} (10-90%: {np.percentile(e,10):.2f}-{np.percentile(e,90):.2f})")
yall=np.concatenate([Y[d] for d in ALL]); print(f"  predict-the-mean over all 18 doses: {np.mean(np.abs(yall-yall.mean())):.2f}")
print("\n(b) metric-guided choice of training appearance (10 candidate doses per town; seed-42 detectors)")
D10=["d00_ref"]+CAL; FR=json.load(open("results/calib_dose_rtts/fid_to_rtts.json")); FF=json.load(open("results/calib_dose_foggy/fid_to_foggy.json"))
TG={"KITTI":(lambda c: M0[c]["fid_KITTI"], lambda c: m(f"results/calib_dose/src_{c}_kittival.json")),
    "Cityscapes":(lambda c: M0[c]["fid_Cityscapes"], lambda c: m(f"results/calib_dose/src_{c}_cityscapesval.json")),
    "Foggy CS b=.02":(lambda c: FF[c]["fid_foggy_0.02"], lambda c: m(f"results/calib_dose_foggy/src_{c}_foggy0.02.json")),
    "RTTS (real fog)":(lambda c: FR[c]["fid_RTTS"], lambda c: m(f"results/calib_dose_rtts/src_{c}_rtts.json"))}
for name,(fd,perf) in TG.items():
    rows=[]
    for t in T:
        cs=[f"{t}_{d}" for d in D10]; f=np.array([fd(c) for c in cs]); p=np.array([perf(c) for c in cs]); rows.append((p[0],p.mean(),p[np.argmin(f)],p.max()))
    r=np.array(rows); gain=(r[:,2]-r[:,0]).mean(); og=(r[:,3]-r[:,0]).mean()
    print(f"  {name:<16} clear {r[:,0].mean():5.1f} | random dose {r[:,1].mean():5.1f} | metric-chosen {r[:,2].mean():5.1f} | oracle {r[:,3].mean():5.1f} | metric captures {100*gain/og if og>0 else float('nan'):4.0f}% of the oracle gain over clear; beats random in {(r[:,2]>r[:,1]).sum()}/8 towns")
