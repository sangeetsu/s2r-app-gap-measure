#!/usr/bin/env python3
"""Ledger §15.76: linear vs isotonic calibration (object FID -> seed-mean appearance-only drop) fitted on 144 pairs (9 original + 9 round-1 held-out
doses x 8 towns); predictions for the 6 round-2 doses written BEFORE any reference detector is evaluated on them."""
import json, glob, time, hashlib, sys, numpy as np
from sklearn.isotonic import IsotonicRegression
H2="results/calib_dose_ho2"; T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]
CAL=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
HO1=["d10_fog35","d11_fog65","d12_fog90","d13_sun6","d14_night_rain","d15_night_fog50","d16_sun10_wet","d17_cloud90","d18_dust50"]
NEW=["d19_rain_fog25","d20_rain_fog50","d21_fog15","d22_fog45","d23_sun0","d24_night_wet"]
if glob.glob(f"{H2}/eval_*") or glob.glob(f"{H2}_rep/eval_*"): sys.exit("REFUSING: round-2 eval files already exist")
m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
own={R:{t:m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json") for t in T} for R in ("results/calib_dose","results/calib_dose_rep")}
def y(t,d):
    if d in CAL: return np.mean([own[R][t]-m(f"{R}/eval_{t}_d00_ref__on__{t}_{d}.json") for R in own])
    return np.mean([own["results/calib_dose"][t]-m(f"results/calib_dose_ho/eval_{t}_d00_ref__on__{t}_{d}.json"),own["results/calib_dose_rep"][t]-m(f"results/calib_dose_ho_rep/eval_{t}_d00_ref__on__{t}_{d}.json")])
M0=json.load(open("results/calib_dose/dose_metrics.json")); M1=json.load(open("results/calib_dose_ho/dose_metrics.json")); MN=json.load(open(f"{H2}/dose_metrics.json"))
X=np.array([(M0 if d in CAL else M1)[f"{t}_{d}"]["obj_fid_ref"] for t in T for d in CAL+HO1]); Y=np.array([y(t,d) for t in T for d in CAL+HO1])
b,a=np.polyfit(X,Y,1); iso=IsotonicRegression(increasing=True,out_of_bounds="clip").fit(X,Y)
out={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"ledger":"15.76","n_cal":len(X),"cal_mean":float(Y.mean()),"linear":{"intercept":float(a),"slope":float(b)},
     "cal_x_range":[float(X.min()),float(X.max())],"cells":{},"control":{}}
for t in T:
    for d in NEW:
        c=f"{t}_{d}"
        if c in MN: x0=MN[c]["obj_fid_ref"]; out["cells"][c]={"town":t,"dose":d,"x":float(x0),"pred_linear":float(a+b*x0),"pred_isotonic":float(iso.predict([x0])[0])}
    c=f"{t}_d06_fog50"
    if c in MN: out["control"][c]={"calibration":float(M0[c]["obj_fid_ref"]),"rerun":float(MN[c]["obj_fid_ref"])}
blob=json.dumps(out,indent=1,sort_keys=True); open(f"{H2}/predictions.json","w").write(blob); h=hashlib.sha256(blob.encode()).hexdigest()
open(f"{H2}/predictions.sha256","w").write(f"{h}  predictions.json  {out['created']}\n"); print(f"[predict2] {len(out['cells'])} cells at {out['created']} sha256 {h}")
for c,v in out["control"].items(): print(f"  control {c}: {v['calibration']:.3f} / {v['rerun']:.3f}")
