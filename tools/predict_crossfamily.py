#!/usr/bin/env python3
"""Ledger §15.79: per-family calibration (object FID -> reference-detector drop on the 9 original doses, Town01 + Town10HD) -> predictions for the 9
round-1 held-out doses, written before any held-out evaluation of these detectors. Also the Faster R-CNN line applied unchanged (reported)."""
import json, glob, time, hashlib, sys, numpy as np
from sklearn.isotonic import IsotonicRegression
FAM={"fcos":("results/calib_dose_arch2","anchina"),"swin":("results/calib_dose_arch3","anchins")}; T=["Town01","Town10HD"]
CAL=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
HO1=["d10_fog35","d11_fog65","d12_fog90","d13_sun6","d14_night_rain","d15_night_fog50","d16_sun10_wet","d17_cloud90","d18_dust50"]; OUT="results/calib_crossfamily"
if glob.glob(f"{OUT}/eval_*"): sys.exit("REFUSING: cross-family held-out evals exist")
m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100; M0=json.load(open("results/calib_dose/dose_metrics.json")); M1=json.load(open("results/calib_dose_ho/dose_metrics.json"))
# Faster R-CNN line (seed-mean, 8 towns, 72 pairs) for the 'one calibration for all detectors' variant
TT=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]
yf=np.array([np.mean([m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{R}/eval_{t}_d00_ref__on__{t}_{d}.json") for R in ("results/calib_dose","results/calib_dose_rep")]) for t in TT for d in CAL])
bf,af=np.polyfit(np.array([M0[f"{t}_{d}"]["obj_fid_ref"] for t in TT for d in CAL]),yf,1)
out={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"ledger":"15.79","frcnn_line":[float(af),float(bf)],"families":{}}
for f,(R,_) in FAM.items():
    x=np.array([M0[f"{t}_{d}"]["obj_fid_ref"] for t in T for d in CAL]); y=np.array([m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{R}/eval_{t}_d00_ref__on__{t}_{d}.json") for t in T for d in CAL])
    b,a=np.polyfit(x,y,1); iso=IsotonicRegression(increasing=True,out_of_bounds="clip").fit(x,y); fam={"intercept":float(a),"slope":float(b),"cal_mean":float(y.mean()),"cells":{}}
    for t in T:
        for d in HO1:
            x0=M1[f"{t}_{d}"]["obj_fid_ref"]; fam["cells"][f"{t}_{d}"]={"x":float(x0),"pred_linear":float(a+b*x0),"pred_isotonic":float(iso.predict([x0])[0]),"pred_frcnn_line":float(af+bf*x0)}
    out["families"][f]=fam
blob=json.dumps(out,indent=1,sort_keys=True); open(f"{OUT}/predictions.json","w").write(blob); h=hashlib.sha256(blob.encode()).hexdigest()
open(f"{OUT}/predictions.sha256","w").write(f"{h}  predictions.json  {out['created']}\n"); print(f"[crossfam] predictions {out['created']} sha256 {h}")
