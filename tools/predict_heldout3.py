#!/usr/bin/env python3
"""Ledger §15.87: fit M0 (line), M1 (town susceptibility), M2 (saturating, town ceiling; primary) on 192 observed pairs; predict round-3 doses BEFORE any evaluation."""
import json, glob, time, hashlib, sys, numpy as np
from scipy import optimize
H3="results/calib_dose_ho3"; T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]
C0=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
R1=["d10_fog35","d11_fog65","d12_fog90","d13_sun6","d14_night_rain","d15_night_fog50","d16_sun10_wet","d17_cloud90","d18_dust50"]
R2=["d19_rain_fog25","d20_rain_fog50","d21_fog15","d22_fog45","d23_sun0","d24_night_wet"]
NEW=["d25_fog5","d26_fog30","d27_fog55","d28_fog80","d29_sun15","d30_sun_m5","d31_night_fog25","d32_dust90","d33_rain_fog75"]
if glob.glob(f"{H3}/eval_*") or glob.glob(f"{H3}_rep/eval_*"): sys.exit("REFUSING: round-3 eval files already exist")
m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
own={R:{t:m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json") for t in T} for R in ("results/calib_dose","results/calib_dose_rep")}; A={t:float(np.mean([own[R][t] for R in own])) for t in T}
SRC={**{d:("results/calib_dose","results/calib_dose_rep","results/calib_dose/dose_metrics.json") for d in C0},**{d:("results/calib_dose_ho","results/calib_dose_ho_rep","results/calib_dose_ho/dose_metrics.json") for d in R1},**{d:("results/calib_dose_ho2","results/calib_dose_ho2_rep","results/calib_dose_ho2/dose_metrics.json") for d in R2}}
MJ={}; tw=[]; X=[]; Y=[]
for d,(a,b,mf) in SRC.items():
    MJ.setdefault(mf,json.load(open(mf)))
    for t in T: tw.append(t); X.append(MJ[mf][f"{t}_{d}"]["obj_fid_ref"]); Y.append(np.mean([own["results/calib_dose"][t]-m(f"{a}/eval_{t}_d00_ref__on__{t}_{d}.json"),own["results/calib_dose_rep"][t]-m(f"{b}/eval_{t}_d00_ref__on__{t}_{d}.json")]))
tw=np.array(tw); X=np.array(X); Y=np.array(Y); b,a=np.polyfit(X,Y,1)
town={}
for t in T:
    i=tw==t; bt=np.sum((X[i]-X[i].mean())*(Y[i]-Y[i].mean()))/np.sum((X[i]-X[i].mean())**2); bs=.5*bt+.5*b; town[t]={"slope":float(bs),"intercept":float(Y[i].mean()-bs*X[i].mean())}
Av=np.array([A[t] for t in tw]); tau=float(optimize.minimize_scalar(lambda z: np.sum((Y-Av*(1-np.exp(-X/z)))**2),bounds=(10,5000),method="bounded").x)
MN=json.load(open(f"{H3}/dose_metrics.json")); out={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"ledger":"15.87","n_fit":len(X),"fit_mean":float(Y.mean()),"M0":{"intercept":float(a),"slope":float(b)},"M1":town,"M2":{"tau":tau,"ceiling_A":A},"cells":{},"control":{}}
for t in T:
    for d in NEW:
        c=f"{t}_{d}"
        if c in MN: x0=MN[c]["obj_fid_ref"]; out["cells"][c]={"town":t,"dose":d,"x":float(x0),"M0":float(a+b*x0),"M1":float(town[t]["intercept"]+town[t]["slope"]*x0),"M2":float(A[t]*(1-np.exp(-x0/tau)))}
    c=f"{t}_d06_fog50"
    if c in MN: out["control"][c]={"calibration":float(MJ["results/calib_dose/dose_metrics.json"][c]["obj_fid_ref"]),"rerun":float(MN[c]["obj_fid_ref"])}
blob=json.dumps(out,indent=1,sort_keys=True); open(f"{H3}/predictions.json","w").write(blob); h=hashlib.sha256(blob.encode()).hexdigest()
open(f"{H3}/predictions.sha256","w").write(f"{h}  predictions.json  {out['created']}\n"); print(f"[predict3] {len(out['cells'])} cells at {out['created']} sha256 {h}; tau={tau:.1f}")
for c,v in out["control"].items(): print(f"  control {c}: {v['calibration']:.3f} / {v['rerun']:.3f}")
