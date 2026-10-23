#!/usr/bin/env python3
"""Ledger §15.92: formula across detector families; predictions for rounds 2+3 written before FCOS/Swin-T are evaluated on them."""
import json, glob, time, hashlib, sys, numpy as np
from scipy import optimize
OUT="results/calib_formula_families"; FAM={"fcos":"results/calib_dose_arch2","swin":"results/calib_dose_arch3"}; T=["Town01","Town10HD"]
C0=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
R1=["d10_fog35","d11_fog65","d12_fog90","d13_sun6","d14_night_rain","d15_night_fog50","d16_sun10_wet","d17_cloud90","d18_dust50"]
R2=["d19_rain_fog25","d20_rain_fog50","d21_fog15","d22_fog45","d23_sun0","d24_night_wet"]; R3=["d25_fog5","d26_fog30","d27_fog55","d28_fog80","d29_sun15","d30_sun_m5","d31_night_fog25","d32_dust90","d33_rain_fog75"]
if glob.glob(f"{OUT}/eval_*"): sys.exit("REFUSING: evals exist")
m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
M={**{d:json.load(open("results/calib_dose/dose_metrics.json")) for d in C0},**{d:json.load(open("results/calib_dose_ho/dose_metrics.json")) for d in R1},**{d:json.load(open("results/calib_dose_ho2/dose_metrics.json")) for d in R2},**{d:json.load(open("results/calib_dose_ho3/dose_metrics.json")) for d in R3}}
TAU_SHARED=json.load(open("results/calib_dose_ho3/predictions.json"))["M2"]["tau"]
out={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"ledger":"15.92","tau_shared":TAU_SHARED,"families":{}}
for f,R in FAM.items():
    A={t:m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json") for t in T}
    def y(t,d): return A[t]-(m(f"{R}/eval_{t}_d00_ref__on__{t}_{d}.json") if d in C0 else m(f"results/calib_crossfamily/eval_{f}_{t}_{d}.json"))
    tw=np.array([t for t in T for d in C0+R1]); x=np.array([M[d][f"{t}_{d}"]["obj_fid_ref"] for t in T for d in C0+R1]); yy=np.array([y(t,d) for t in T for d in C0+R1]); Av=np.array([A[t] for t in tw])
    tau=float(optimize.minimize_scalar(lambda z: np.sum((yy-Av*(1-np.exp(-x/z)))**2),bounds=(10,5000),method="bounded").x); b,a=np.polyfit(x,yy,1)
    fam={"A":A,"tau_own":tau,"line":[float(a),float(b)],"cal_mean":float(yy.mean()),"cells":{}}
    for t in T:
        for d in R2+R3:
            x0=M[d][f"{t}_{d}"]["obj_fid_ref"]; fam["cells"][f"{t}_{d}"]={"x":float(x0),"F_own":float(A[t]*(1-np.exp(-x0/tau))),"F_shared":float(A[t]*(1-np.exp(-x0/TAU_SHARED))),"L_own":float(a+b*x0)}
    out["families"][f]=fam; print(f"[{f}] A {A} tau_own {tau:.1f} (shared {TAU_SHARED:.1f}); line {a:.2f}+{b:.4f}x")
blob=json.dumps(out,indent=1,sort_keys=True); open(f"{OUT}/predictions.json","w").write(blob); h=hashlib.sha256(blob.encode()).hexdigest()
open(f"{OUT}/predictions.sha256","w").write(f"{h}  predictions.json  {out['created']}\n"); print(f"predictions {out['created']} sha256 {h}")
