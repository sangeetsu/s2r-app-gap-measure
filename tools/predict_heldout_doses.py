#!/usr/bin/env python3
"""Ledger §15.65 (E1): write predicted appearance-only drops for the held-out doses BEFORE any reference detector is evaluated on them.
Calibration = OLS of the seed-mean drop on metric(ref, dose) over the 72 calibration pairs (results/calib_dose + calib_dose_rep)."""
import json, os, glob, time, hashlib, sys, numpy as np
from scipy import stats
HO="results/calib_dose_ho"; T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]
CAL=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
NEW={"d10_fog35":"interpolation","d11_fog65":"interpolation","d13_sun6":"interpolation","d12_fog90":"extrapolation","d14_night_rain":"compound",
     "d15_night_fog50":"compound","d16_sun10_wet":"compound","d17_cloud90":"novel","d18_dust50":"novel"}
MET=[("obj_fid_ref","dm","primary"),("nnd_dose_to_ref","dm","secondary"),("cov_dose_by_ref","dm","secondary"),("paired_obj_disp","dm","secondary"),
     ("lpips","ll","secondary"),("incfid_ref","ll","secondary"),("img_fid_ref","dm","secondary")]
if glob.glob(f"{HO}/eval_*") or glob.glob(f"{HO}_rep/eval_*"): sys.exit("REFUSING: held-out eval files already exist")
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
def drop(R,t,d): return m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{R}/eval_{t}_d00_ref__on__{t}_{d}.json")
y=np.array([(drop("results/calib_dose",t,d)+drop("results/calib_dose_rep",t,d))/2 for t in T for d in CAL])
src={"dm":(json.load(open("results/calib_dose/dose_metrics.json")),json.load(open(f"{HO}/dose_metrics.json"))),
     "ll":(json.load(open("results/calib_dose/dose_lowlevel.json")),json.load(open(f"{HO}/dose_lowlevel.json")))}
out={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"ledger":"15.65","calibration_pairs":len(y),"subsets":NEW,"models":{},"cells":{},"control_reproduction":{}}
for k,s,role in MET:
    cal,ho=src[s]; x=np.array([cal[f"{t}_{d}"][k] for t in T for d in CAL]); n=len(x); b,a=np.polyfit(x,y,1); res=y-(a+b*x); sres=np.sqrt((res**2).sum()/(n-2)); xb=x.mean(); sxx=((x-xb)**2).sum(); tq=stats.t.ppf(.95,n-2)
    out["models"][k]={"role":role,"intercept":float(a),"slope":float(b),"resid_sd":float(sres),"n":n,"spearman_cal":float(stats.spearmanr(x,y)[0])}
    for t in T:
        for d in NEW:
            c=f"{t}_{d}"
            if c not in ho: continue
            x0=ho[c][k]; yh=a+b*x0; hw=tq*sres*np.sqrt(1+1/n+(x0-xb)**2/sxx)
            out["cells"].setdefault(c,{"town":t,"dose":d,"subset":NEW[d]})[k]={"x":float(x0),"pred":float(yh),"pi90":[float(yh-hw),float(yh+hw)]}
        c=f"{t}_d06_fog50"
        if c in ho: out["control_reproduction"].setdefault(c,{})[k]={"calibration":float(cal[c][k]),"rerun":float(ho[c][k])}
out["calibration_mean_drop"]=float(y.mean())
blob=json.dumps(out,indent=1,sort_keys=True); open(f"{HO}/predictions.json","w").write(blob); h=hashlib.sha256(blob.encode()).hexdigest()
open(f"{HO}/predictions.sha256","w").write(f"{h}  predictions.json  {out['created']}\n")
print(f"[predict] {len(out['cells'])} held-out cells predicted at {out['created']}; sha256 {h}")
for c,v in out["control_reproduction"].items(): print(f"  control {c}: "+", ".join(f"{k} {w['calibration']:.3f}/{w['rerun']:.3f}" for k,w in v.items()))
