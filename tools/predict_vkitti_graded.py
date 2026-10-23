#!/usr/bin/env python3
"""Ledger §15.93: fit formula and line on alpha in {0.5, 1.0}; predict alpha in {0.25, 0.75} before those cells are evaluated."""
import json, glob, time, hashlib, sys, numpy as np
from scipy import optimize
OUT="results/calib_vkitti_graded"; S=["Scene01","Scene02","Scene18","Scene20"]; V=["morning","sunset","overcast","fog","rain"]; ORIG={"morning":"d20_morning","sunset":"d21_sunset","overcast":"d22_overcast","fog":"d23_fog","rain":"d24_rain"}
if [f for f in glob.glob(f"{OUT}/eval_*") if "_a25" in f or "_a75" in f]+[f for f in glob.glob(f"{OUT}_rep/eval_*") if "_a25" in f or "_a75" in f]: sys.exit("REFUSING: test evals exist")
ap=lambda f: json.load(open(f))["metrics"]["AP_Car"]*100; M=json.load(open(f"{OUT}/dose_metrics.json"))
A={s:np.mean([ap(f"results/calib_vkitti{x}/eval_{s}_d00_ref__on__{s}_d00_ref.json") for x in ("","_rep")]) for s in S}
def cell(s,v,a): return f"{s}_{ORIG[v]}" if a==100 else f"{s}_d4{V.index(v)}_{v}_a{a}"
def obs(s,c): 
    if "_a" in c: return np.mean([A[s]*0+ap(f"results/calib_vkitti{x}/eval_{s}_d00_ref__on__{s}_d00_ref.json")-ap(f"{OUT}{x}/eval_{s}_d00_ref__on__{c}.json") for x in ("","_rep")])
    return np.mean([ap(f"results/calib_vkitti{x}/eval_{s}_d00_ref__on__{s}_d00_ref.json")-ap(f"results/calib_vkitti{x}/eval_{s}_d00_ref__on__{c}.json") for x in ("","_rep")])
cal=[(s,cell(s,v,a)) for s in S for v in V for a in (50,100)]
x=np.array([M[c]["obj_fid_ref"] for _,c in cal]); y=np.array([obs(s,c) for s,c in cal]); Av=np.array([A[s] for s,_ in cal])
tau=float(optimize.minimize_scalar(lambda t: np.sum((y-Av*(1-np.exp(-x/t)))**2),bounds=(10,1e5),method="bounded").x); b,a=np.polyfit(x,y,1)
out={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"ledger":"15.93","tau":tau,"line":[float(a),float(b)],"A":A,"cal_mean":float(y.mean()),"cells":{}}
for s in S:
    for v in V:
        for al in (25,75):
            c=cell(s,v,al); x0=M[c]["obj_fid_ref"]; out["cells"][c]={"scene":s,"variant":v,"alpha":al,"x":float(x0),"formula":float(A[s]*(1-np.exp(-x0/tau))),"line":float(a+b*x0)}
blob=json.dumps(out,indent=1,sort_keys=True); open(f"{OUT}/predictions.json","w").write(blob); h=hashlib.sha256(blob.encode()).hexdigest()
open(f"{OUT}/predictions.sha256","w").write(f"{h}  predictions.json  {out['created']}\n"); print(f"[vg] predictions {out['created']} sha256 {h}; tau {tau:.0f}; cal pairs {len(cal)}")
