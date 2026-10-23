#!/usr/bin/env python3
"""Ledger §15.98 (RT5): predict the real-twin drop with the frozen weather formula (tau = 239.3, §15.89) BEFORE any reference-detector eval on d50/d51."""
import json, glob, time, hashlib, sys, math, numpy as np
R = "results/pilot_realtwin"; T = ["Town03", "Town05"]; TAU = 239.3
if glob.glob(f"{R}/eval_*d00_ref__on__*_d5*") or glob.glob(f"{R}_rep/eval_*d00_ref__on__*_d5*"):
    sys.exit("REFUSING: reference-detector evals on d50/d51 already exist")
m = lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"] * 100
A = {t: float(np.mean([m(f"{Rr}/eval_{t}_d00_ref__on__{t}_d00_ref.json") for Rr in ("results/calib_dose", "results/calib_dose_rep")])) for t in T}
DM = json.load(open(f"{R}/dose_metrics.json"))
pred = {}
for t in T:
    for d in ("d50_realtwin", "d51_resample"):
        x = DM[f"{t}_{d}"]["obj_fid_ref"]
        pred[f"{t}_{d}"] = dict(A=A[t], obj_fid=x, pred_drop=A[t] * (1 - math.exp(-x / TAU)))
out = dict(written=time.strftime("%Y-%m-%dT%H:%M:%S"), tau=TAU, rule="RT5: |pred - obs| <= 4 pp in both towns (d50)", predictions=pred)
s = json.dumps(out, indent=1, sort_keys=True); h = hashlib.sha256(s.encode()).hexdigest()
open(f"{R}/predictions.json", "w").write(s); open(f"{R}/predictions.sha256", "w").write(h + "\n")
print(s); print("sha256", h)
