#!/usr/bin/env python3
"""Ledger §15.98: score the real-twin pilot (RT1-RT5 + control)."""
import json, math, os, hashlib, numpy as np
R = "results/pilot_realtwin"; T = ["Town03", "Town05"]; SEEDS = {"42": ("results/calib_dose", R), "17": ("results/calib_dose_rep", R + "_rep")}
m = lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"] * 100
P = json.load(open(f"{R}/predictions.json")); s = json.dumps(P, indent=1, sort_keys=True)
assert hashlib.sha256(s.encode()).hexdigest() == open(f"{R}/predictions.sha256").read().strip(), "prediction hash mismatch"
for t in T:
    for d in ("d50_realtwin", "d51_resample"):
        for _, (_, rr) in SEEDS.items():
            f = f"{rr}/eval_{t}_d00_ref__on__{t}_{d}.json"; assert os.path.getmtime(f) > os.path.getmtime(f"{R}/predictions.json"), f"order violated: {f}"
print(f"predictions written {P['written']}, sha256 verified, order verified\n")
S = json.load(open(f"{R}/survival.json")); DM = json.load(open(f"{R}/dose_metrics.json"))
num = sum(S[t]["summary"]["all"]["rest"][0] for t in T); den = sum(S[t]["summary"]["all"]["rest"][1] for t in T)
pn = sum(S[t]["summary"]["Pedestrian"]["rest"][0] for t in T); pd = sum(S[t]["summary"]["Pedestrian"]["rest"][1] for t in T)
cn = sum(S[t]["summary"]["all"]["ctrl"][0] for t in T)
print(f"RT1 survival {num}/{den} = {num/den:.1%} (Pedestrian {pn}/{pd}; resampling control {cn}/{den}); unlabeled/frame " +
      ", ".join(f"{t} orig {S[t]['summary']['inserted_per_frame']['orig']:.2f} twin {S[t]['summary']['inserted_per_frame']['rest']:.2f}" for t in T))
rt1 = num / den >= 0.90; print(f"RT1 {'PASS' if rt1 else 'FAIL'}\n")
A = {t: P["predictions"][f"{t}_d50_realtwin"]["A"] for t in T}
def drop(t, d): return float(np.mean([m(f"{a}/eval_{t}_d00_ref__on__{t}_d00_ref.json") - m(f"{b}/eval_{t}_d00_ref__on__{t}_{d}.json") for a, b in SEEDS.values()]))
def real(t, cell, tgt, ref=False):
    v = []
    for seed, (a, b) in SEEDS.items():
        base = a if ref else b
        f = f"{base}/src_{cell}_kittival.json" if tgt == "KITTI" else f"{base}/src_{cell}_cityscapesval.json"
        v.append(m(f))
    return float(np.mean(v)), v
print("town   cell            drop(pp)  objFID   pred(τ239)  KITTI gain  Cityscapes gain  fid_KITTI (ref→cell)")
gains = {"d50_realtwin": [], "d51_resample": []}; taus = {}; rt5 = []; ctrl = []
for t in T:
    for d in ("d50_realtwin", "d51_resample"):
        y = drop(t, d); x = DM[f"{t}_{d}"]["obj_fid_ref"]; yp = P["predictions"][f"{t}_{d}"]["pred_drop"]
        g = {}
        for tgt in ("KITTI", "Cityscapes"):
            r0, _ = real(t, f"{t}_d00_ref", tgt, ref=True); r1, per = real(t, f"{t}_{d}", tgt); g[tgt] = r1 - r0; gains[d].append(r1 - r0)
        fk0, fk1 = DM[f"{t}_d00_ref"].get("fid_KITTI"), DM[f"{t}_{d}"].get("fid_KITTI")
        print(f"{t} {d:<15} {y:8.2f} {x:8.1f} {yp:10.2f} {g['KITTI']:+10.2f} {g['Cityscapes']:+15.2f}   {fk0:.1f} → {fk1:.1f}")
        if d == "d50_realtwin":
            taus[t] = -x / math.log(1 - y / A[t]) if 0 < y < A[t] else float("nan"); rt5.append(abs(yp - y) <= 4)
        else: ctrl.append(abs(y) <= 2)
print()
g50, g51 = gains["d50_realtwin"], gains["d51_resample"]
rt2 = sum(v > 0 for v in g50) >= 3 and np.mean(g50) > np.mean(g51)
print(f"RT2 real gain d50 {[round(v,2) for v in g50]} mean {np.mean(g50):+.2f} vs control mean {np.mean(g51):+.2f} → {'PASS' if rt2 else 'FAIL'}")
tv = list(taus.values()); rt3 = all(np.isfinite(tv)) and max(tv) / min(tv) <= 2
print(f"RT3 τ_twin {', '.join(f'{t} {v:.0f}' for t, v in taus.items())} (ratio {max(tv)/min(tv) if all(np.isfinite(tv)) else float('nan'):.2f}) → {'PASS' if rt3 else 'FAIL'}" + ("" if rt1 else "  [not interpreted: RT1 failed]"))
rt4 = all(DM[f"{t}_d50_realtwin"]["fid_KITTI"] < DM[f"{t}_d00_ref"]["fid_KITTI"] for t in T)
print(f"RT4 fid_KITTI lower for the twin in both towns → {'PASS' if rt4 else 'FAIL'}; Cityscapes: " + ", ".join(f"{t} {DM[f'{t}_d00_ref']['fid_Cityscapes']:.1f} → {DM[f'{t}_d50_realtwin']['fid_Cityscapes']:.1f}" for t in T))
print(f"RT5 weather constant (τ 239.3) predicts the twin drop within ±4 pp in both towns → {'PASS' if all(rt5) else 'FAIL'}")
print(f"Control d51 drop within ±2 pp in both towns → {'OK' if all(ctrl) else 'NOT OK'}")
print("\nTwin-trained detectors: own held-out and on the clean reference (direction):")
for t in T:
    for seed, (_, b) in SEEDS.items():
        c = f"{t}_d50_realtwin"
        print(f"  {t} seed {seed}: own {m(f'{b}/eval_{c}__on__{c}.json'):.1f}, on d00_ref {m(f'{b}/eval_{c}__on__{t}_d00_ref.json'):.1f}; reference own {m(f'{SEEDS[seed][0]}/eval_{t}_d00_ref__on__{t}_d00_ref.json'):.1f}")
print("\nPer-seed real mAP@.5 (KITTI val / Cityscapes val):")
for t in T:
    for cell, ref in ((f"{t}_d00_ref", True), (f"{t}_d50_realtwin", False), (f"{t}_d51_resample", False)):
        k = real(t, cell, "KITTI", ref)[1]; c = real(t, cell, "Cityscapes", ref)[1]
        print(f"  {cell:<20} KITTI {k[0]:.1f}/{k[1]:.1f}  Cityscapes {c[0]:.1f}/{c[1]:.1f}")
