#!/usr/bin/env python3
"""Score the pre-registered Chou et al. sim->real test (ledger §15.42)."""
import json, glob, os, numpy as np
from pathlib import Path
from scipy import stats
K=json.load(open("results/manifold_markers/markers.json"))
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
def nbox(root,c): return sum(1 for f in sorted(glob.glob(f"{root}/{c}/frame_*.txt"))[:800] for l in open(f) if l.split() and l.split()[0] in ("Car","Pedestrian","Cyclist"))
MK=["D_eff","Psi_eff","R_eff","alpha","PR"]
def val(r,k): return float(np.mean(list(r["PR"].values()))) if k=="PR" else r.get(k,np.nan)
def resdir(name):
    pre,cell=name.split("_",1)
    if pre=="anchin": return "results/calib_imnet_towns2" if os.path.exists(f"results/calib_imnet_towns2/src_{cell}_kittival.json") else "results/calib_imnet"
    return {"anch":"results/calib_anchored","anchinrep":"results/calib_imnet_rep","anchinlq":"results/calib_imnet_lowq","anchinlc":"results/calib_imnet_lc"}[pre]
def real(name,key):
    pre,cell=name.split("_",1); f=f"{resdir(name)}/src_{cell}_{key}.json"; return m(f) if os.path.exists(f) else np.nan
def own(name):
    pre,cell=name.split("_",1); R=resdir(name)
    for f in (f"{R}/eval_{cell}__on__{cell}.json",f"{R}/eval_{cell}__on__{cell}_low.json"):
        if os.path.exists(f): return m(f)
    return np.nan
def _partial_r(rx,ry,k):
    """Pearson r of two residual vectors with its two-sided p-value on n - 2 - k degrees of freedom (k controlled variables)."""
    r=float(stats.pearsonr(rx,ry)[0]); df=len(rx)-2-k; t=r*np.sqrt(df/max(1e-12,1-r*r)); return r,float(2*stats.t.sf(abs(t),df))
def partial(x,y,z):
    Z=np.column_stack([np.ones(len(y)),z]); rx=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]; ry=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]; return _partial_r(rx,ry,Z.shape[1]-1)
A=sorted(n for n in K if n.startswith("anchin_"))
print(f"=== Rule 1/2/4: {len(A)} ImageNet-backbone sources, markers on own source data vs real-target mAP ===")
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    y=np.array([real(n,key) for n in A]); idm=np.array([own(n) for n in A]); lb=np.log([nbox(K[n]["root"],K[n]["cell"]) for n in A])
    r0=stats.pearsonr(idm,y); print(f"  {T}: ID accuracy (own-cell held-out mAP) r={r0[0]:+.3f} (p={r0[1]:.3f})")
    for k in MK:
        x=np.array([val(K[n],k) for n in A]); ok=~np.isnan(x)&~np.isnan(y); r=stats.pearsonr(x[ok],y[ok]); pr=partial(x[ok],y[ok],lb[ok])
        print(f"    {k:<8} r={r[0]:+.3f} (p={r[1]:.4f})   partial|log boxes {pr[0]:+.3f} (p={pr[1]:.3f})")
print("\n=== Rule 3: Epic vs Low (same town), marker difference vs real-mAP difference ===")
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    rows=[(c,real("anchin_"+c,key)-real("anchinlq_"+c,key),{k:val(K["anchin_"+c],k)-val(K["anchinlq_"+c],k) for k in MK}) for c in [n.split("_",1)[1] for n in K if n.startswith("anchinlq_")] if "anchin_"+c in K]
    print(f"  {T}: ΔmAP(Epic-Low) "+", ".join(f"{c.split('_')[0]} {d:+.1f}" for c,d,_ in rows))
    for k in MK: print(f"    {k:<8} sign agreement {sum(np.sign(dm)==np.sign(dk[k]) for _,dm,dk in rows)}/{len(rows)}   Δ: "+", ".join(f"{dk[k]:+.3f}" for _,_,dk in rows))
print("\n=== Learning curve (content fixed, data quantity varies): within-cell r(marker, real mAP), pooled fixed effects ===")
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    for k in MK:
        X,Y=[],[]
        for c in sorted({n.split("_",1)[1].rsplit("_n",1)[0] for n in K if n.startswith("anchinlc_")}):
            names=[f"anchinlc_{c}_n{s}" for s in (100,200,400)]+[f"anchin_{c}"]; names=[n for n in names if n in K]
            x=np.array([val(K[n],k) for n in names]); yy=np.array([real(n,key) for n in names]); ok=~np.isnan(x)&~np.isnan(yy)
            X+=list(x[ok]-x[ok].mean()); Y+=list(yy[ok]-yy[ok].mean())
        r=stats.pearsonr(X,Y); print(f"  {T} {k:<8} within-cell r={r[0]:+.3f} (p={r[1]:.4f}, n={len(X)})")
print("\n=== Noise: seed-17 replicates vs seed-42 (8 cells): reliability = 1 - noise var / across-source var ===")
for k in MK:
    reps=[n.split("_",1)[1] for n in K if n.startswith("anchinrep_")]; d=np.array([val(K["anchinrep_"+c],k)-val(K["anchin_"+c],k) for c in reps if "anchin_"+c in K])
    d=d[~np.isnan(d)]; allv=np.array([val(K[n],k) for n in A]); allv=allv[~np.isnan(allv)]; nv=(np.std(d,ddof=1)/np.sqrt(2))**2; print(f"  {k:<8} per-run noise sd {np.sqrt(nv):.4f} (n={len(d)})  across-source sd {allv.std(ddof=1):.4f}  reliability {max(0,1-nv/allv.var(ddof=1)):.2f}")
C=sorted(n for n in K if n.startswith("anch_"))
print(f"\n=== COCO control: {len(C)} COCO-init detectors (real mAP flat) ===")
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    y=np.array([real(n,key) for n in C]); xs={k:np.array([val(K[n],k) for n in C]) for k in MK}; print(f"  {T} mAP {y.mean():.1f} ± {y.std():.1f}: "+"; ".join(f"{k} r={stats.pearsonr(xs[k][~np.isnan(xs[k])],y[~np.isnan(xs[k])])[0]:+.2f}" for k in MK))
print("  marker means COCO vs ImageNet: "+"; ".join(f"{k} {np.nanmean([val(K[n],k) for n in C]):.3f} vs {np.nanmean([val(K[n],k) for n in A]):.3f}" for k in MK))
