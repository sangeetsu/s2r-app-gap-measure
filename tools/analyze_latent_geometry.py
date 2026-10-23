#!/usr/bin/env python3
"""Tests for tools/latent_geometry.py (H1 asymmetry, H2 task projection) against observed detector drops."""
import json, glob, os, numpy as np
from pathlib import Path
from scipy import stats
P=json.load(open("results/latent_geometry/pairs.json"))["pairs"]; G="results/calib_imnet"; L="results/calib_imnet_lowq"
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
MEAS=["fid_full","fid_task","fid_nuis","cov_T_by_S","cov_task_T_by_S","nnd_T_to_S","nnd_task_T_to_S"]
grid=sorted(c for c in P if "@low" not in c and glob.glob(f"{G}/eval_{c}__on__{c}.json")); anchor={c:m(f"{G}/eval_{c}__on__{c}.json") for c in grid}
pairs=[(S,T) for S in grid for T in grid if S!=T and T in P[S]]; y=np.array([anchor[S]-m(f"{G}/eval_{S}__on__{T}.json") for S,T in pairs])
src=np.array([anchor[S] for S,T in pairs]); tgt=np.array([anchor[T] for S,T in pairs])
def partial(x,zs):
    Z=np.column_stack([np.ones(len(y))]+zs); rx=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]; ry=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]; return stats.pearsonr(rx,ry)
def loo(cols):
    A=np.column_stack([np.ones(len(y))]+cols); errs=[]
    for h in grid:
        te=np.array([h in pr for pr in pairs]); tr=~te; b=np.linalg.lstsq(A[tr],y[tr],rcond=None)[0]; errs+=list(np.abs(y[te]-A[te]@b))
    e=np.array(errs); return 1-np.mean(e**2)/y.var(), 100*(e<=4).mean()
print(f"=== A. within-CARLA, {len(pairs)} directed pairs, ImageNet-backbone drops; both anchors LOO R2 {loo([src,tgt])[0]:+.3f}")
print(f"  {'measure':<18}{'raw r':>8}{'partial|anchors':>18}{'LOO R2 +meas':>14}{'<=4pp':>7}")
for k in MEAS:
    x=np.array([P[S][T][k] for S,T in pairs],float); r0=stats.pearsonr(x,y); pr=partial(x,[src,tgt]); r2,p4=loo([src,tgt,x])
    print(f"  {k:<18}{r0[0]:+8.3f}  {pr[0]:+.3f} (p={pr[1]:.0e})    {r2:+.3f}   {p4:4.0f}%")
print("\n=== B. Epic<->Low asymmetry (5 towns): observed drop vs coverage in each direction")
print(f"  {'town':<10}{'drop E->L':>10}{'drop L->E':>10} | {'cov(L|E)':>9}{'cov(E|L)':>9} | {'fidT E->L':>10}{'fidT L->E':>10} | {'fidF E->L':>10}{'fidF L->E':>10}")
agree=0; n=0
for c in [c for c in grid if c+"@low" in P]:
    lw=c+"@low"; dEL=anchor[c]-m(f"{L}/eval_epic_{c}__on__{c}_low.json"); dLE=m(f"{L}/eval_{c}__on__{c}_low.json")-m(f"{L}/eval_{c}__on__{c}_epic.json")
    a,b=P[c][lw],P[lw][c]; n+=1; agree+=int((dLE>dEL)==(a["cov_T_by_S"]>b["cov_T_by_S"]))
    print(f"  {c.split('_')[0]:<10}{dEL:10.1f}{dLE:10.1f} | {a['cov_T_by_S']:9.2f}{b['cov_T_by_S']:9.2f} | {a['fid_task']:10.2f}{b['fid_task']:10.2f} | {a['fid_full']:10.1f}{b['fid_full']:10.1f}")
print(f"  harder direction = lower coverage in {agree}/{n} towns")
def nbox(c):
    for r in (f"{Path.home()}/datasets/carla_extracted","/mnt/features/carla_new"):
        if os.path.isdir(f"{r}/{c}"): return sum(1 for f in sorted(glob.glob(f"{r}/{c}/frame_*.txt"))[:800] for l in open(f) if l.split() and l.split()[0] in ("Car","Pedestrian","Cyclist"))
lb=np.array([np.log(nbox(c)) for c in grid])
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    yr=np.array([m(f"{G}/src_{c}_{key}.json") for c in grid]); print(f"\n=== C. {T}: {len(grid)} sources, real mAP; r / partial|log boxes")
    for k in MEAS:
        x=np.array([P[c][T][k] for c in grid],float); r0=stats.pearsonr(x,yr); Z=np.column_stack([np.ones(len(yr)),lb]); rx=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]; ry=yr-Z@np.linalg.lstsq(Z,yr,rcond=None)[0]; pr=stats.pearsonr(rx,ry)
        print(f"  {k:<18} r={r0[0]:+.3f} (p={r0[1]:.3f})  partial|boxes {pr[0]:+.3f} (p={pr[1]:.3f})")
    lows=[c for c in grid if c+"@low" in P and glob.glob(f"{L}/src_{c}_{key}.json")]
    if len(lows)>=4:
        dm=np.array([m(f"{G}/src_{c}_{key}.json")-m(f"{L}/src_{c}_{key}.json") for c in lows])
        print(f"  Epic-Low per town, ΔmAP vs Δmeasure (n={len(lows)}): "+"; ".join(f"{k} r={stats.pearsonr([P[c+'@low'][T][k]-P[c][T][k] for c in lows],-dm)[0]:+.2f}" for k in ("fid_full","fid_task","cov_T_by_S")))
