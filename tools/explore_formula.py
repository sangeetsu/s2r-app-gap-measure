#!/usr/bin/env python3
"""EXPLORATORY (post hoc; ledger §15.86): do the failure patterns improve prediction? Fit on the 72 original pairs, score on round 1 (72) and round 2 (48).
M0 linear object FID (pre-registered baseline); M1 + per-town susceptibility (town slope from its own 9 calibration doses, shrunk to the global slope);
M2 saturating with a per-town ceiling (drop = own_AP * (1 - exp(-FID/tau))); M3 family-aware slope (fog-containing vs not, known to the user in advance);
M4 = M1 + M3."""
import json, numpy as np
from scipy import stats, optimize
T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]
C0=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
R1=["d10_fog35","d11_fog65","d12_fog90","d13_sun6","d14_night_rain","d15_night_fog50","d16_sun10_wet","d17_cloud90","d18_dust50"]
R2=["d19_rain_fog25","d20_rain_fog50","d21_fog15","d22_fog45","d23_sun0","d24_night_wet"]
m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
own={R:{t:m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json") for t in T} for R in ("results/calib_dose","results/calib_dose_rep")}; ownm={t:np.mean([own[R][t] for R in own]) for t in T}
SRC={**{d:("results/calib_dose","results/calib_dose_rep","results/calib_dose/dose_metrics.json") for d in C0},**{d:("results/calib_dose_ho","results/calib_dose_ho_rep","results/calib_dose_ho/dose_metrics.json") for d in R1},**{d:("results/calib_dose_ho2","results/calib_dose_ho2_rep","results/calib_dose_ho2/dose_metrics.json") for d in R2}}
MJ={}; rows=[]
for d,(a,b,mf) in SRC.items():
    MJ.setdefault(mf,json.load(open(mf)))
    for t in T: rows.append((t,d,MJ[mf][f"{t}_{d}"]["obj_fid_ref"],np.mean([own["results/calib_dose"][t]-m(f"{a}/eval_{t}_d00_ref__on__{t}_{d}.json"),own["results/calib_dose_rep"][t]-m(f"{b}/eval_{t}_d00_ref__on__{t}_{d}.json")]),"fog" in d))
tw=np.array([r[0] for r in rows]); dz=np.array([r[1] for r in rows]); x=np.array([r[2] for r in rows]); y=np.array([r[3] for r in rows]); fog=np.array([r[4] for r in rows])
cal=np.isin(dz,C0); s1=np.isin(dz,R1); s2=np.isin(dz,R2)
def score(p,lab):
    out=[lab]
    for s,n in ((s1,"round 1"),(s2,"round 2")): out.append(f"{n}: MAE {np.abs(p[s]-y[s]).mean():5.2f} rho {stats.spearmanr(p[s],y[s])[0]:+.2f}")
    print("  "+" | ".join(out))
b,a=np.polyfit(x[cal],y[cal],1); p0=a+b*x; score(p0,"M0 linear (pre-registered form)        ")
p1=np.zeros_like(y)
for t in T:
    i=cal&(tw==t); bt=np.sum((x[i]-x[i].mean())*(y[i]-y[i].mean()))/np.sum((x[i]-x[i].mean())**2); w=.5; bs=w*bt+(1-w)*b; at=y[i].mean()-bs*x[i].mean(); p1[tw==t]=at+bs*x[tw==t]
score(p1,"M1 + town susceptibility (shrunk slope) ")
ceil=np.array([ownm[t] for t in tw]); f=lambda tau: np.sum((y[cal]-ceil[cal]*(1-np.exp(-x[cal]/tau)))**2); tau=optimize.minimize_scalar(f,bounds=(10,5000),method="bounded").x; p2=ceil*(1-np.exp(-x/tau)); score(p2,f"M2 saturating, town ceiling (tau={tau:.0f})   ")
X3=np.column_stack([np.ones(len(y)),x,x*fog,fog]); c3=np.linalg.lstsq(X3[cal],y[cal],rcond=None)[0]; p3=X3@c3; score(p3,"M3 family-aware (fog vs not)           ")
p4=np.zeros_like(y)
for t in T:
    i=cal&(tw==t); Xi=X3[i]; ci=np.linalg.lstsq(Xi,y[i],rcond=None)[0]; cs=.5*ci+.5*c3; p4[tw==t]=X3[tw==t]@cs
score(p4,"M4 town susceptibility + family-aware   ")
print(f"\n  predict-the-calibration-mean: round 1 MAE {np.abs(y[cal].mean()-y[s1]).mean():.2f}, round 2 {np.abs(y[cal].mean()-y[s2]).mean():.2f}")
for nm,p in (("M0",p0),("M4",p4)):
    print(f"  {nm} by dose (round 1+2, town mean obs/pred): "+", ".join(f"{d[4:]} {y[dz==d].mean():.1f}/{p[dz==d].mean():.1f}" for d in R1+R2))
