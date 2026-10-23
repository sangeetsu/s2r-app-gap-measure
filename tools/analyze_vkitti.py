#!/usr/bin/env python3
"""Score ledger §15.67: V1 appearance cost, V2 metric validity (25 pairs, leave-one-scene-out), V3 CARLA calibration transfer, V4 real transfer,
asymmetry; E2b S1 sim->real prediction. All Car AP@0.5."""
import json, os, numpy as np
from scipy import stats
R="results/calib_vkitti"; SC=["Scene01","Scene02","Scene06","Scene18","Scene20"]; V=["d20_morning","d21_sunset","d22_overcast","d23_fog","d24_rain"]
def ap(f): return json.load(open(f))["metrics"]["AP_Car"]*100 if os.path.exists(f) else np.nan
def drop(Rr,s,v): return ap(f"{Rr}/eval_{s}_d00_ref__on__{s}_d00_ref.json")-ap(f"{Rr}/eval_{s}_d00_ref__on__{s}_{v}.json")
P=[(s,v) for s in SC for v in V]; G=np.array([s for s,_ in P])
y42=np.array([drop(R,s,v) for s,v in P]); y17=np.array([drop(R+"_rep",s,v) for s,v in P]); y=np.nanmean([y42,y17],0)
print("=== V1: clone-detector held-out Car-AP drop (pp), seed mean ===")
print(f"  {'scene':<9}"+"".join(f"{v[4:]:>10}" for v in V)+"   own AP (s42/s17)")
for s in SC: print(f"  {s:<9}"+"".join(f"{(drop(R,s,v)+drop(R+'_rep',s,v))/2:10.1f}" for v in V)+f"   {ap(f'{R}/eval_{s}_d00_ref__on__{s}_d00_ref.json'):.1f}/{ap(f'{R}_rep/eval_{s}_d00_ref__on__{s}_d00_ref.json'):.1f}")
print("  mean    "+"".join(f"{np.nanmean([y[i] for i,(s_,v_) in enumerate(P) if v_==v]):10.1f}" for v in V))
k=sum(np.nanmean([y[i] for i,(s_,_) in enumerate(P) if s_==s])>0 for s in SC); print(f"  scenes with mean drop > 0: {k}/5 -> V1 {'PASS' if k>=4 else 'FAIL'}; seed agreement r={stats.pearsonr(y42,y17)[0]:.3f}, mean |Δseed| {np.nanmean(np.abs(y42-y17)):.2f}")
M=json.load(open(f"{R}/dose_metrics.json")) if os.path.exists(f"{R}/dose_metrics.json") else {}; L=json.load(open(f"{R}/dose_lowlevel.json")) if os.path.exists(f"{R}/dose_lowlevel.json") else {}
def loso(x,yy):
    e=[]
    for h in SC: te=G==h; b=np.polyfit(x[~te],yy[~te],1); e+=list((yy[te]-np.polyval(b,x[te]))**2)
    return 1-np.mean(e)/np.mean((yy-yy.mean())**2)
def cperm(x,yy):
    rho=stats.spearmanr(x,yy)[0]; rng=np.random.default_rng(0); null=[]
    for _ in range(2000):
        xp=x.copy()
        for h in SC: i=np.where(G==h)[0]; xp[i]=rng.permutation(x[i])
        null.append(stats.spearmanr(xp,yy)[0])
    return rho,(1+np.sum(np.abs(null)>=abs(rho)))/2001
print("\n=== V2: metric(clone render, variant render) vs drop, 25 pairs ===")
for src,k,role in [(M,"obj_fid_ref","PRIMARY"),(M,"nnd_dose_to_ref",""),(M,"cov_dose_by_ref",""),(M,"paired_obj_disp",""),(M,"probe_flip",""),(M,"img_fid_ref",""),(L,"lpips",""),(L,"incfid_ref","")]:
    if not src: continue
    x=np.array([src[f"{s}_{v}"][k] for s,v in P]); rho,p=cperm(x,y); r2=loso(x,y)
    print(f"  {k:<18} rho {rho:+.3f} (scene-clustered p={p:.4f}) LOSO R2 {r2:+.3f}"+(f" -> V2 {'PASS' if abs(rho)>=.7 and r2>0 else 'FAIL'}" if role else ""))
if M:
    C=json.load(open("results/calib_dose/dose_metrics.json")); TW=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]; DD=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
    def cap(Rr,t,d): return ap(f"{Rr}/eval_{t}_d00_ref__on__{t}_d00_ref.json")-ap(f"{Rr}/eval_{t}_d00_ref__on__{t}_{d}.json")
    yc=np.array([(cap("results/calib_dose",t,d)+cap("results/calib_dose_rep",t,d))/2 for t in TW for d in DD]); xc=np.array([C[f"{t}_{d}"]["obj_fid_ref"] for t in TW for d in DD]); b=np.polyfit(xc,yc,1)
    x=np.array([M[f"{s}_{v}"]["obj_fid_ref"] for s,v in P]); pr=np.polyval(b,x); mae=np.nanmean(np.abs(pr-y)); mae0=np.nanmean(np.abs(yc.mean()-y))
    print(f"\n=== V3: CARLA Car-AP calibration (object FID; 72 pairs; slope {b[0]:.3f}) applied to VKITTI2: MAE {mae:.2f} vs CARLA-mean {mae0:.2f} -> V3 {'PASS' if mae<mae0 else 'FAIL'}; rho(pred,obs) {stats.spearmanr(pr,y)[0]:+.3f}")
    print(f"    object FID ranges: CARLA {xc.min():.1f}-{xc.max():.1f}; VKITTI2 {x.min():.1f}-{x.max():.1f}")
print("\n=== V4: real transfer (KITTI val Car AP), dose-trained - clone-trained ===")
dk=np.array([ap(f"{R}/src_{s}_{v}_kittival.json")-ap(f"{R}/src_{s}_d00_ref_kittival.json") for s,v in P])
print("  by variant: "+", ".join(f"{v[4:]} {np.nanmean(dk[[i for i,(s_,v_) in enumerate(P) if v_==v]]):+.1f}" for v in V)+f"; clone-trained KITTI Car AP {np.nanmean([ap(f'{R}/src_{s}_d00_ref_kittival.json') for s in SC]):.1f}")
if M and not np.all(np.isnan(dk)):
    dx=np.array([M[f"{s}_{v}"]["fid_KITTI"]-M[f"{s}_d00_ref"]["fid_KITTI"] for s,v in P]); ok=~np.isnan(dk); rho=stats.spearmanr(dx[ok],dk[ok])[0]; rng=np.random.default_rng(0); null=[]
    for _ in range(2000):
        yy=dk.copy()
        for h in SC: i=np.where((G==h)&ok)[0]; yy[i]=rng.permutation(dk[i])
        null.append(stats.spearmanr(dx[ok],yy[ok])[0])
    p=(1+np.sum(np.abs(null)>=abs(rho)))/2001; print(f"  Δ FID-to-KITTI vs Δ KITTI Car AP: rho {rho:+.3f}, scene-clustered p={p:.4f} (n={ok.sum()}) -> V4 {'PASS' if rho<0 and p<.05 else 'FAIL'}")
print("\n=== Asymmetry: clone-trained on variant vs variant-trained on clone (each vs own held-out) ===")
h=0;n=0
for s,v in P:
    a1=drop(R,s,v); a2=ap(f"{R}/eval_{s}_{v}__on__{s}_{v}.json")-ap(f"{R}/eval_{s}_{v}__on__{s}_d00_ref.json")
    if np.isnan(a1) or np.isnan(a2): continue
    n+=1; h+=a1>a2
print(f"  clone->variant harder in {h}/{n} pairs")
if os.path.exists(f"{R}/predictions_sim2real.json"):
    Q=json.load(open(f"{R}/predictions_sim2real.json")); print(f"\n=== E2b S1: sim->real (clone detector on real KITTI tracking frames, same sequences); predictions {Q['created']} ===")
    errs=[];inside=0;rows=[]
    for s in SC:
        o=np.nanmean([ap(f"{Rr}/eval_{s}_d00_ref__on__{s}_d00_ref.json")-ap(f"{Rr}/eval_{s}_d00_ref__on__{s}_real.json") for Rr in (R,R+"_rep")]); q=Q["scenes"][s]["obj_fid_ref"]
        if np.isnan(o): continue
        errs.append(abs(q["pred"]-o)); inside+=q["pi90"][0]<=o<=q["pi90"][1]; rows.append((s,o,q))
        rr=ap(f"{R}/eval_{s}_real__on__{s}_real.json")-ap(f"{R}/eval_{s}_real__on__{s}_d00_ref.json")
        print(f"  {s}: object FID clone->real {q['x']:.1f}; predicted drop {q['pred']:.1f} [{q['pi90'][0]:.1f}, {q['pi90'][1]:.1f}]; observed {o:.1f}; real-trained on clone drop {rr:.1f}")
    if errs:
        mae0=np.mean([abs(Q['weather_mean_drop']-o) for _,o,_ in rows]); print(f"  inside 90% PI {inside}/{len(errs)}; MAE {np.mean(errs):.2f} vs weather-mean {mae0:.2f} -> S1 {'PASS' if inside>=4 and np.mean(errs)<mae0 else 'FAIL'}")
