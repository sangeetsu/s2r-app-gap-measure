#!/usr/bin/env python3
"""Score the pre-registered dose-response rules (ledger §15.47): R1 monotone appearance-only drop, R2 metric validity, R3 real transfer,
asymmetry. Within-CARLA drop = ref detector on ref held-out minus on dose held-out (frames 800:1000, identical content)."""
import json, glob, os, sys, numpy as np
from scipy import stats
R=os.environ.get("DOSE_DIR","results/calib_dose"); MF=sys.argv[1] if len(sys.argv)>1 else f"{R}/dose_metrics.json"; M=json.load(open(MF)) if os.path.exists(MF) else {}
print(f"[analyze_dose] results {R}; metrics {MF}")
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
def mm(f): return m(f) if os.path.exists(f) else np.nan
TOWNS=sorted({f.split("eval_")[1].split("_d00_ref__on__")[0] for f in glob.glob(f"{R}/eval_*_d00_ref__on__*_d00_ref.json")}); DOSES=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
FAM={"sun":DOSES[:4],"fog":DOSES[4:7]}
drop={}; 
for t in TOWNS:
    ref=f"{t}_d00_ref"; a=mm(f"{R}/eval_{ref}__on__{ref}.json")
    for d in DOSES: drop[(t,d)]=a-mm(f"{R}/eval_{ref}__on__{t}_{d}.json")
print("=== R1: appearance-only drop of the reference detector (pp), frames identical ===")
print(f"  {'town':<9}"+"".join(f"{d[4:]:>8}" for d in DOSES))
for t in TOWNS: print(f"  {t:<9}"+"".join(f"{drop[(t,d)]:8.1f}" for d in DOSES))
for fam,ds in FAM.items():
    rhos=[stats.spearmanr(range(len(ds)),[drop[(t,d)] for d in ds])[0] for t in TOWNS if not any(np.isnan(drop[(t,d)]) for d in ds)]
    need=max(2,int(np.ceil(0.75*len(rhos)))); print(f"  {fam}: Spearman(dose order, drop) per town {np.round(rhos,2).tolist()} -> {sum(r>=.8-1e-9 for r in rhos)}/{len(rhos)} towns >= .8 (need {need}) -> R1 {'PASS' if sum(r>=.8-1e-9 for r in rhos)>=need else 'FAIL'}")
if M:
    keys=[k for k in next(iter(M.values())) if k not in ("n_obj","n_boxes","fid_KITTI","fid_Cityscapes")]
    pairs=[(t,d) for t in TOWNS for d in DOSES if f"{t}_{d}" in M and not np.isnan(drop[(t,d)])]; y=np.array([drop[p] for p in pairs]); tw=np.array([p[0] for p in pairs])
    print(f"\n=== R2: label-free metric(ref, dose) vs appearance-only drop, {len(pairs)} pairs ===")
    for k in keys:
        x=np.array([M[f"{t}_{d}"][k] for t,d in pairs]); rho,p=stats.spearmanr(x,y); e=[]
        for h in TOWNS:
            te=tw==h; tr=~te
            if te.sum()==0: continue
            b=np.polyfit(x[tr],y[tr],1); e+=list((y[te]-np.polyval(b,x[te]))**2)
        r2=1-np.mean(e)/np.mean((y-y.mean())**2)
        print(f"  {k:<18} Spearman {rho:+.3f} (p={p:.1e})  LOTO R2 {r2:+.3f}  -> R2 {'PASS' if abs(rho)>=.7 and r2>0 else 'FAIL'}   (|rho|; sign-agnostic since 2026-10-16)")
print("\n=== R3: real mAP, dose-trained vs reference-trained detector (same town) ===")
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    rows=[]
    for t in TOWNS:
        r0=mm(f"{R}/src_{t}_d00_ref_{key}.json")
        for d in DOSES: rows.append((t,d,mm(f"{R}/src_{t}_{d}_{key}.json")-r0))
    print(f"  {T}: Δ real mAP (dose - ref) by dose, mean over towns: "+", ".join(f"{d[4:]} {np.nanmean([r[2] for r in rows if r[1]==d]):+.1f}" for d in DOSES))
    if M and all(f"fid_{T}" in v for v in M.values()):
        ok=[(t,d,v) for t,d,v in rows if f"{t}_{d}" in M and not np.isnan(v)]
        if len(ok)>=8:
            dx=np.array([M[f"{t}_{d}"][f"fid_{T}"]-M[f"{t}_d00_ref"][f"fid_{T}"] for t,d,_ in ok]); dy=np.array([v for *_,v in ok]); rho=stats.spearmanr(dx,dy)[0]
            rng=np.random.default_rng(0); twn=np.array([t for t,_,_ in ok]); null=[]
            for _ in range(2000):
                perm=dy.copy()
                for h in TOWNS: idx=np.where(twn==h)[0]; perm[idx]=rng.permutation(dy[idx])
                null.append(stats.spearmanr(dx,perm)[0])
            print(f"    Δ FID-to-{T} vs Δ real mAP: Spearman {rho:+.3f}, town-clustered permutation p={(1+np.sum(np.abs(null)>=abs(rho)))/2001:.4f} (n={len(ok)})")
print("\n=== Asymmetry: ref-trained on dose vs dose-trained on ref (drop from each detector's own held-out) ===")
agree=0; n=0
for t in TOWNS:
    for d in DOSES:
        c=f"{t}_{d}"; ref=f"{t}_d00_ref"
        a1=mm(f"{R}/eval_{ref}__on__{ref}.json")-mm(f"{R}/eval_{ref}__on__{c}.json"); a2=mm(f"{R}/eval_{c}__on__{c}.json")-mm(f"{R}/eval_{c}__on__{ref}.json")
        if np.isnan(a1) or np.isnan(a2): continue
        print(f"  {c:<22} ref->dose {a1:+6.1f}   dose->ref {a2:+6.1f}   harder: {'dose->ref' if a2>a1 else 'ref->dose'}")
