#!/usr/bin/env python3
"""Score the pre-registered Foggy Cityscapes rules F1-F3 (ledger §15.59)."""
import json, os, numpy as np
from scipy import stats
R="results/calib_dose_foggy"; D="results/calib_dose"; M=json.load(open(f"{D}/dose_metrics.json")); FF=json.load(open(f"{R}/fid_to_foggy.json")) if os.path.exists(f"{R}/fid_to_foggy.json") else {}
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
def mm(f): return m(f) if os.path.exists(f) else np.nan
T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]; DOSES=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
print("=== F1: simulated fog training vs real fog (Foggy Cityscapes val); mAP, town mean; Δ = fog-trained − ref-trained ===")
print(f"  {'beta':>6}  {'ref':>6}  {'fog25':>6} {'fog50':>6} {'fog75':>6} | Δfog75  towns Δ>0")
for b in ("0","0.005","0.01","0.02"):
    key=lambda c: f"{D}/src_{c}_cityscapesval.json" if b=="0" else f"{R}/src_{c}_foggy{b}.json"
    ref=np.array([mm(key(f"{t}_d00_ref")) for t in T]); f25=np.array([mm(key(f"{t}_d05_fog25")) for t in T]); f50=np.array([mm(key(f"{t}_d06_fog50")) for t in T]); f75=np.array([mm(key(f"{t}_d07_fog75")) for t in T])
    d75=f75-ref; print(f"  {b:>6}  {np.nanmean(ref):6.1f}  {np.nanmean(f25):6.1f} {np.nanmean(f50):6.1f} {np.nanmean(f75):6.1f} | {np.nanmean(d75):+6.1f}  {int(np.nansum(d75>0))}/{int(np.sum(~np.isnan(d75)))}")
d02=np.array([mm(f"{R}/src_{t}_d07_fog75_foggy0.02.json")-mm(f"{R}/src_{t}_d00_ref_foggy0.02.json") for t in T]); ok=~np.isnan(d02)
if ok.sum()>=6:
    k=int((d02[ok]>0).sum()); n=int(ok.sum()); p=stats.binomtest(k,n,0.5,alternative="greater").pvalue
    means=[np.nanmean([mm(f"{D}/src_{t}_d07_fog75_cityscapesval.json")-mm(f"{D}/src_{t}_d00_ref_cityscapesval.json") for t in T])]+[np.nanmean([mm(f"{R}/src_{t}_d07_fog75_foggy{b}.json")-mm(f"{R}/src_{t}_d00_ref_foggy{b}.json") for t in T]) for b in ("0.005","0.01","0.02")]
    mono=all(means[i]<=means[i+1] for i in range(3)); print(f"  F1: fog75 beats ref on beta=0.02 in {k}/{n} towns (sign test p={p:.3f}); Δ by beta 0/.005/.01/.02 = {', '.join(f'{v:+.1f}' for v in means)} monotone={mono} -> {'PASS' if k>=6 and mono else 'FAIL'}")
if FF:
    pairs=[(t,d) for t in T for d in DOSES if os.path.exists(f"{R}/src_{t}_{d}_foggy0.02.json")]; tw=np.array([t for t,_ in pairs])
    dy=np.array([m(f"{R}/src_{t}_{d}_foggy0.02.json")-m(f"{R}/src_{t}_d00_ref_foggy0.02.json") for t,d in pairs])
    dxf=np.array([FF[f"{t}_{d}"]["fid_foggy_0.02"]-FF[f"{t}_d00_ref"]["fid_foggy_0.02"] for t,d in pairs]); dxc=np.array([M[f"{t}_{d}"]["fid_Cityscapes"]-M[f"{t}_d00_ref"]["fid_Cityscapes"] for t,d in pairs])
    rng=np.random.default_rng(0)
    def cp(dx):
        rho=stats.spearmanr(dx,dy)[0]; null=[]
        for _ in range(2000):
            p_=dy.copy()
            for t in T: i=np.where(tw==t)[0]; p_[i]=rng.permutation(dy[i])
            null.append(stats.spearmanr(dx,p_)[0])
        return rho,(1+np.sum(np.abs(null)>=abs(rho)))/2001
    rf,pf=cp(dxf); rc,pc=cp(dxc)
    print(f"\n=== F2/F3 ({len(pairs)} pairs): Δ FID-to-FOGGY vs Δ foggy mAP: Spearman {rf:+.3f} (clustered p={pf:.4f}) -> F2 {'PASS' if rf<0 and pf<.05 else 'FAIL'}")
    print(f"  Δ FID-to-CLEAR-CS vs Δ foggy mAP: Spearman {rc:+.3f} (p={pc:.4f}); F3 (foggy-FID predicts better than clear-FID): {'PASS' if abs(rf)>abs(rc) else 'FAIL'}")
