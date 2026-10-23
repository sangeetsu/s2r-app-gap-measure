#!/usr/bin/env python3
"""Score ledger §15.65 (E1) held-out-dose prediction: H1 rank, H2 accuracy, H3 interval coverage; subsets; compound-dose baselines."""
import json, os, glob, numpy as np
from scipy import stats
HO="results/calib_dose_ho"; P=json.load(open(f"{HO}/predictions.json")); T=sorted({v["town"] for v in P["cells"].values()})
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
def mm(f): return m(f) if os.path.exists(f) else np.nan
def own(R,t): return m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json")
def obs(c,t): return np.nanmean([own("results/calib_dose",t)-mm(f"{HO}/eval_{t}_d00_ref__on__{c}.json"),own("results/calib_dose_rep",t)-mm(f"{HO}_rep/eval_{t}_d00_ref__on__{c}.json")])
def caldrop(t,d): return np.mean([own(R,t)-m(f"{R}/eval_{t}_d00_ref__on__{t}_{d}.json") for R in ("results/calib_dose","results/calib_dose_rep")])
ev=sorted(glob.glob(f"{HO}/eval_*.json")+glob.glob(f"{HO}_rep/eval_*.json"),key=os.path.getmtime)
print(f"predictions written {P['created']} (mtime {os.path.getmtime(HO+'/predictions.json'):.0f}); first held-out eval file mtime {os.path.getmtime(ev[0]):.0f} -> order {'OK' if os.path.getmtime(ev[0])>os.path.getmtime(HO+'/predictions.json') else 'VIOLATED'}")
cells=sorted(P["cells"]); y=np.array([obs(c,P["cells"][c]["town"]) for c in cells]); ok=~np.isnan(y); sub=np.array([P["cells"][c]["subset"] for c in cells])
print(f"{ok.sum()} held-out pairs with observations; calibration mean drop {P['calibration_mean_drop']:.2f} pp")
for k,mod in P["models"].items():
    pr=np.array([P["cells"][c][k]["pred"] for c in cells]); lo=np.array([P["cells"][c][k]["pi90"][0] for c in cells]); hi=np.array([P["cells"][c][k]["pi90"][1] for c in cells])
    rho=stats.spearmanr(pr[ok],y[ok])[0]; mae=np.abs(pr-y)[ok].mean(); mae0=np.abs(P["calibration_mean_drop"]-y)[ok].mean(); cov=((y>=lo)&(y<=hi))[ok].mean()
    h1,h2,h3=rho>=.7,(mae<=7.0 and mae<mae0),cov>=.8
    print(f"\n=== {k} ({mod['role']}): H1 rho {rho:+.3f} {'PASS' if h1 else 'FAIL'} | H2 MAE {mae:.2f} (mean-pred {mae0:.2f}) {'PASS' if h2 else 'FAIL'} | H3 coverage {100*cov:.0f}% {'PASS' if h3 else 'FAIL'}")
    for s_ in ("interpolation","extrapolation","compound","novel"):
        i=ok&(sub==s_)
        if i.sum()>2: print(f"   {s_:<14} n={i.sum():2d}  rho {stats.spearmanr(pr[i],y[i])[0]:+.2f}  MAE {np.abs(pr-y)[i].mean():5.2f}  bias {np.mean(pr[i]-y[i]):+5.2f}  coverage {100*((y>=lo)&(y<=hi))[i].mean():3.0f}%")
print("\n=== per dose: observed drop (town mean) vs object-FID prediction")
for d in sorted({v["dose"] for v in P["cells"].values()}):
    i=np.array([P["cells"][c]["dose"]==d for c in cells])&ok; pr=np.array([P["cells"][c]["obj_fid_ref"]["pred"] for c in cells])
    print(f"  {d:<16} observed {y[i].mean():6.1f}  predicted {pr[i].mean():6.1f}  (towns {i.sum()})")
print("\n=== compound doses: object-FID prediction vs label-using baselines (additive / max of measured component drops)")
COMP={"d14_night_rain":("d04_night","d09_rain"),"d15_night_fog50":("d04_night","d06_fog50"),"d16_sun10_wet":("d02_sun10","d08_wet")}
for d,(a_,b_) in COMP.items():
    rows=[(obs(f"{t}_{d}",t),P["cells"][f"{t}_{d}"]["obj_fid_ref"]["pred"],caldrop(t,a_)+caldrop(t,b_),max(caldrop(t,a_),caldrop(t,b_))) for t in T if f"{t}_{d}" in P["cells"]]
    r=np.array(rows); r=r[~np.isnan(r[:,0])]
    if len(r): print(f"  {d:<16} MAE: metric {np.abs(r[:,1]-r[:,0]).mean():5.2f}  additive {np.abs(r[:,2]-r[:,0]).mean():5.2f}  max {np.abs(r[:,3]-r[:,0]).mean():5.2f}   (observed mean {r[:,0].mean():.1f})")
