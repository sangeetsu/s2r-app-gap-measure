#!/usr/bin/env python3
"""Ledger §15.69 (E3): FID from each dose cell to RTTS (mode fid) and scoring of G1/G2 (mode analyze)."""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
from scipy import stats
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
R="results/calib_dose_rtts"; T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]
D=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
def fid_mode():
    from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
    from s2rgap.data.carla import CARLADetection
    from domain_distance_matrix import fid, _cap
    from fid_to_real import img_feats
    lk=SpatialFeatureLookup.from_npz("/mnt/features/carla_doses/spatial_features_dinov3_80cells.npz"); lr=SpatialFeatureLookup.from_npz("/mnt/features/rtts_features/spatial_features_dinov3_rtts.npz")
    Fr=np.array([np.array(lr.z_mmap[i],dtype=np.float32).reshape(-1,lr.z_mmap.shape[-1]).mean(0) for i in range(len(lr.z_mmap))],np.float32); out={}
    for t in T:
        for d in ["d00_ref"]+D:
            c=f"{t}_{d}"; ds=CARLADetection(root="/mnt/features/carla_doses",cells=[c]); C=img_feats(lk,[r[2] for r in ds.records]); out[c]={"fid_RTTS":float(fid(_cap(Fr,1),_cap(C,2)))}; print(f"[rtts-fid] {c} {out[c]['fid_RTTS']:.2f}",flush=True)
    json.dump(out,open(f"{R}/fid_to_rtts.json","w"),indent=1)
def m(f): return json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
def mm(f): return m(f) if os.path.exists(f) else np.nan
def analyze():
    print("=== G1: RTTS mAP (Car+Pedestrian) of dose-trained detectors, town mean; clear Cityscapes for contrast ===")
    for seed,Rr in (("42",R),("17",R+"_rep")):
        v={d:np.array([mm(f"{Rr}/src_{t}_{d}_rtts.json") for t in T]) for d in ["d00_ref"]+D}
        print(f"  seed {seed}: "+"  ".join(f"{d[4:]} {np.nanmean(x):.1f}" for d,x in v.items() if not np.all(np.isnan(x))))
        d75=v["d07_fog75"]-v["d00_ref"]; ok=~np.isnan(d75)
        if ok.sum()>=6: k=int((d75[ok]>0).sum()); print(f"    fog75 - ref: town mean {np.nanmean(d75):+.2f}; positive in {k}/{ok.sum()} towns (sign test p={stats.binomtest(k,int(ok.sum()),.5,alternative='greater').pvalue:.3f})"+(f" -> G1 {'PASS' if k>=6 else 'FAIL'}" if seed=="42" else ""))
        for d in ("d05_fog25","d06_fog50"): dd=v[d]-v["d00_ref"]; print(f"    {d[4:]} - ref: {np.nanmean(dd):+.2f} ({int(np.nansum(dd>0))}/8 towns)")
    cs={d:np.nanmean([mm(f"results/calib_dose/src_{t}_{d}_cityscapesval.json") for t in T]) for d in ("d00_ref","d05_fog25","d06_fog50","d07_fog75")}; print("  clear Cityscapes (seed 42): "+"  ".join(f"{d[4:]} {x:.1f}" for d,x in cs.items()))
    F=json.load(open(f"{R}/fid_to_rtts.json")); P=[(t,d) for t in T for d in D]; tw=np.array([t for t,_ in P])
    dy=np.array([mm(f"{R}/src_{t}_{d}_rtts.json")-mm(f"{R}/src_{t}_d00_ref_rtts.json") for t,d in P]); dx=np.array([F[f"{t}_{d}"]["fid_RTTS"]-F[f"{t}_d00_ref"]["fid_RTTS"] for t,d in P]); ok=~np.isnan(dy)
    rho=stats.spearmanr(dx[ok],dy[ok])[0]; rng=np.random.default_rng(0); null=[]
    for _ in range(2000):
        y=dy.copy()
        for t in T: i=np.where((tw==t)&ok)[0]; y[i]=rng.permutation(dy[i])
        null.append(stats.spearmanr(dx[ok],y[ok])[0])
    p=(1+np.sum(np.abs(null)>=abs(rho)))/2001; print(f"\n=== G2 ({ok.sum()} pairs): Δ FID-to-RTTS vs Δ RTTS mAP Spearman {rho:+.3f}, town-clustered p={p:.4f} -> {'PASS' if rho<0 and p<.05 else 'FAIL'}")
    fam={"fog":D[4:7],"non-fog":[d for d in D if "fog" not in d]}
    for k,ds in fam.items():
        i=np.array([d in ds for _,d in P])&ok; print(f"  {k}: rho {stats.spearmanr(dx[i],dy[i])[0]:+.3f} (n={i.sum()})")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=["fid","analyze"]); a=p.parse_args(); fid_mode() if a.mode=="fid" else analyze()
