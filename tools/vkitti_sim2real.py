#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ledger §15.67 E2b: content-matched sim->real on Virtual KITTI 2 clone vs real KITTI tracking (same sequences).
mode distances: clone->real distances per scene on the SAME scale as dose_metrics.py (object features standardised by the clone probe's
  training-set mean/sd; clone objects = the same 3000-object subset (rng 42); real objects unpaired) -> results/calib_vkitti/sim2real_distances.json
mode predict: VKITTI2 weather calibration line (seed-mean clone-detector Car-AP drop ~ object FID, 25 pairs) -> predicted sim->real Car-AP drop per
  scene with 90% PI -> results/calib_vkitti/predictions_sim2real.json (+sha256). Refuses if any clone-on-real eval exists."""
import argparse, glob, json, os, sys, time, hashlib
from pathlib import Path
import numpy as np, torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
R="results/calib_vkitti"; SC=["Scene01","Scene02","Scene06","Scene18","Scene20"]; VARS=["d20_morning","d21_sunset","d22_overcast","d23_fog","d24_rain"]
def distances():
    from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
    from s2rgap.data.carla import CARLADetection
    from dose_metrics import objs
    from domain_matrix_probe import gather, CARLA_IMG_HW
    from domain_distance_matrix import fid, _cap
    from latent_geometry import knn_radius
    dev=torch.device("cuda"); ls=SpatialFeatureLookup.from_npz("/mnt/features/vkitti2/spatial_features_dinov3.npz"); lr=SpatialFeatureLookup.from_npz("/mnt/features/vkitti2_real/spatial_features_dinov3.npz"); out={}
    for s in SC:
        ref=f"{s}_d00_ref"; dsr=CARLADetection(root="/mnt/features/vkitti2",cells=[ref]); n=len(dsr); Fr,Or=objs(ls,dsr,range(n))
        rng=np.random.default_rng(42); sel=np.sort(rng.choice(len(Or),min(3000,len(Or)),replace=False))
        Xp,yp=gather(ls,torch.utils.data.Subset(dsr,range(0,800 if n>=1000 else int(0.75*n))),lambda d,i:CARLA_IMG_HW,np.random.default_rng(7),4,ref+" probe"); mu,sd=Xp.mean(0),Xp.std(0)+1e-6
        O0=(Or[sel]-mu)/sd; S=torch.from_numpy(O0).to(dev); rS=knn_radius(S)
        dsx=CARLADetection(root="/mnt/features/vkitti2_real",cells=[ref]); Fx,Ox=objs(lr,dsx,range(len(dsx)))
        sx=np.sort(np.random.default_rng(43).choice(len(Ox),min(3000,len(Ox)),replace=False)); O1=(Ox[sx]-mu)/sd; Tt=torch.from_numpy(O1).to(dev); D=torch.cdist(Tt,S); dm=D.min(1).values
        out[s]={"obj_fid_ref":float(fid(O0,O1)),"nnd_dose_to_ref":float(dm.median()/rS.median()),"cov_dose_by_ref":float((dm<=rS[D.argmin(1)]).float().mean()),
                "img_fid_ref":float(fid(_cap(Fr,1),_cap(Fx,2))),"n_obj_clone":int(len(sel)),"n_obj_real":int(len(sx)),"n_frames_real":len(dsx)}
        print(f"[sim2real] {s} "+" ".join(f"{k}={v:.3f}" for k,v in out[s].items() if not k.startswith("n_")),flush=True)
    json.dump(out,open(f"{R}/sim2real_distances.json","w"),indent=1)
def ap(f): return json.load(open(f))["metrics"]["AP_Car"]*100
def predict():
    if glob.glob(f"{R}/eval_*__on__*_real.json") or glob.glob(f"{R}_rep/eval_*__on__*_real.json"): sys.exit("REFUSING: clone-on-real evals exist")
    M=json.load(open(f"{R}/dose_metrics.json")); Dd=json.load(open(f"{R}/sim2real_distances.json"))
    def drop(Rr,s,v): return ap(f"{Rr}/eval_{s}_d00_ref__on__{s}_d00_ref.json")-ap(f"{Rr}/eval_{s}_d00_ref__on__{s}_{v}.json")
    out={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"ledger":"15.67 E2b","models":{},"scenes":{}}
    y=np.array([(drop(R,s,v)+drop(R+"_rep",s,v))/2 for s in SC for v in VARS]); out["weather_mean_drop"]=float(y.mean())
    for k in ("obj_fid_ref","nnd_dose_to_ref","cov_dose_by_ref","img_fid_ref"):
        x=np.array([M[f"{s}_{v}"][k] for s in SC for v in VARS]); n=len(x); b,a=np.polyfit(x,y,1); sres=np.sqrt(((y-a-b*x)**2).sum()/(n-2)); xb=x.mean(); sxx=((x-xb)**2).sum(); tq=stats.t.ppf(.95,n-2)
        out["models"][k]={"intercept":float(a),"slope":float(b),"resid_sd":float(sres),"spearman_cal":float(stats.spearmanr(x,y)[0]),"role":"primary" if k=="obj_fid_ref" else "secondary","x_range_cal":[float(x.min()),float(x.max())]}
        for s in SC:
            x0=Dd[s][k]; yh=a+b*x0; hw=tq*sres*np.sqrt(1+1/n+(x0-xb)**2/sxx); out["scenes"].setdefault(s,{})[k]={"x":float(x0),"pred":float(yh),"pi90":[float(yh-hw),float(yh+hw)]}
    blob=json.dumps(out,indent=1,sort_keys=True); open(f"{R}/predictions_sim2real.json","w").write(blob); h=hashlib.sha256(blob.encode()).hexdigest()
    open(f"{R}/predictions_sim2real.sha256","w").write(f"{h}  predictions_sim2real.json  {out['created']}\n"); print(f"[sim2real] predictions written {out['created']} sha256 {h}")
    for s in SC: print(f"  {s}: object FID {out['scenes'][s]['obj_fid_ref']['x']:.1f} (calibration range {out['models']['obj_fid_ref']['x_range_cal'][0]:.1f}-{out['models']['obj_fid_ref']['x_range_cal'][1]:.1f}) -> predicted drop {out['scenes'][s]['obj_fid_ref']['pred']:.1f} pp")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["distances","predict"],required=True); a=p.parse_args()
    distances() if a.mode=="distances" else predict()
