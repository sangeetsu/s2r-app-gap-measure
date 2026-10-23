#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LABEL-FREE task-conditional gap: dense per-patch probe + ATC on unlabeled target frames.

dIV_fg (ablation_log §15.21) needs target GT boxes to score the probe. ATC (Garg et al.,
ICLR 2022, arXiv:2201.04234) removes that: learn a confidence threshold tau on SOURCE held-out
patches such that P(conf>tau) equals source held-out accuracy; on the TARGET, the fraction of
patches with conf>tau estimates target accuracy — no target labels, no target boxes. Here the
probe is per-patch (4-class, fg/bg) over frozen DINOv3 features, trained on source frames
0:800, thresholded on 800:1000, applied to ALL patches of target frames (held-out 800:1000
for CARLA cells; val for KITTI/Cityscapes).
  gap_atc(S->T) := acc_in(S) − ATC_estimate(S,T)        (label-free)
  gap_true(S->T) := acc_in(S) − true patch acc(S,T)      (uses labels; diagnostic only)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection
from dense_localization_probe import build_targets
CARLA_HW=(375,1242); CELLS=["Town01_ClearNoon","Town01_OvercastNoon","Town01_SoftRainNoon","Town01_ClearNight","Town03_ClearNoon","Town03_OvercastNoon","Town03_SoftRainNoon","Town03_ClearNight"]
def log(m): print(f"[atc] {m}",flush=True)

def patches(lookup, ds, idxs, hw_fn, stride=7, cap=120000, seed=0):
    X,Y=[],[]; gh,gw=lookup.target_hw
    for i in idxs:
        _,t=ds[i]; rel=t["frame_rel"]
        if not lookup.has(rel): continue
        z=np.array(lookup.z_mmap[lookup.path_to_idx[rel]],dtype=np.float16)
        cls,_=build_targets(t["boxes"].numpy(),t["labels"].numpy(),hw_fn(ds,i),(gh,gw))
        fg=cls>0; bg=~fg; keep=np.zeros_like(bg); keep.flat[::stride]=True; sel=fg|(bg&keep)
        X.append(z[sel]); Y.append(cls[sel])
    X=np.concatenate(X); Y=np.concatenate(Y)
    if len(X)>cap:
        r=np.random.default_rng(seed); k=r.choice(len(X),cap,replace=False); X,Y=X[k],Y[k]
    return X,Y

class MLP(nn.Module):
    def __init__(s,d): super().__init__(); s.a=nn.Linear(d,512); s.b=nn.Linear(512,4)
    def forward(s,x): return s.b(F.dropout(F.relu(s.a(x)),0.1,s.training))

def fit(X,Y,mu,sd,dev,ep=15,seed=42):
    torch.manual_seed(seed); m=MLP(X.shape[1]).to(dev); o=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    mu_t,sd_t=torch.from_numpy(mu).to(dev),torch.from_numpy(sd).to(dev)
    dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X),torch.from_numpy(Y)),batch_size=4096,shuffle=True)
    for _ in range(ep):
        m.train()
        for xb,yb in dl: xb=(xb.to(dev).float()-mu_t)/sd_t; o.zero_grad(); F.cross_entropy(m(xb),yb.to(dev)).backward(); o.step()
    m.eval(); return m
def predict(m,X,mu,sd,dev):
    mu_t,sd_t=torch.from_numpy(mu).to(dev),torch.from_numpy(sd).to(dev); P=[]
    with torch.no_grad():
        for i in range(0,len(X),16384):
            P.append(F.softmax(m((torch.from_numpy(X[i:i+16384]).to(dev).float()-mu_t)/sd_t),1).cpu().numpy())
    return np.concatenate(P)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--carla-features",default=str(Path.home()/"datasets/carla_extracted/spatial_features_dinov3.npz"))
    p.add_argument("--kitti-features",default=str(Path.home()/"datasets/kitti/spatial_features_dinov3.npz"))
    p.add_argument("--cs-features",default=str(Path.home()/"datasets/cityscapes/spatial_features_dinov3_val.npz"))
    p.add_argument("--carla-root",default=str(Path.home()/"datasets/carla_extracted")); p.add_argument("--kitti-root",default=str(Path.home()/"datasets/kitti/extracted/training"))
    p.add_argument("--kitti-splits",default=str(Path.home()/"datasets/kitti/splits.json")); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--out",default="results/calib_anchored/dense_probe_atc_s42.json"); a=p.parse_args()
    dev=torch.device("cuda"); lc=SpatialFeatureLookup.from_npz(a.carla_features)
    src_tr,held={},{}
    for c in CELLS:
        ds=CARLADetection(root=a.carla_root,cells=[c]); n=len(ds)
        src_tr[c]=patches(lc,ds,range(0,800),lambda d,i:CARLA_HW,seed=a.seed); held[c]=patches(lc,ds,range(800,n),lambda d,i:CARLA_HW,seed=a.seed)
        log(f"{c}: train {len(src_tr[c][0])} held {len(held[c][0])} patches")
    lk=SpatialFeatureLookup.from_npz(a.kitti_features); kv=KITTIDetection(kitti_root=a.kitti_root,split_json=a.kitti_splits,split="val"); kimg=Path(a.kitti_root)/"image_2"
    def khw(d,i):
        with Image.open(kimg/d.frame_names[i]) as im: return (im.height,im.width)
    held["KITTI"]=patches(lk,kv,range(len(kv)),khw,seed=a.seed); log(f"KITTI: {len(held['KITTI'][0])} patches")
    try:
        from s2rgap.data.cityscapes_coco import CityscapesCOCO
        lcs=SpatialFeatureLookup.from_npz(a.cs_features); cs=CityscapesCOCO(Path.home()/"datasets/cityscapes/extracted/leftImg8bit",REPO/"nvme/datasets/aldi_labels/cityscapes_val_instances.json",split="val")
        held["Cityscapes"]=patches(lcs,cs,range(len(cs)),lambda d,i:(1024,2048),seed=a.seed); log(f"Cityscapes: {len(held['Cityscapes'][0])} patches")
    except Exception as e: log(f"Cityscapes skipped: {e}")
    out={}
    for S in CELLS:
        X,Y=src_tr[S]; mu=X[:50000].astype(np.float32).mean(0); sd=X[:50000].astype(np.float32).std(0)+1e-6
        m=fit(X,Y,mu,sd,dev,seed=a.seed)
        Ph=predict(m,held[S][0],mu,sd,dev); acc_in=float((Ph.argmax(1)==held[S][1]).mean()); conf_h=Ph.max(1)
        tau=float(np.quantile(conf_h,1-acc_in))              # ATC threshold: P(conf>tau)=acc_in on source held-out
        row={"acc_in":acc_in,"tau":tau}
        for T,(Xt,Yt) in held.items():
            if T==S: continue
            Pt=predict(m,Xt,mu,sd,dev); est=float((Pt.max(1)>tau).mean()); tru=float((Pt.argmax(1)==Yt).mean())
            row[T]={"atc_est_acc":est,"true_acc":tru,"gap_atc":acc_in-est,"gap_true":acc_in-tru}
        out[S]=row; log(f"{S:<22} in {acc_in*100:4.1f} tau {tau:.2f} | KITTI atc {row['KITTI']['atc_est_acc']*100:4.1f} true {row['KITTI']['true_acc']*100:4.1f}" + (f" | CS atc {row['Cityscapes']['atc_est_acc']*100:4.1f} true {row['Cityscapes']['true_acc']*100:4.1f}" if "Cityscapes" in row else ""))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,"w"),indent=1); log(f"saved -> {a.out}")
if __name__=="__main__": raise SystemExit(main())
