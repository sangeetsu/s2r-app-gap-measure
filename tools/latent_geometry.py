#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Latent geometry of the appearance gap in frozen DINOv3 space (2026-10-15; motivated by ledger §15.41).

Two hypotheses derived from our own anomalies, tested on cached features only:
 H1 ASYMMETRY. Transfer is asymmetric (Epic->Low detectors hold, Low->Epic lose 6-24 pp) but FID/MMD/CORAL are symmetric. The
    matching object is support COVERAGE: the fraction of the target's object features lying inside the source's k-NN support
    (the 'recall' of Kynkaanniemi et al. 2019 / 'coverage' of Naeem et al. 2020, here between SOURCE and TARGET object manifolds).
 H2 TASK PROJECTION. A linear hypothesis on frozen features only sees x projected on span(W). Shift inside the source probe's
    4-D discriminative subspace (task shift) should track the drop; shift in the 1020-D orthogonal complement (nuisance shift:
    haze, colour, exposure) should not. Raw FID mixes both, which is why it registers appearance without pricing it.
Per directed pair (S,T), object level (fg GT boxes, box-pooled, standardised with S statistics):
  fid_full, fid_task, fid_nuis, cov_T_by_S, cov_S_by_T, nnd_T_to_S, nnd_S_to_T (median NN distance / S's own k-NN radius),
  cov_task_T_by_S (coverage inside the task subspace).
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection
from linear_probe_object_class import bbox_to_feat_region
from domain_matrix_probe import gather, CARLA_IMG_HW
from domain_distance_matrix import fid
from sklearn.linear_model import LogisticRegression
def log(m): print(f"[geo] {m}",flush=True)
CAP=3000; K=3

def obj_feats(lookup, ds, idxs, rng, cap=CAP):
    gh,gw=lookup.target_hw; X=[]
    for i in idxs:
        im,t=ds[i]; rel=t["frame_rel"]
        if not lookup.has(rel): continue
        H,W=int(im.shape[-2]),int(im.shape[-1]); z=np.array(lookup.z_mmap[lookup.path_to_idx[rel]],dtype=np.float32)
        for b,l in zip(t["boxes"].numpy(),t["labels"].numpy()):
            if l<=0: continue
            i1,j1,i2,j2=bbox_to_feat_region(tuple(b),(H,W),(gh,gw)); p=z[i1:i2,j1:j2].reshape(-1,z.shape[-1])
            if len(p): X.append(p.mean(0))
    X=np.array(X,np.float32)
    if len(X)>cap: X=X[rng.choice(len(X),cap,replace=False)]
    return X

def knn_radius(S,k=K):
    d=torch.cdist(S,S); d.fill_diagonal_(float("inf")); return d.topk(k,largest=False).values[:,-1]
def coverage(T,S,rS):
    d=torch.cdist(T,S); m,idx=d.min(1); return float((m<=rS[idx]).float().mean()), float((m.median()/rS.median()).item())

def measures(Xs,Xt,mu,sd,Q,dev):
    s=(Xs-mu)/sd; t=(Xt-mu)/sd
    out={"fid_full":float(fid(s,t)),"fid_task":float(fid(s@Q,t@Q)),"fid_nuis":float(fid(s-(s@Q)@Q.T,t-(t@Q)@Q.T))}
    S=torch.from_numpy(s).to(dev); T=torch.from_numpy(t).to(dev); rS=knn_radius(S); rT=knn_radius(T)
    out["cov_T_by_S"],out["nnd_T_to_S"]=coverage(T,S,rS); out["cov_S_by_T"],out["nnd_S_to_T"]=coverage(S,T,rT)
    Sq=torch.from_numpy(s@Q).to(dev); Tq=torch.from_numpy(t@Q).to(dev); out["cov_task_T_by_S"],out["nnd_task_T_to_S"]=coverage(Tq,Sq,knn_radius(Sq))
    out["n_S"],out["n_T"]=int(len(s)),int(len(t)); return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",default="results/latent_geometry/pairs.json"); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); dev=torch.device("cuda"); rng=np.random.default_rng(a.seed); t0=time.time()
    roots={str(Path.home()/"datasets/carla_extracted"):str(Path.home()/"datasets/carla_extracted/spatial_features_dinov3.npz"),
           "/mnt/features/carla_new":"/mnt/features/carla_new/spatial_features_dinov3_16cells.npz",
           "/mnt/features/carla_lowq":sorted(glob.glob("/mnt/features/carla_lowq/spatial_features_dinov3_*cells.npz"))[-1]}
    dom={}   # name -> dict(train=(X,y) for probe, src=obj feats 0:800, tgt=obj feats 800:1000)
    for root,feat in roots.items():
        lk=SpatialFeatureLookup.from_npz(feat); tag="" if "lowq" not in root else "@low"
        for d in sorted(Path(root).iterdir()):
            if not (d.is_dir() and d.name.startswith("Town") and len(list(d.glob("frame_*.jpg")))>=1000): continue
            ds=CARLADetection(root=root,cells=[d.name]); n=len(ds); name=d.name+tag
            Xtr,ytr=gather(lk,torch.utils.data.Subset(ds,range(0,800)),lambda d_,i:CARLA_IMG_HW,rng,4,name+" probe")
            dom[name]={"train":(Xtr,ytr),"src":obj_feats(lk,ds,range(0,800),rng),"tgt":obj_feats(lk,ds,range(800,n),rng)}
            log(f"{name}: src {len(dom[name]['src'])} tgt {len(dom[name]['tgt'])} ({time.time()-t0:.0f}s)")
    lk=SpatialFeatureLookup.from_npz(Path.home()/"datasets/kitti/spatial_features_dinov3.npz")
    kv=KITTIDetection(kitti_root=str(Path.home()/"datasets/kitti/extracted/training"),split_json=str(Path.home()/"datasets/kitti/splits.json"),split="val")
    dom["KITTI"]={"tgt":obj_feats(lk,kv,range(0,len(kv),max(1,len(kv)//1000)),rng)}; log(f"KITTI: {len(dom['KITTI']['tgt'])} objects")
    from s2rgap.data.cityscapes_coco import CityscapesCOCO
    lk=SpatialFeatureLookup.from_npz(Path.home()/"datasets/cityscapes/spatial_features_dinov3_val.npz")
    cs=CityscapesCOCO(Path.home()/"datasets/cityscapes/extracted/leftImg8bit",REPO/"nvme/datasets/aldi_labels/cityscapes_val_instances.json",split="val")
    dom["Cityscapes"]={"tgt":obj_feats(lk,cs,range(len(cs)),rng)}; log(f"Cityscapes: {len(dom['Cityscapes']['tgt'])} objects")
    sources=[n for n in dom if "train" in dom[n]]; targets=list(dom)
    out=json.load(open(a.out)) if os.path.exists(a.out) else {"k":K,"cap":CAP,"pairs":{}}   # resume (cgroup-killed at 14 GB on 2026-10-15)
    for S in sources:
        if S in out["pairs"]: continue
        X,y=dom[S]["train"]; mu,sd=X.mean(0),X.std(0)+1e-6
        clf=LogisticRegression(max_iter=500,C=1.0).fit((X-mu)/sd,y); Q,_=np.linalg.qr(clf.coef_.T.astype(np.float32))   # 1024 x 4 orthonormal task basis
        tr_acc=float(clf.score((X-mu)/sd,y)); out["pairs"][S]={"probe_train_acc":tr_acc}
        for T in targets:
            if T==S: Xt=dom[S]["tgt"]          # own held-out (anchor side)
            else: Xt=dom[T]["tgt"]
            if len(Xt)<50: continue
            out["pairs"][S][T]=measures(dom[S]["src"],Xt,mu,sd,Q,dev)
        r=out["pairs"][S]; log(f"{S:<26} probe {tr_acc:.3f} | KITTI fid_full {r['KITTI']['fid_full']:6.1f} fid_task {r['KITTI']['fid_task']:5.2f} cov {r['KITTI']['cov_T_by_S']:.2f} | ({time.time()-t0:.0f}s)")
        Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,"w"))
    log(f"saved -> {a.out}")
if __name__=="__main__": raise SystemExit(main())
