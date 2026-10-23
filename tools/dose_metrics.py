#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Label-free metrics between each dose cell and its town's reference cell, frame-matched (ledger §15.47).
Distribution-level: image FID (all 1000 frames), object FID (std by ref), directed NN distance dose->ref, coverage of dose by ref.
Paired (possible only because frames/boxes are identical across doses): mean per-frame and per-object feature displacement (normalised by
the ref's own k-NN radius), task-projected object displacement (onto the ref probe's discriminative subspace), probe flip rate.
Real-target: image FID(dose cell, KITTI / Cityscapes) exactly as tools/fid_to_real.py."""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection
from linear_probe_object_class import bbox_to_feat_region
from domain_matrix_probe import gather, CARLA_IMG_HW
from domain_distance_matrix import fid, _cap
from latent_geometry import knn_radius
from fid_to_real import img_feats
def log(m): print(f"[dosem] {m}",flush=True)
def objs(lk, ds, idxs):
    gh,gw=lk.target_hw; X=[]; I=[]; F=[]
    for i in idxs:
        im,t=ds[i]; z=np.array(lk.z_mmap[lk.path_to_idx[t["frame_rel"]]],dtype=np.float32); H,W=int(im.shape[-2]),int(im.shape[-1])
        F.append(z.reshape(-1,z.shape[-1]).mean(0))
        for b,l in zip(t["boxes"].numpy(),t["labels"].numpy()):
            i1,j1,i2,j2=bbox_to_feat_region(tuple(b),(H,W),(gh,gw)); p=z[i1:i2,j1:j2].reshape(-1,z.shape[-1])
            if len(p): X.append(p.mean(0)); I.append(i)
    return np.array(F,np.float32), np.array(X,np.float32)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="/mnt/features/carla_doses"); p.add_argument("--features",required=True); p.add_argument("--out",default="results/calib_dose/dose_metrics.json")
    p.add_argument("--skip-real",action="store_true",help="other feature spaces: within-CARLA metrics only (no same-space real-target caches)")
    a=p.parse_args(); dev=torch.device("cuda"); lk=SpatialFeatureLookup.from_npz(a.features); out=json.load(open(a.out)) if os.path.exists(a.out) else {}
    if not a.skip_real:
        lkk=SpatialFeatureLookup.from_npz(Path.home()/"datasets/kitti/spatial_features_dinov3.npz"); kv=KITTIDetection(kitti_root=str(Path.home()/"datasets/kitti/extracted/training"),split_json=str(Path.home()/"datasets/kitti/splits.json"),split="val")
        K=img_feats(lkk,[kv[i][1]["frame_rel"] for i in range(min(1000,len(kv)))])
        lkc=SpatialFeatureLookup.from_npz(Path.home()/"datasets/cityscapes/spatial_features_dinov3_val.npz"); CS=np.array([np.array(lkc.z_mmap[i],dtype=np.float32).reshape(-1,lkc.z_mmap.shape[-1]).mean(0) for i in range(min(1000,len(lkc)))],np.float32)
    cells=sorted(d.name for d in Path(a.root).iterdir() if d.is_dir() and any(d.glob("frame_*.jpg"))); towns=sorted({c.split("_d")[0] for c in cells})
    for town in towns:
        ref=f"{town}_d00_ref"; dsr=CARLADetection(root=a.root,cells=[ref]); n=len(dsr); Fr,Or=objs(lk,dsr,range(n))
        rng=np.random.default_rng(42); sel=np.sort(rng.choice(len(Or),min(3000,len(Or)),replace=False))
        Xp,yp=gather(lk,torch.utils.data.Subset(dsr,range(0,800 if n>=1000 else int(0.75*n))),lambda d,i:CARLA_IMG_HW,np.random.default_rng(7),4,ref+" probe"); mu,sd=Xp.mean(0),Xp.std(0)+1e-6
        clf=LogisticRegression(max_iter=500).fit((Xp-mu)/sd,yp); Q,_=np.linalg.qr(clf.coef_.T.astype(np.float32))
        O0=(Or[sel]-mu)/sd; S=torch.from_numpy(O0).to(dev); rS=knn_radius(S); rimg=float(torch.cdist(torch.from_numpy(Fr).to(dev),torch.from_numpy(Fr).to(dev)).median())
        p0=clf.predict(O0)
        for c in [x for x in cells if x.startswith(town+"_d")]:
            if c in out: continue
            ds=CARLADetection(root=a.root,cells=[c]); assert len(ds)==n, (c,len(ds),n)
            Fd,Od=objs(lk,ds,range(n)); assert len(Od)==len(Or), (c,len(Od),len(Or))
            O1=(Od[sel]-mu)/sd; T=torch.from_numpy(O1).to(dev); dm=torch.cdist(T,S).min(1).values
            r={"img_fid_ref":float(fid(_cap(Fr,1),_cap(Fd,2))),"obj_fid_ref":float(fid(O0,O1)),
               "nnd_dose_to_ref":float(dm.median()/rS.median()),"cov_dose_by_ref":float((dm<=rS[torch.cdist(T,S).argmin(1)]).float().mean()),
               "paired_img_disp":float(np.linalg.norm(Fd-Fr,axis=1).mean()/rimg),"paired_obj_disp":float(np.linalg.norm(O1-O0,axis=1).mean()/float(rS.median())),
               "paired_task_disp":float(np.linalg.norm((O1-O0)@Q,axis=1).mean()),"probe_flip":float((clf.predict(O1)!=p0).mean()),
               "n_obj":int(len(sel))}
            if not a.skip_real: r.update({"fid_KITTI":float(fid(_cap(K,1),_cap(Fd,2))),"fid_Cityscapes":float(fid(_cap(CS,1),_cap(Fd,2)))})
            out[c]=r; json.dump(out,open(a.out,"w"),indent=1); log(f"{c:<22} "+" ".join(f"{k}={v:.3f}" for k,v in r.items() if k!="n_obj"))
if __name__=="__main__": raise SystemExit(main())
