#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deployable variant of the frame-matched dose metrics (plan 2026-10-16 R-D): boxes come from the town's REFERENCE detector run on the clean
render (score >= 0.5), not from GT. The same boxes are pooled on the reference and on each dose render (frames are identical), so the
metrics stay paired and need no target labels. Also records detector-side label-free baselines on the dose frames (DoC, count ratio)."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from linear_probe_object_class import bbox_to_feat_region
from domain_matrix_probe import gather, CARLA_IMG_HW
from domain_distance_matrix import fid
from latent_geometry import knn_radius
def log(m): print(f"[detbox] {m}",flush=True)
def load_det(ck,dev):
    m=fasterrcnn_resnet50_fpn(weights=None); m.roi_heads.box_predictor=FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,4)
    st=torch.load(ck,map_location=dev,weights_only=False); m.load_state_dict(st.get("state_dict") or st.get("model") or st); return m.to(dev).eval()
@torch.no_grad()
def detect(det,ds,dev,thr=0.5,bs=8):
    out=[]
    for s in range(0,len(ds),bs):
        ims=[ds[i][0].to(dev) for i in range(s,min(s+bs,len(ds)))]
        for o in det(ims):
            k=o["scores"]>=thr; out.append((o["boxes"][k].cpu().numpy(),o["labels"][k].cpu().numpy(),o["scores"][k].cpu().numpy()))
    return out
def pool(lk,ds,boxes_per_frame):
    gh,gw=lk.target_hw; X=[]
    for i,(b,_,_) in enumerate(boxes_per_frame):
        if len(b)==0: continue
        im,t=ds[i]; z=np.array(lk.z_mmap[lk.path_to_idx[t["frame_rel"]]],dtype=np.float32); H,W=int(im.shape[-2]),int(im.shape[-1])
        for bb in b:
            i1,j1,i2,j2=bbox_to_feat_region(tuple(bb),(H,W),(gh,gw)); p=z[i1:i2,j1:j2].reshape(-1,z.shape[-1]); X.append(p.mean(0) if len(p) else z[min(i1,gh-1),min(j1,gw-1)])
    return np.array(X,np.float32)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="/mnt/features/carla_doses"); p.add_argument("--features",required=True); p.add_argument("--out",default="results/calib_dose/dose_metrics_detbox.json"); p.add_argument("--thr",type=float,default=0.5)
    a=p.parse_args(); dev=torch.device("cuda"); lk=SpatialFeatureLookup.from_npz(a.features); out=json.load(open(a.out)) if os.path.exists(a.out) else {}
    cells=sorted(d.name for d in Path(a.root).iterdir() if d.is_dir() and any(d.glob("frame_*.jpg"))); towns=sorted({c.split("_d")[0] for c in cells})
    for town in towns:
        ref=f"{town}_d00_ref"; dsr=CARLADetection(root=a.root,cells=[ref]); det=load_det(f"checkpoints/anchind_{ref}/last.pt",dev)
        B=detect(det,dsr,dev,a.thr); nref=sum(len(b) for b,_,_ in B); conf_ref=float(np.mean(np.concatenate([s for _,_,s in B]))); log(f"{ref}: {nref} detector boxes on clean render ({nref/len(dsr):.1f}/frame)")
        Xp,yp=gather(lk,torch.utils.data.Subset(dsr,range(0,800)),lambda d,i:CARLA_IMG_HW,np.random.default_rng(7),4,ref+" probe"); mu,sd=Xp.mean(0),Xp.std(0)+1e-6
        clf=LogisticRegression(max_iter=500).fit((Xp-mu)/sd,yp); Q,_=np.linalg.qr(clf.coef_.T.astype(np.float32))
        O0=(pool(lk,dsr,B)-mu)/sd; S=torch.from_numpy(O0).to(dev); rS=knn_radius(S); p0=clf.predict(O0)
        for c in [x for x in cells if x.startswith(town+"_d")]:
            if c in out: continue
            ds=CARLADetection(root=a.root,cells=[c]); O1=(pool(lk,ds,B)-mu)/sd; T=torch.from_numpy(O1).to(dev); dm=torch.cdist(T,S).min(1).values
            Bd=detect(det,ds,dev,a.thr); nd=sum(len(b) for b,_,_ in Bd); cd=float(np.mean(np.concatenate([s for _,_,s in Bd]))) if nd else float("nan")
            r={"probe_flip":float((clf.predict(O1)!=p0).mean()),"obj_fid_ref":float(fid(O0,O1)),"paired_obj_disp":float(np.linalg.norm(O1-O0,axis=1).mean()/float(rS.median())),
               "paired_task_disp":float(np.linalg.norm((O1-O0)@Q,axis=1).mean()),"nnd_dose_to_ref":float(dm.median()/rS.median()),
               "det_doc":conf_ref-cd,"det_count_ratio":float(np.log(max(nd,1)/max(nref,1))),"n_boxes":int(len(O0))}
            out[c]=r; json.dump(out,open(a.out,"w"),indent=1); log(f"{c:<22} "+" ".join(f"{k}={v:.3f}" for k,v in r.items() if k!="n_boxes"))
        del det; torch.cuda.empty_cache()
if __name__=="__main__": raise SystemExit(main())
