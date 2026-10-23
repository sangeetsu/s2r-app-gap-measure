#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LABEL-FREE box-level gap: unsupervised object discovery on frozen DINOv3 -> pooled probe.

§15.22: the box-pooled probe gap predicts anchored detector drop (r~.73-.75 on both real targets)
but needs target GT boxes; the dense per-patch gap is null even with labels. So make the box
step label-free: discover objects on target frames from the frozen feature graph itself.

Discovery = MaskCut-style iterated Normalized Cut on the patch-affinity graph (Wang et al.,
CutLER, CVPR 2023, arXiv:2301.11320; TokenCut, Wang et al., CVPR 2022, arXiv:2202.11539):
  A_ij = 1[cos(f_i,f_j) >= tau] (tau=0.15 as in TokenCut), Fiedler vector of the normalized
  Laplacian -> bipartition; foreground = side containing the max-absolute-eigenvector patch,
  provided it is not the larger-than-half-image side; connected component -> bbox; mask it out,
  repeat K=3 times per frame. Pure numpy on the 22x76 (or 32x64) grid: cheap.

Then for source cell S with its anchored probe (trained on 0:800 GT boxes):
  acc_in(S) on own held-out GT boxes (as before);
  on target T (held-out frames, NO labels): discovered boxes -> pooled feature -> probe softmax;
  two label-free scores:  conf_T = mean max-prob over discovered boxes (plain confidence)
                          atc_T  = P(max-prob > tau_S) with tau_S from source held-out GT boxes
  gap_conf := acc_in - conf_T ; gap_atc := acc_in - atc_T.
Also reports discovery sanity: fraction of discovered boxes with IoU>=0.5 to ANY GT box
(diagnostic only; uses labels).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.sparse.csgraph import laplacian
from scipy.ndimage import label as cc_label
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection
from linear_probe_object_class import bbox_to_feat_region
from domain_matrix_probe import gather, CARLA_IMG_HW
CELLS=["Town01_ClearNoon","Town01_OvercastNoon","Town01_SoftRainNoon","Town01_ClearNight","Town03_ClearNoon","Town03_OvercastNoon","Town03_SoftRainNoon","Town03_ClearNight"]
def log(m): print(f"[disc] {m}",flush=True)

def maskcut(z, tau=0.15, K=3):
    """z: (gh,gw,D) -> list of (i1,j1,i2,j2) grid boxes (inclusive-exclusive)."""
    gh,gw,D=z.shape; f=z.reshape(-1,D).astype(np.float32); f/=np.linalg.norm(f,axis=1,keepdims=True)+1e-8
    A=(f@f.T>=tau).astype(np.float32); alive=np.ones(gh*gw,bool); boxes=[]
    for _ in range(K):
        idx=np.where(alive)[0]
        if len(idx)<20: break
        Ai=A[np.ix_(idx,idx)]; Ai+=1e-6
        Lsym=laplacian(Ai,normed=True)
        try: w,v=np.linalg.eigh(Lsym)
        except np.linalg.LinAlgError: break
        fied=v[:,1]; seed=np.argmax(np.abs(fied)); side=fied*np.sign(fied[seed])>0
        if side.mean()>0.5: side=~side                     # foreground must be the smaller side
        mask=np.zeros(gh*gw,bool); mask[idx[side]]=True; m2=mask.reshape(gh,gw)
        lab,n=cc_label(m2)
        if n==0: break
        comp=lab==np.argmax(np.bincount(lab[lab>0]))        # largest connected component
        ys,xs=np.where(comp)
        if len(ys)<2: alive[idx[side]]=False; continue
        boxes.append((ys.min(),xs.min(),ys.max()+1,xs.max()+1)); alive[comp.reshape(-1)]=False
    return boxes

class MLP(nn.Module):
    def __init__(s,d): super().__init__(); s.a=nn.Linear(d,512); s.b=nn.Linear(512,4)
    def forward(s,x): return s.b(F.dropout(F.relu(s.a(x)),0.1,s.training))
def fit(X,y,dev,seed):
    torch.manual_seed(seed); mu,sd=X.mean(0),X.std(0)+1e-6; m=MLP(X.shape[1]).to(dev); o=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(((X-mu)/sd).astype(np.float32)),torch.from_numpy(y)),batch_size=256,shuffle=True)
    for _ in range(40):
        m.train()
        for xb,yb in dl: o.zero_grad(); F.cross_entropy(m(xb.to(dev)),yb.to(dev)).backward(); o.step()
    m.eval(); return m,mu,sd
def probs(m,mu,sd,X,dev):
    with torch.no_grad(): return F.softmax(m(torch.from_numpy(((X-mu)/sd).astype(np.float32)).to(dev)),1).cpu().numpy()

def discovered_feats(lookup, ds, idxs, hw_fn, max_frames=200):
    """pooled features of discovered boxes + diagnostic IoU hit-rate vs GT."""
    gh,gw=lookup.target_hw; X=[]; hits=0; tot=0
    for i in list(idxs)[:max_frames]:
        _,t=ds[i]; rel=t["frame_rel"]
        if not lookup.has(rel): continue
        z=np.array(lookup.z_mmap[lookup.path_to_idx[rel]],dtype=np.float32); H,W=hw_fn(ds,i)
        gts=[bbox_to_feat_region(tuple(b),(H,W),(gh,gw)) for b in t["boxes"].numpy()]
        for (i1,j1,i2,j2) in maskcut(z):
            X.append(z[i1:i2,j1:j2].reshape(-1,z.shape[-1]).mean(0)); tot+=1
            for (a1,b1,a2,b2) in gts:
                inter=max(0,min(i2,a2)-max(i1,a1))*max(0,min(j2,b2)-max(j1,b1)); u=(i2-i1)*(j2-j1)+(a2-a1)*(b2-b1)-inter
                if u>0 and inter/u>=0.5: hits+=1; break
    return np.array(X,np.float32), (hits/max(1,tot)), tot

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=42); p.add_argument("--out",default="results/calib_anchored/discovery_box_gap_s42.json")
    p.add_argument("--carla-features",default=str(Path.home()/"datasets/carla_extracted/spatial_features_dinov3.npz")); p.add_argument("--kitti-features",default=str(Path.home()/"datasets/kitti/spatial_features_dinov3.npz"))
    p.add_argument("--cs-features",default=str(Path.home()/"datasets/cityscapes/spatial_features_dinov3_val.npz")); p.add_argument("--carla-root",default=str(Path.home()/"datasets/carla_extracted"))
    p.add_argument("--kitti-root",default=str(Path.home()/"datasets/kitti/extracted/training")); p.add_argument("--kitti-splits",default=str(Path.home()/"datasets/kitti/splits.json"))
    a=p.parse_args(); dev=torch.device("cuda"); rng=np.random.default_rng(a.seed); lc=SpatialFeatureLookup.from_npz(a.carla_features)
    tr,held_gt,disc={},{},{}
    for c in CELLS:
        ds=CARLADetection(root=a.carla_root,cells=[c]); n=len(ds)
        tr[c]=gather(lc,torch.utils.data.Subset(ds,range(0,800)),lambda d,i:CARLA_IMG_HW,rng,4,c+" tr")
        held_gt[c]=gather(lc,torch.utils.data.Subset(ds,range(800,n)),lambda d,i:CARLA_IMG_HW,rng,4,c+" held")
        disc[c]=discovered_feats(lc,ds,range(800,n),lambda d,i:CARLA_IMG_HW); log(f"{c}: discovered {disc[c][2]} boxes, GT-hit {disc[c][1]*100:.0f}%")
    lk=SpatialFeatureLookup.from_npz(a.kitti_features); kv=KITTIDetection(kitti_root=a.kitti_root,split_json=a.kitti_splits,split="val"); kimg=Path(a.kitti_root)/"image_2"
    def khw(d,i):
        with Image.open(kimg/d.frame_names[i]) as im: return (im.height,im.width)
    disc["KITTI"]=discovered_feats(lk,kv,range(0,len(kv),max(1,len(kv)//300)),khw,300); log(f"KITTI: discovered {disc['KITTI'][2]} boxes, GT-hit {disc['KITTI'][1]*100:.0f}%")
    try:
        from s2rgap.data.cityscapes_coco import CityscapesCOCO
        lcs=SpatialFeatureLookup.from_npz(a.cs_features); cs=CityscapesCOCO(Path.home()/"datasets/cityscapes/extracted/leftImg8bit",REPO/"nvme/datasets/aldi_labels/cityscapes_val_instances.json",split="val")
        disc["Cityscapes"]=discovered_feats(lcs,cs,range(len(cs)),lambda d,i:(1024,2048),300); log(f"Cityscapes: discovered {disc['Cityscapes'][2]} boxes, GT-hit {disc['Cityscapes'][1]*100:.0f}%")
    except Exception as e: log(f"Cityscapes skipped: {e}")
    out={}
    for S in CELLS:
        m,mu,sd=fit(*tr[S],dev,a.seed); Xh,yh=held_gt[S]; Ph=probs(m,mu,sd,Xh,dev); fg=yh>0
        acc_in=float((Ph[fg].argmax(1)==yh[fg]).mean()); tau=float(np.quantile(Ph[fg].max(1),1-acc_in))
        row={"acc_in_fg":acc_in,"tau":tau}
        for T,(Xd,hit,ntot) in disc.items():
            if T==S or len(Xd)==0: continue
            Pt=probs(m,mu,sd,Xd,dev); fgmask=Pt.argmax(1)>0                     # drop boxes the probe itself calls background
            Pf=Pt[fgmask] if fgmask.sum()>=10 else Pt
            conf=float(Pf.max(1).mean()); atc=float((Pf.max(1)>tau).mean())
            row[T]={"conf":conf,"atc":atc,"gap_conf":acc_in-conf,"gap_atc":acc_in-atc,"n_boxes":int(len(Xd)),"disc_gt_hit":hit}
        out[S]=row; log(f"{S:<22} in {acc_in*100:4.1f} | KITTI gap_atc {row['KITTI']['gap_atc']*100:+5.1f}" + (f" | CS gap_atc {row['Cityscapes']['gap_atc']*100:+5.1f}" if "Cityscapes" in row else ""))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,"w"),indent=1); log(f"saved -> {a.out}")
if __name__=="__main__": raise SystemExit(main())
