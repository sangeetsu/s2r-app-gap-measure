#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Chou, Kirsanov, Yang & Chung (ICLR 2026, arXiv 2603.01879) representational-geometry markers for sim->real detectors (ledger §15.42).

Representation = each detector's penultimate layer (Faster R-CNN box head, 1024-D) on GT boxes of its OWN source cell, frames 800:1000.
Markers: participation ratio per class (their Eq. 3); GLUE capacity alpha, D_eff, R_eff, Psi_eff (their Eqs. 4-5 + appendix):
  for dichotomy y and t ~ N(0, I_N): lambda = argmin_{lambda>=0} || t - A_y^T lambda ||, A_y = rows y^mu z_i^mu  (exact cone projection, NNLS)
  anchor s^mu = sum_i lambda_i^mu z_i^mu / sum_i lambda_i^mu ; centre s0^mu = E_{y,t}[s^mu] ; axis s1^mu = s^mu - s0^mu
  a = (S_y t)^T (S_y S_y^T)^+ (S_y t),  b = (S1 t)^T (S1 S1^T)^+ (S1 t),  c = (S1 t)^T (S0 S0^T + S1 S1^T)^+ (S1 t)
  D_eff = E[b]/P ; R_eff = sqrt(E[c]/E[b-c]) ; Psi_eff = E[c]/E[a] ; alpha = P/E[a]
Manifold pairs (P=2) among classes with >= M points; M=40 per manifold, n_t=100 per dichotomy, 10 resamples; global-mean centring.
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.optimize import nnls
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src"))
from s2rgap.data.carla import CARLADetection
def log(m): print(f"[mm] {m}",flush=True)
ROOTS=[str(Path.home()/"datasets/carla_extracted"),"/mnt/features/carla_new","/mnt/features/carla_new2","/mnt/features/carla_confirm"]; LOWQ="/mnt/features/carla_lowq"

def glue_pair(Z1,Z2,n_t,rng):
    M=len(Z1); N=Z1.shape[1]; T=rng.standard_normal((n_t,N)); Y=[(1,-1),(-1,1)]; anchors={}
    for y in Y:
        A=np.vstack([y[0]*Z1,y[1]*Z2]).T          # N x 2M, columns = signed points
        for k,t in enumerate(T):
            lam,_=nnls(A,t,maxiter=50*A.shape[1]); l1,l2=lam[:M],lam[M:]
            if l1.sum()<=1e-12 or l2.sum()<=1e-12: continue      # projection does not touch both manifolds: anchors undefined
            anchors[(y,k)]=np.stack([l1@Z1/l1.sum(), l2@Z2/l2.sum()])
    if len(anchors)<10: return None
    S0=np.mean(list(anchors.values()),axis=0)
    a=b=c=0.0; n=0
    for (y,k),S in anchors.items():
        t=T[k]; D=np.diag(y); Sy=D@S; Sy0=D@S0; Sy1=D@(S-S0)
        u=Sy@t; a+=u@np.linalg.pinv(Sy@Sy.T)@u
        v=Sy1@t; b+=v@np.linalg.pinv(Sy1@Sy1.T)@v; c+=v@np.linalg.pinv(Sy0@Sy0.T+Sy1@Sy1.T)@v; n+=1
    a,b,c=a/n,b/n,c/n; P=2
    return {"alpha":P/a,"D_eff":b/P,"R_eff":float(np.sqrt(c/max(b-c,1e-12))),"Psi_eff":c/a,"frac_valid":n/(2*n_t)}

def participation_ratio(Z):
    ev=np.clip(np.linalg.eigvalsh(np.cov(Z.T)),0,None); return float(ev.sum()**2/(ev**2).sum())

def markers(feats,labels,M,n_t,reps,rng):
    cls=[k for k in (1,2,3) if (labels==k).sum()>=M]; out={"classes":cls,"n_per_class":{int(k):int((labels==k).sum()) for k in (1,2,3)}}
    out["PR"]={int(k):participation_ratio(feats[labels==k][:1000]) for k in cls}
    pairs=[(i,j) for ii,i in enumerate(cls) for j in cls[ii+1:]]; agg={}
    for (i,j) in pairs:
        vals=[]
        for r in range(reps):
            Z1=feats[labels==i][rng.choice((labels==i).sum(),M,replace=False)]; Z2=feats[labels==j][rng.choice((labels==j).sum(),M,replace=False)]
            mu=np.vstack([Z1,Z2]).mean(0); g=glue_pair(Z1-mu,Z2-mu,n_t,rng)
            if g: vals.append(g)
        if vals: agg[f"{i}-{j}"]={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
    out["glue"]=agg
    if agg:
        for k in ("alpha","D_eff","R_eff","Psi_eff"): out[k]=float(np.mean([v[k] for v in agg.values()]))
    return out

@torch.no_grad()
def box_head_feats(ck,root,cell,dev,frames=(800,1000),bs=4):
    m=fasterrcnn_resnet50_fpn(weights=None); m.roi_heads.box_predictor=FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,4)
    st=torch.load(ck,map_location=dev,weights_only=False); m.load_state_dict(st.get("state_dict") or st.get("model") or st); m.to(dev).eval()
    ds=CARLADetection(root=root,cells=[cell]); idx=list(range(frames[0],min(frames[1],len(ds)))); F,L=[],[]
    for s in range(0,len(idx),bs):
        ims,tg=[],[]
        for i in idx[s:s+bs]:
            im,t=ds[i]
            if len(t["boxes"])==0: continue
            ims.append(im.to(dev)); tg.append({"boxes":t["boxes"].to(dev),"labels":t["labels"].to(dev)})
        if not ims: continue
        il,tt=m.transform(ims,tg); fm=m.backbone(il.tensors)
        x=m.roi_heads.box_head(m.roi_heads.box_roi_pool(fm,[t["boxes"] for t in tt],il.image_sizes))
        F.append(x.cpu().numpy()); L.append(torch.cat([t["labels"] for t in tt]).cpu().numpy())
    del m; torch.cuda.empty_cache(); return np.concatenate(F),np.concatenate(L)

def parse(ck):
    name=Path(ck).parent.name; pre,rest=name.split("_",1); cell=re.sub(r"_n\d+$","",rest)
    root=LOWQ if pre=="anchinlq" else next(r for r in ROOTS if os.path.isdir(f"{r}/{cell}"))
    return name,pre,cell,root

def main():
    p=argparse.ArgumentParser(); p.add_argument("--globs",nargs="+",default=["checkpoints/anchin_*/last.pt","checkpoints/anchinrep_*/last.pt","checkpoints/anchinlc_*/last.pt","checkpoints/anchinlq_*/last.pt","checkpoints/anch_Town*/last.pt"])
    p.add_argument("--out",default="results/manifold_markers/markers.json"); p.add_argument("--M",type=int,default=40); p.add_argument("--n-t",type=int,default=100); p.add_argument("--reps",type=int,default=10)
    a=p.parse_args(); dev=torch.device("cuda"); out=json.load(open(a.out)) if os.path.exists(a.out) else {}; t0=time.time()
    cks=[c for g in a.globs for c in sorted(glob.glob(g))]; log(f"{len(cks)} detectors")
    for ck in cks:
        name,pre,cell,root=parse(ck)
        if name in out: continue
        F,L=box_head_feats(ck,root,cell,dev); rng=np.random.default_rng(42)
        r=markers(F.astype(np.float64),L,a.M,a.n_t,a.reps,rng); r.update({"set":pre,"cell":cell,"root":root})
        out[name]=r; Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,"w"),indent=1)
        log(f"{name:<34} n={r['n_per_class']} D_eff {r.get('D_eff',np.nan):.3f} Psi {r.get('Psi_eff',np.nan):.3f} R {r.get('R_eff',np.nan):.3f} alpha {r.get('alpha',np.nan):.3f} PR {np.mean(list(r['PR'].values())):.1f} ({time.time()-t0:.0f}s)")
    log(f"saved -> {a.out}")
if __name__=="__main__": raise SystemExit(main())
