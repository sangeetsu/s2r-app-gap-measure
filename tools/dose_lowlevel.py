#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Low-level / classic baselines on the frame-matched testbed (reviewer item 1): if these track the appearance-only drop as well as the
frozen-FM metrics, the 'latent' part of the claim is not earned. Paired (frame i of dose vs frame i of reference): LPIPS-AlexNet, SSIM, PSNR,
|Δ mean luminance|, |Δ RMS contrast|. Distribution: Inception-pool3 FID (pytorch-fid) ref vs dose, and dose vs KITTI / Cityscapes (same
images as the DINOv3 FID-to-real)."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, lpips
from PIL import Image
from skimage.metrics import structural_similarity
from pytorch_fid.inception import InceptionV3
from scipy import linalg
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.data.kitti import KITTIDetection
def log(m): print(f"[lowlevel] {m}",flush=True)
def fid_np(a,b):
    mu1,mu2=a.mean(0),b.mean(0); s1,s2=np.cov(a,rowvar=False),np.cov(b,rowvar=False); cm,_=linalg.sqrtm(s1.dot(s2),disp=False)
    if np.iscomplexobj(cm): cm=cm.real
    return float(((mu1-mu2)**2).sum()+np.trace(s1)+np.trace(s2)-2*np.trace(cm))
@torch.no_grad()
def inc_feats(model,paths,dev,bs=32):
    F=[]
    for i in range(0,len(paths),bs):
        x=torch.stack([torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((299,299),Image.BILINEAR),np.float32)/255.).permute(2,0,1) for p in paths[i:i+bs]]).to(dev)
        F.append(model(x)[0].squeeze(-1).squeeze(-1).cpu().numpy())
    return np.concatenate(F)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="/mnt/features/carla_doses"); p.add_argument("--out",default="results/calib_dose/dose_lowlevel.json"); p.add_argument("--n-paired",type=int,default=300)
    a=p.parse_args(); dev=torch.device("cuda"); out=json.load(open(a.out)) if os.path.exists(a.out) else {}
    lp=lpips.LPIPS(net="alex",verbose=False).to(dev).eval(); inc=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    kv=KITTIDetection(kitti_root=str(Path.home()/"datasets/kitti/extracted/training"),split_json=str(Path.home()/"datasets/kitti/splits.json"),split="val")
    kimg=Path.home()/"datasets/kitti/extracted/training/image_2"; kpaths=[str(kimg/kv[i][1]["frame_rel"]) for i in range(min(1000,len(kv)))]
    cs=sorted((Path.home()/"datasets/cityscapes/extracted/leftImg8bit/val").rglob("*_leftImg8bit.png"))[:500]
    real={"KITTI":inc_feats(inc,kpaths,dev),"Cityscapes":inc_feats(inc,[str(x) for x in cs],dev)}; log(f"inception feats: KITTI {len(real['KITTI'])}, CS {len(real['Cityscapes'])}")
    cells=sorted(d.name for d in Path(a.root).iterdir() if d.is_dir() and any(d.glob("frame_*.jpg"))); towns=sorted({c.split("_d")[0] for c in cells}); incf={}
    def feats(c):
        if c not in incf: incf[c]=inc_feats(inc,sorted(str(x) for x in (Path(a.root)/c).glob("frame_*.jpg")),dev)
        return incf[c]
    for town in towns:
        ref=f"{town}_d00_ref"; rp=sorted((Path(a.root)/ref).glob("frame_*.jpg")); step=max(1,len(rp)//a.n_paired); rp=rp[::step][:a.n_paired]
        for c in [x for x in cells if x.startswith(town+"_d")]:
            if c in out: continue
            dp=[Path(a.root)/c/x.name for x in rp]; L=[];S=[];P=[];dl=[];dc=[]
            for r_,d_ in zip(rp,dp):
                A=np.asarray(Image.open(r_).convert("RGB")); B=np.asarray(Image.open(d_).convert("RGB")); ga=A.mean(2); gb=B.mean(2)
                S.append(structural_similarity(ga,gb,data_range=255)); mse=((A.astype(np.float32)-B.astype(np.float32))**2).mean(); P.append(10*np.log10(255**2/max(mse,1e-6)))
                dl.append(abs(ga.mean()-gb.mean())/255); dc.append(abs(ga.std()-gb.std())/255)
                ta=torch.from_numpy(A).permute(2,0,1).float().div(127.5).sub(1)[None].to(dev); tb=torch.from_numpy(B).permute(2,0,1).float().div(127.5).sub(1)[None].to(dev)
                with torch.no_grad(): L.append(float(lp(ta,tb)))
            Fd=feats(c); r={"lpips":float(np.mean(L)),"ssim":float(np.mean(S)),"psnr":float(np.mean(P)),"dlum":float(np.mean(dl)),"dcontrast":float(np.mean(dc)),
               "incfid_ref":fid_np(feats(ref),Fd),"incfid_KITTI":fid_np(real["KITTI"],Fd),"incfid_Cityscapes":fid_np(real["Cityscapes"],Fd),"n_paired":len(rp)}
            out[c]=r; json.dump(out,open(a.out,"w"),indent=1); log(f"{c:<22} "+" ".join(f"{k}={v:.3f}" for k,v in r.items() if k!="n_paired"))
if __name__=="__main__": raise SystemExit(main())
