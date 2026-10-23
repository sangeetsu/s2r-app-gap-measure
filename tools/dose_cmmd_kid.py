#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E4 (plan_2026-10-17): modern distribution-distance baselines on the frame-matched testbed.
CMMD (Jayasumana et al., CVPR 2024): CLIP ViT-L/14-336 image embeddings (bicubic resize to 336x336, no crop, CLIP normalisation, L2-normalised;
as in google-research/cmmd and its PyTorch port), biased MMD^2 with a Gaussian RBF kernel, sigma 10, scaled by 1000.
KID (Binkowski et al., ICLR 2018): Inception-v3 pool3 features (pytorch-fid weights), unbiased MMD^2 with the cubic polynomial kernel
(x.y/d + 1)^3, mean over 100 random subsets of size min(1000, n) (torch-fidelity defaults).
Pairs: ref cell vs dose cell (all frames), and every cell vs KITTI val (first 1000, same list as fid_to_real) / Cityscapes val (first 500)."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as Fn
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from pytorch_fid.inception import InceptionV3
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src"))
from s2rgap.data.kitti import KITTIDetection
def log(m): print(f"[cmmd] {m}",flush=True)
class DS(torch.utils.data.Dataset):
    def __init__(s,paths): s.p=paths
    def __len__(s): return len(s.p)
    def __getitem__(s,i): return torch.from_numpy(np.asarray(Image.open(s.p[i]).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1)
@torch.no_grad()
def embed(paths,clip,proc,inc,dev):
    C,I=[],[]; mean=torch.tensor(proc.image_mean,device=dev)[:,None,None]; std=torch.tensor(proc.image_std,device=dev)[:,None,None]
    dl=torch.utils.data.DataLoader(DS(paths),batch_size=1,num_workers=4)   # images differ in size across sources; batch per image, stack resized
    buf=[]
    def flush():
        xs=[b.to(dev)[None] for b in buf]   # source images can differ by a few pixels (KITTI); resize each before stacking
        c=torch.cat([Fn.interpolate(x,size=(336,336),mode="bicubic",align_corners=False) for x in xs])
        e=clip(pixel_values=(c-mean)/std).image_embeds; C.append((e/e.norm(dim=-1,keepdim=True)).float().cpu())
        g=torch.cat([Fn.interpolate(x,size=(299,299),mode="bilinear",align_corners=False) for x in xs]); I.append(inc(g)[0].squeeze(-1).squeeze(-1).float().cpu()); buf.clear()
    for x in dl:
        buf.append(x[0])
        if len(buf)==32: flush()
    if buf: flush()
    return torch.cat(C).numpy(), torch.cat(I).numpy().astype(np.float64)
def cmmd(x,y,sigma=10.,scale=1000.):
    x=torch.from_numpy(x).double(); y=torch.from_numpy(y).double(); g=1/(2*sigma**2)
    xx=(x*x).sum(1); yy=(y*y).sum(1)
    kxx=torch.exp(-g*(-2*x@x.T+xx[:,None]+xx[None])).mean(); kyy=torch.exp(-g*(-2*y@y.T+yy[:,None]+yy[None])).mean(); kxy=torch.exp(-g*(-2*x@y.T+xx[:,None]+yy[None])).mean()
    return float(scale*(kxx+kyy-2*kxy))
def kid(x,y,subsets=100,size=1000,seed=0):
    rng=np.random.default_rng(seed); m=min(size,len(x),len(y)); d=x.shape[1]; v=[]
    for _ in range(subsets):
        a=x[rng.choice(len(x),m,replace=False)]; b=y[rng.choice(len(y),m,replace=False)]
        kaa=(a@a.T/d+1)**3; kbb=(b@b.T/d+1)**3; kab=(a@b.T/d+1)**3
        v.append((kaa.sum()-np.trace(kaa))/(m*(m-1))+(kbb.sum()-np.trace(kbb))/(m*(m-1))-2*kab.mean())
    return float(np.mean(v))
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="/mnt/features/carla_doses"); p.add_argument("--out",default="results/calib_dose/dose_cmmd_kid.json"); a=p.parse_args()
    dev=torch.device("cuda"); out=json.load(open(a.out)) if os.path.exists(a.out) else {}
    clip=CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336").to(dev).eval(); proc=CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    inc=InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]],resize_input=False,normalize_input=True).to(dev).eval()
    kv=KITTIDetection(kitti_root=str(Path.home()/"datasets/kitti/extracted/training"),split_json=str(Path.home()/"datasets/kitti/splits.json"),split="val")
    kimg=Path.home()/"datasets/kitti/extracted/training/image_2"; kp=[str(kimg/kv[i][1]["frame_rel"]) for i in range(min(1000,len(kv)))]
    cs=[str(x) for x in sorted((Path.home()/"datasets/cityscapes/extracted/leftImg8bit/val").rglob("*_leftImg8bit.png"))[:500]]
    real={"KITTI":embed(kp,clip,proc,inc,dev),"Cityscapes":embed(cs,clip,proc,inc,dev)}; log(f"real embeddings: KITTI {len(kp)}, CS {len(cs)}")
    cells=sorted(d.name for d in Path(a.root).iterdir() if d.is_dir() and any(d.glob("frame_*.jpg"))); towns=sorted({c.split("_d")[0] for c in cells})
    for t in towns:
        todo=[c for c in cells if c.startswith(t+"_d") and c not in out]
        if not todo: continue
        ref=f"{t}_d00_ref"; E={ref:embed(sorted(str(x) for x in (Path(a.root)/ref).glob("frame_*.jpg")),clip,proc,inc,dev)}
        for c in todo:
            if c not in E: E[c]=embed(sorted(str(x) for x in (Path(a.root)/c).glob("frame_*.jpg")),clip,proc,inc,dev)
            (Cc,Ic),(Cr,Ir)=E[c],E[ref]
            r={"cmmd_ref":cmmd(Cr,Cc),"kid_ref":kid(Ir,Ic),"n":len(Cc)}
            for T,(Ct,It) in real.items(): r[f"cmmd_{T}"]=cmmd(Ct,Cc); r[f"kid_{T}"]=kid(It,Ic)
            out[c]=r; json.dump(out,open(a.out,"w"),indent=1); log(f"{c:<22} "+" ".join(f"{k}={v:.4f}" for k,v in r.items() if k!="n"))
            if c!=ref: del E[c]
if __name__=="__main__": raise SystemExit(main())
