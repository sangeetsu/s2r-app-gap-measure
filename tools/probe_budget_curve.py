#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Label budget curve for the anchored task-conditional gap dIV_fg (ledger §15.21).

dIV_fg(S->T) = acc_fg(S probe on own held-out GT boxes) - acc_fg(S probe on T GT boxes) needs labeled TARGET
boxes. Within a simulator, labels are free. For a real target, how many labeled frames are enough?
For each (S,T): per-frame correctness of the S probe on T's fg GT boxes; draw k frames (B draws), recompute
dIV_fg^k; report (a) Pearson r with the anchored mAP drop across pairs per draw (mean, 5-95%), (b) measurement
noise sd of dIV_fg^k vs the between-pair signal sd  ->  reliability = 1 - noise_var/total_var.
Targets: CARLA held-out 800:1000 (<=200 frames), KITTI val (all frames by default), Cityscapes val (<=150).
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, torch
from scipy import stats
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection
from linear_probe_object_class import bbox_to_feat_region
from domain_matrix_probe import gather, CARLA_IMG_HW
from discovery_box_gap import probs, MLP
import torch.nn.functional as F
def fit(X,y,dev,seed,epochs=50):   # identical to domain_matrix_probe.fit_eval training (50 epochs) so dIV matches the probe matrix
    torch.manual_seed(seed); mu,sd=X.mean(0),X.std(0)+1e-6; m=MLP(X.shape[1]).to(dev); o=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    dl=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(((X-mu)/sd).astype(np.float32)),torch.from_numpy(y)),batch_size=256,shuffle=True)
    for _ in range(epochs):
        m.train()
        for xb,yb in dl: o.zero_grad(); F.cross_entropy(m(xb.to(dev)),yb.to(dev)).backward(); o.step()
    m.eval(); return m,mu,sd
def log(m): print(f"[budget] {m}",flush=True)
R="results/calib_anchored"
def mAP(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100

def frames_fg(lookup, ds, idxs):
    """list over frames of (feat (n_i,D), label (n_i,)) for fg GT boxes; H,W from the image tensor."""
    gh,gw=lookup.target_hw; out=[]
    for i in idxs:
        im,t=ds[i]; rel=t["frame_rel"]
        if not lookup.has(rel): continue
        H,W=int(im.shape[-2]),int(im.shape[-1]); z=np.array(lookup.z_mmap[lookup.path_to_idx[rel]],dtype=np.float32); X,y=[],[]
        for b,l in zip(t["boxes"].numpy(),t["labels"].numpy()):
            if l<=0: continue
            i1,j1,i2,j2=bbox_to_feat_region(tuple(b),(H,W),(gh,gw)); p=z[i1:i2,j1:j2].reshape(-1,z.shape[-1])
            if len(p): X.append(p.mean(0)); y.append(int(l))
        if X: out.append((np.array(X,np.float32),np.array(y)))
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--seed",type=int,default=42); p.add_argument("--draws",type=int,default=30)
    p.add_argument("--budgets",default="5,10,20,50,100,200"); p.add_argument("--n-kitti",type=int,default=10**6); p.add_argument("--n-cs",type=int,default=150)
    p.add_argument("--out",default=None)
    a=p.parse_args(); a.out=a.out or f"{R}/probe_budget_curve_s{a.seed}.json"; dev=torch.device("cuda"); rng=np.random.default_rng(a.seed); K=[int(k) for k in a.budgets.split(",")]
    cells=sorted({f.split("eval_")[1].split("__on__")[0] for f in glob.glob(f"{R}/eval_*.json")})
    anchor={s:mAP(f"{R}/eval_{s}__on__{s}.json") for s in cells if glob.glob(f"{R}/eval_{s}__on__{s}.json")}; cells=[c for c in cells if c in anchor]
    root=Path.home()/"datasets/carla_extracted"; lk=SpatialFeatureLookup.from_npz(root/"spatial_features_dinov3.npz")
    held={}; probes={}; acc_in={}
    for c in cells:
        ds=CARLADetection(root=str(root),cells=[c]); n=len(ds)
        X,y=gather(lk,torch.utils.data.Subset(ds,range(0,800)),lambda d,i:CARLA_IMG_HW,rng,4,c+" 0:800"); probes[c]=fit(X,y,dev,a.seed)
        held[c]=frames_fg(lk,ds,range(800,n))
    lk_k=SpatialFeatureLookup.from_npz(Path.home()/"datasets/kitti/spatial_features_dinov3.npz")
    kv=KITTIDetection(kitti_root=str(Path.home()/"datasets/kitti/extracted/training"),split_json=str(Path.home()/"datasets/kitti/splits.json"),split="val")
    held["KITTI"]=frames_fg(lk_k,kv,list(range(0,len(kv),max(1,len(kv)//a.n_kitti)))[:a.n_kitti])
    try:
        from s2rgap.data.cityscapes_coco import CityscapesCOCO
        lk_c=SpatialFeatureLookup.from_npz(Path.home()/"datasets/cityscapes/spatial_features_dinov3_val.npz")
        cs=CityscapesCOCO(Path.home()/"datasets/cityscapes/extracted/leftImg8bit",REPO/"nvme/datasets/aldi_labels/cityscapes_val_instances.json",split="val")
        held["Cityscapes"]=frames_fg(lk_c,cs,list(range(0,len(cs),max(1,len(cs)//a.n_cs)))[:a.n_cs])
    except Exception as e: log(f"Cityscapes unavailable: {e}")
    log("targets: "+", ".join(f"{T}:{len(v)}f/{sum(len(y) for _,y in v)}b" for T,v in held.items()))
    # per-(S,T) per-frame correctness
    corr={}
    for S in cells:
        m,mu,sd=probes[S]
        for T,frames in held.items():
            cf=[(probs(m,mu,sd,X,dev).argmax(1)==y) for X,y in frames]; corr[(S,T)]=cf
        acc_in[S]=float(np.concatenate(corr[(S,S)]).mean())
    def pairs_for(T=None):
        if T is None: return [(S,U) for S in cells for U in cells if S!=U and glob.glob(f"{R}/eval_{S}__on__{U}.json")]
        key={"KITTI":"kittival","Cityscapes":"cityscapesval"}[T]; return [(S,T) for S in cells if glob.glob(f"{R}/src_{S}_{key}.json")]
    def drop_of(S,T):
        if T in ("KITTI","Cityscapes"): return anchor[S]-mAP(f"{R}/src_{S}_{ {'KITTI':'kittival','Cityscapes':'cityscapesval'}[T] }.json")
        return anchor[S]-mAP(f"{R}/eval_{S}__on__{T}.json")
    def div_k(S,T,k,rg):
        cf=corr[(S,T)]; n=len(cf); idx=rg.choice(n,min(k,n),replace=False) if k<n else np.arange(n)
        return acc_in[S]-float(np.concatenate([cf[i] for i in idx]).mean())
    out={"budgets_frames":K,"draws":a.draws,"acc_in":acc_in,"results":{}}
    for name,P in [("CARLA_pairs",pairs_for()),("KITTI",pairs_for("KITTI")),("Cityscapes",pairs_for("Cityscapes"))]:
        if len(P)<4: continue
        y=np.array([drop_of(S,T) for S,T in P]); full=np.array([div_k(S,T,10**9,rng) for S,T in P]); r_full=stats.pearsonr(full,y)[0]
        res={"n_pairs":len(P),"r_full":r_full,"signal_sd":float(full.std()),"per_k":{}}
        print(f"\n=== {name}: {len(P)} pairs, full-sample r={r_full:+.3f}, dIV signal sd={full.std()*100:.1f} pp ===")
        print(f"  {'k frames':>8} {'r mean':>7} {'[5%,95%]':>16} {'noise sd':>9} {'reliab.':>8}")
        for k in K:
            rs=[]; noise=[]
            for b in range(a.draws):
                rg=np.random.default_rng(a.seed*1000+b*7+k); d=np.array([div_k(S,T,k,rg) for S,T in P]); rs.append(stats.pearsonr(d,y)[0]); noise.append(d-full)
            noise=np.array(noise); nsd=float(np.sqrt((noise**2).mean())); rel=float(max(0.0,1-nsd**2/(full.std()**2+nsd**2)))
            res["per_k"][k]={"r_mean":float(np.mean(rs)),"r_lo":float(np.percentile(rs,5)),"r_hi":float(np.percentile(rs,95)),"noise_sd":nsd,"reliability":rel}
            print(f"  {k:>8} {np.mean(rs):+7.3f} [{np.percentile(rs,5):+.3f},{np.percentile(rs,95):+.3f}] {nsd*100:8.1f}pp {rel:8.2f}")
        out["results"][name]=res
    json.dump(out,open(a.out,"w"),indent=1); log(f"saved -> {a.out}")
if __name__=="__main__": raise SystemExit(main())
