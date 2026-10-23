#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""LABEL-FREE box-level gap, variant 2: boxes = the source detector's own predictions on the target.

MaskCut discovery on frozen DINOv3 hits only 3% of GT objects on driving frames (plan 2026-10-13), so
the proposal mechanism is the source detector S itself — standard in label-free accuracy estimation
(DoC: Guillory et al. ICML'21; ATC: Garg et al. ICLR'22 arXiv:2201.04234; BoS: ICLR'24 arXiv:2403.13803).
NO target labels enter any signal; labeled quantities are kept only as diagnostics (det_prec, true_acc).

Source S = anchored detector (trained on S frames 0:800) + box probe (S GT boxes 0:800, 4-class incl. bg).
  S held-out 800:1000: det boxes (score>=tau_det) -> IoU>=.5 match -> label (0=bg):
      acc_in = probe accuracy on those boxes; tau_S = ATC threshold = quantile_{1-acc_in}(max-prob)
      plus the in-domain reference value of every label-free statistic below.
  Target T (CARLA cell 800:1000 | KITTI val | Cityscapes val): det boxes -> pooled frozen feature -> probe:
      probe_atc = P(max-prob > tau_S)   probe_conf = mean max-prob   agree = P(probe argmax == det label)
      probe_bg  = P(probe argmax == 0)  det_conf   = mean det score (DoC)   dets_pf = boxes per frame
  gap_atc := acc_in - probe_atc (label-free analogue of dIV_fg).
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, torch
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection
from linear_probe_object_class import bbox_to_feat_region
from domain_matrix_probe import gather, CARLA_IMG_HW
from discovery_box_gap import fit, probs
def log(m): print(f"[dbg] {m}",flush=True)
TAUS=(0.3,0.5,0.7)

def load_det(ck,dev):
    m=fasterrcnn_resnet50_fpn(weights=None); m.roi_heads.box_predictor=FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,4)
    c=torch.load(ck,map_location=dev,weights_only=False); m.load_state_dict(c.get("state_dict") or c.get("model") or c); return m.to(dev).eval()

def run_target(det,lookup,ds,idxs,dev,bs):
    """per-box arrays: feat (n,D), score, dlabel, glabel (IoU>=.5-matched GT label, 0=bg); n_frames. Image H,W from the tensor."""
    feats,scores,dl,gl=[],[],[],[]; idxs=list(idxs); nf=0; gh,gw=lookup.target_hw
    for s in range(0,len(idxs),bs):
        batch=[]
        for i in idxs[s:s+bs]:
            im,t=ds[i]
            if lookup.has(t["frame_rel"]): batch.append((im,t))
        if not batch: continue
        with torch.inference_mode(): outs=det([b[0].to(dev) for b in batch])
        for (im,t),o in zip(batch,outs):
            nf+=1; H,W=int(im.shape[-2]),int(im.shape[-1]); z=np.array(lookup.z_mmap[lookup.path_to_idx[t["frame_rel"]]],dtype=np.float32)
            b=o["boxes"].cpu(); sc=o["scores"].cpu().numpy(); lb=o["labels"].cpu().numpy()
            if len(b)==0: continue
            g=t["boxes"].float(); gla=t["labels"].numpy()
            if len(g): mx,am=box_iou(b,g).max(1); ml=np.where(mx.numpy()>=0.5,gla[am.numpy()],0)
            else: ml=np.zeros(len(b),int)
            for k,(x1,y1,x2,y2) in enumerate(b.numpy()):
                i1,j1,i2,j2=bbox_to_feat_region((x1,y1,x2,y2),(H,W),(gh,gw)); p=z[i1:i2,j1:j2].reshape(-1,z.shape[-1])
                if len(p)==0: continue
                feats.append(p.mean(0)); scores.append(sc[k]); dl.append(lb[k]); gl.append(ml[k])
    return dict(feat=np.array(feats,np.float32),score=np.array(scores,np.float32),dlabel=np.array(dl,int),glabel=np.array(gl,int),n_frames=nf)

def stats(R,P,tau_det,tau_S):
    m=R["score"]>=tau_det
    if m.sum()<10: return None
    Pm=P[m]; mp=Pm.max(1); am=Pm.argmax(1); dlab=R["dlabel"][m]; glab=R["glabel"][m]
    return dict(n_boxes=int(m.sum()),dets_pf=float(m.sum()/max(1,R["n_frames"])),det_conf=float(R["score"][m].mean()),probe_conf=float(mp.mean()),
                probe_atc=float((mp>tau_S).mean()),probe_pbg=float(Pm[:,0].mean()),agree=float((am==dlab).mean()),probe_bg=float((am==0).mean()),
                det_prec=float((glab>0).mean()),true_acc=float((am==glab).mean()))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42); p.add_argument("--probe-seeds",default=None,help="comma list: ensemble-average the probe softmax over these seeds"); p.add_argument("--out",default="results/calib_anchored/detector_box_gap_s42.json")
    p.add_argument("--carla-root",default=str(Path.home()/"datasets/carla_extracted")); p.add_argument("--carla-features",default=str(Path.home()/"datasets/carla_extracted/spatial_features_dinov3.npz"))
    p.add_argument("--extra-root",default="/mnt/features/carla_new"); p.add_argument("--extra-features",default=None)
    p.add_argument("--kitti-root",default=str(Path.home()/"datasets/kitti/extracted/training")); p.add_argument("--kitti-splits",default=str(Path.home()/"datasets/kitti/splits.json"))
    p.add_argument("--kitti-features",default=str(Path.home()/"datasets/kitti/spatial_features_dinov3.npz")); p.add_argument("--cs-features",default=str(Path.home()/"datasets/cityscapes/spatial_features_dinov3_val.npz"))
    p.add_argument("--n-kitti",type=int,default=300); p.add_argument("--n-cs",type=int,default=150); p.add_argument("--ckpt-glob",default="checkpoints/anch_*/last.pt")
    a=p.parse_args(); dev=torch.device("cuda"); rng=np.random.default_rng(a.seed); SEEDS=[int(x) for x in a.probe_seeds.split(",")] if a.probe_seeds else [a.seed]
    roots=[(a.carla_root,SpatialFeatureLookup.from_npz(a.carla_features))]
    if a.extra_features and Path(a.extra_features).exists(): roots.append((a.extra_root,SpatialFeatureLookup.from_npz(a.extra_features)))
    cellroot={}
    for r,lk in roots:
        for d in sorted(Path(r).iterdir()):
            if d.is_dir() and len(list(d.glob("frame_*.jpg")))>=1000: cellroot[d.name]=(r,lk)
    import re; ckmap={re.sub(r"^anch[a-z]*_","",Path(c).parent.name): c for c in sorted(glob.glob(a.ckpt_glob))}   # anch_X (COCO) / anchin_X (ImageNet)
    srcs=[s for s in ckmap if s in cellroot]
    targets=list(cellroot)+["KITTI","Cityscapes"]
    out=json.load(open(a.out)) if Path(a.out).exists() else {}
    lk_k=SpatialFeatureLookup.from_npz(a.kitti_features); kv=KITTIDetection(kitti_root=a.kitti_root,split_json=a.kitti_splits,split="val")
    kidx=list(range(0,len(kv),max(1,len(kv)//a.n_kitti)))[:a.n_kitti]
    try:
        from s2rgap.data.cityscapes_coco import CityscapesCOCO
        lk_c=SpatialFeatureLookup.from_npz(a.cs_features); cs=CityscapesCOCO(Path.home()/"datasets/cityscapes/extracted/leftImg8bit",REPO/"nvme/datasets/aldi_labels/cityscapes_val_instances.json",split="val")
        cidx=list(range(0,len(cs),max(1,len(cs)//a.n_cs)))[:a.n_cs]
    except Exception as e: log(f"Cityscapes unavailable: {e}"); cs=None
    log(f"{len(srcs)} sources, {len(targets)} targets, KITTI {len(kidx)} frames, CS {len(cidx) if cs else 0} frames")
    for S in srcs:
        if S in out and all(T in out[S] for T in targets if T!=S and not (T=="Cityscapes" and cs is None)): log(f"skip {S} (done)"); continue
        t0=time.time(); r,lk=cellroot[S]; ds=CARLADetection(root=r,cells=[S]); n=len(ds)
        X,y=gather(lk,torch.utils.data.Subset(ds,range(0,800)),lambda d,i:CARLA_IMG_HW,rng,4,S+" 0:800"); models=[fit(X,y,dev,sd_) for sd_ in SEEDS]; P_of=lambda Fx: np.mean([probs(m_,mu_,s_,Fx,dev) for (m_,mu_,s_) in models],0)
        det=load_det(ckmap[S],dev)
        Rin=run_target(det,lk,ds,range(800,n),dev,8); Pin=P_of(Rin["feat"]); row={"n_frames_in":Rin["n_frames"],"probe_seeds":SEEDS}
        for td in TAUS:
            msk=Rin["score"]>=td; acc_in=float((Pin[msk].argmax(1)==Rin["glabel"][msk]).mean()); tau_S=float(np.quantile(Pin[msk].max(1),1-acc_in))
            row[f"in@{td}"]=dict(acc_in=acc_in,tau=tau_S,**(stats(Rin,Pin,td,tau_S) or {}))
        for T in targets:
            if T==S: continue
            if T=="KITTI": R=run_target(det,lk_k,kv,kidx,dev,8)
            elif T=="Cityscapes":
                if cs is None: continue
                R=run_target(det,lk_c,cs,cidx,dev,2)
            else: rt,lkt=cellroot[T]; dt=CARLADetection(root=rt,cells=[T]); R=run_target(det,lkt,dt,range(800,len(dt)),dev,8)
            if len(R["feat"])==0: continue
            P=P_of(R["feat"]); row[T]={}
            for td in TAUS:
                st=stats(R,P,td,row[f"in@{td}"]["tau"])
                if st: st["gap_atc"]=row[f"in@{td}"]["acc_in"]-st["probe_atc"]; row[T][str(td)]=st
        out[S]=row; Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,"w"),indent=1)
        k=row.get("KITTI",{}).get("0.5",{}); i5=row["in@0.5"]
        log(f"{S:<22} acc_in@.5 {i5['acc_in']*100:4.1f} dets/f {i5.get('dets_pf',0):4.1f} | KITTI gap_atc {k.get('gap_atc',np.nan)*100:+5.1f} agree {k.get('agree',np.nan)*100:4.1f} doc {(i5.get('det_conf',np.nan)-k.get('det_conf',np.nan))*100:+5.1f} true_gap {(i5['acc_in']-k.get('true_acc',np.nan))*100:+5.1f} | {time.time()-t0:.0f}s")
    log(f"saved -> {a.out}")
if __name__=="__main__": raise SystemExit(main())
