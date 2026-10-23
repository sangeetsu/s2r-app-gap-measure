#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Published label-free references on our grid: BoS and Agreement-on-the-Line (AoL).

BoS — Yang et al., "Bounding Box Stability against Feature Dropout Reflects Detector
Generalization across Environments", ICLR 2024 (arXiv:2403.13803): run the detector twice on
each unlabeled target image, the second pass with dropout on backbone features; bipartite-
match boxes and average IoU. Higher stability <-> higher mAP (their FD baseline was worst).
AoL — Baek et al., NeurIPS 2022 (arXiv:2206.13089): agreement between two models on unlabeled
data tracks accuracy. Never instantiated for detectors (verified scan 2026-10-13); here:
box-level agreement (matched IoU>=0.5, same class) between pairs of source detectors on the
target. Both are POST-training estimators (need the trained detector) — reported as the
references a reviewer will demand, not as competitors to the pre-training dIV metric.

Outputs per (source, target): bos_stability, aol_mean_agreement. Pairs with results/calib_anchored.
"""
from __future__ import annotations
import argparse, json, glob, sys
from pathlib import Path
import numpy as np, torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from PIL import Image
import torchvision.transforms.functional as TF
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src"))
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection

def log(m): print(f"[bos] {m}", flush=True)
def load(ck, dev):
    m=fasterrcnn_resnet50_fpn(weights=None); m.roi_heads.box_predictor=FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,4)
    d=torch.load(ck,map_location=dev,weights_only=False); m.load_state_dict(d.get("state_dict") or d.get("model") or d); return m.to(dev).eval()

class FeatDropout(torch.nn.Module):
    def __init__(s, body, p): super().__init__(); s.body=body; s.p=p
    def forward(s, x):
        out=s.body(x); return type(out)((k, torch.nn.functional.dropout(v, s.p, training=True)) for k,v in out.items())

def match_iou(b1,b2,l1,l2,thr=0.5):
    if len(b1)==0 or len(b2)==0: return 0.0, 0.0
    iou=box_iou(b1,b2); same=(l1[:,None]==l2[None,:]); iou=iou*same
    # greedy bipartite
    used=set(); tot=0.0; n=0
    for i in iou.max(1).values.argsort(descending=True).tolist():
        j=int(iou[i].argmax());
        if j in used: continue
        used.add(j); tot+=float(iou[i,j]); n+=1
    return tot/max(1,len(b1)), float(((iou>=thr).any(1)).float().mean())

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ckpt-glob", default="checkpoints/anch_*/last.pt")
    p.add_argument("--targets", nargs="+", default=["KITTI"]+[f"Town0{t}_{w}" for t in (1,3) for w in ("ClearNoon","OvercastNoon","SoftRainNoon","ClearNight")])
    p.add_argument("--carla-root", default=str(Path.home()/"datasets/carla_extracted")); p.add_argument("--extra-root", default="/mnt/features/carla_new")
    p.add_argument("--kitti-root", default=str(Path.home()/"datasets/kitti/extracted/training")); p.add_argument("--kitti-splits", default=str(Path.home()/"datasets/kitti/splits.json"))
    p.add_argument("--n-images", type=int, default=100); p.add_argument("--dropout", type=float, default=0.3); p.add_argument("--score", type=float, default=0.3)
    p.add_argument("--out", default="results/calib_anchored/bos_aol.json")
    a=p.parse_args(); dev=torch.device("cuda"); torch.manual_seed(0)
    cks=sorted(glob.glob(a.ckpt_glob)); srcs=[__import__("re").sub(r"^anch[a-z]*_","",Path(c).parent.name) for c in cks]; models={s:load(c,dev) for s,c in zip(srcs,cks)}
    log(f"{len(models)} detectors; {len(a.targets)} targets; {a.n_images} imgs each")
    def target_images(T):
        if T=="KITTI":
            ds=KITTIDetection(kitti_root=a.kitti_root, split_json=a.kitti_splits, split="val"); idx=range(0,len(ds),max(1,len(ds)//a.n_images))
        elif T=="Cityscapes":
            from s2rgap.data.cityscapes_coco import CityscapesCOCO
            ds=CityscapesCOCO(Path.home()/"datasets/cityscapes/extracted/leftImg8bit",REPO/"nvme/datasets/aldi_labels/cityscapes_val_instances.json",split="val"); idx=range(0,len(ds),max(1,len(ds)//a.n_images))
        else:
            root=a.carla_root if Path(a.carla_root,T).exists() else a.extra_root
            ds=CARLADetection(root=root, cells=[T]); idx=range(800, min(1000,len(ds)), max(1,200//a.n_images))
        return [ds[i][0] for i in list(idx)[:a.n_images]]
    out={}
    with torch.inference_mode():
        for T in a.targets:
            imgs=[im.to(dev) for im in target_images(T)]
            preds={}
            for s,m in models.items():
                clean=[{k:v[o["scores"]>=a.score] for k,v in o.items()} for o in sum([m(imgs[i:i+8]) for i in range(0,len(imgs),8)],[])]
                body=m.backbone.body; m.backbone.body=FeatDropout(body,a.dropout)
                drop=[{k:v[o["scores"]>=a.score] for k,v in o.items()} for o in sum([m(imgs[i:i+8]) for i in range(0,len(imgs),8)],[])]
                m.backbone.body=body
                bos=np.mean([match_iou(c["boxes"],d["boxes"],c["labels"],d["labels"])[0] for c,d in zip(clean,drop)])
                preds[s]=clean; out.setdefault(s,{})[T]={"bos_stability":float(bos)}
            for s in models:
                ag=[np.mean([match_iou(preds[s][i]["boxes"],preds[o][i]["boxes"],preds[s][i]["labels"],preds[o][i]["labels"])[1] for i in range(len(imgs))]) for o in models if o!=s]
                out[s][T]["aol_mean_agreement"]=float(np.mean(ag))
            log(f"{T:<22} BoS " + " ".join(f"{s.split('_')[0][-1]}{s.split('_')[1][:2]}:{out[s][T]['bos_stability']:.2f}" for s in models))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,"w"),indent=1); log(f"saved -> {a.out}")
if __name__=="__main__": raise SystemExit(main())
