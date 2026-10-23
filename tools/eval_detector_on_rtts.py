#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ledger §15.69 (E3): evaluate a detector on RTTS (real hazy images, VOC XML). car -> Car, person -> Pedestrian; other classes and VOC
boxes kept incl. difficult (mirror lacks the flag); mAP@0.5 over Car + Pedestrian with s2rgap.eval_detection.evaluate_map (Cyclist predictions ignored)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src"))
from s2rgap.eval_detection import evaluate_map
from s2rgap.detector import load_detector_from_ckpt
MAP={"car":1,"person":2}; CLASSES={"background":0,"Car":1,"Pedestrian":2}
def load(root):
    """COCO-format annotations of the HF mirror (instances_trainval2017 + instances_test2017 = all 4,322 images); bbox xywh -> xyxy.
    The mirror carries no 'difficult' flag, so every box is kept (ledger §15.69 deviation)."""
    root=Path(root); img=root/"VOCdevkit/VOC2007/JPEGImages"; items=[]
    for f in ("trainval","test"):
        d=json.load(open(root/f"coco/instances_{f}2017.json")); cats={c["id"]:c["name"].lower() for c in d["categories"]}; by={}
        for a in d["annotations"]:
            n=cats[a["category_id"]]
            if n in MAP: x,y,w,h=a["bbox"]; by.setdefault(a["image_id"],[]).append(([x,y,x+w,y+h],MAP[n]))
        for im in d["images"]:
            r=by.get(im["id"],[]); items.append((img/im["file_name"],np.array([b for b,_ in r],np.float32).reshape(-1,4),np.array([l for _,l in r],np.int64)))
    return sorted(items,key=lambda z:z[0].name)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--ckpt",required=True); p.add_argument("--root",default="/mnt/features/rtts"); p.add_argument("--result",required=True); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--count-only",action="store_true")
    a=p.parse_args(); items=load(a.root)
    if a.count_only:
        print(f"{len(items)} images; Car {sum((l==1).sum() for _,_,l in items)}; Pedestrian {sum((l==2).sum() for _,_,l in items)}"); return 0
    dev=torch.device("cuda"); m=load_detector_from_ckpt(a.ckpt,dev); P=[];T=[]; t0=time.time()
    with torch.inference_mode():
        for s in range(0,len(items),a.batch_size):
            ch=items[s:s+a.batch_size]; ims=[torch.from_numpy(np.asarray(Image.open(f).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).to(dev) for f,_,_ in ch]
            for pr,(f,b,l) in zip(m(ims),ch):
                P.append({k:v.detach().cpu() for k,v in pr.items()}); T.append({"boxes":torch.from_numpy(b),"labels":torch.from_numpy(l)})
    met=evaluate_map(P,T,CLASSES,iou_threshold=0.5)
    json.dump({"source_ckpt":a.ckpt,"dataset":"RTTS (car->Car, person->Pedestrian; difficult dropped)","n_frames":len(items),"metrics":met,"seconds":time.time()-t0},open(a.result,"w"),indent=2)
    print(f"[rtts] {a.ckpt}: {met}"); return 0
if __name__=="__main__": raise SystemExit(main())
