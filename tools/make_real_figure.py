#!/usr/bin/env python3
"""F8: the simulator-trained detectors on REAL images. Town chosen by rule (fog25's real-fog gain closest to the 8-town mean); example images chosen by
seeded random draw (seed 0) among images with >= 4 Car/Pedestrian boxes — not hand-picked. Boxes: score >= .5. Bars: town-mean mAP (seed 42, 8 towns)."""
import json, sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.detector import load_detector_from_ckpt
from s2rgap.data.kitti import KITTIDetection
from s2rgap.data.cityscapes_coco import CityscapesCOCO
from eval_detector_on_rtts import load as rtts_load
plt.rcParams.update({"font.size":7,"axes.titlesize":7.5,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]; m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
SRC={"KITTI (clear, real)":lambda c:f"results/calib_dose/src_{c}_kittival.json","Cityscapes (clear, real)":lambda c:f"results/calib_dose/src_{c}_cityscapesval.json",
     "Foggy Cityscapes β=.02 (synthetic fog on real photos)":lambda c:f"results/calib_dose_foggy/src_{c}_foggy0.02.json","RTTS (real haze photos)":lambda c:f"results/calib_dose_rtts/src_{c}_rtts.json"}
mean={k:(np.array([m(f(f"{t}_d00_ref")) for t in T]),np.array([m(f(f"{t}_d05_fog25")) for t in T])) for k,f in SRC.items()}
fgain={t:np.mean([m(SRC[k](f"{t}_d05_fog25"))-m(SRC[k](f"{t}_d00_ref")) for k in list(SRC)[2:]]) for t in T}; avg=np.mean(list(fgain.values()))
town=min(T,key=lambda t:abs(fgain[t]-avg)); print("town by rule:",town,{t:round(v,2) for t,v in fgain.items()},"mean",round(avg,2))
rng=np.random.default_rng(0)
kv=KITTIDetection(kitti_root=str(Path.home()/"datasets/kitti/extracted/training"),split_json=str(Path.home()/"datasets/kitti/splits.json"),split="val")
ki=[i for i in range(len(kv)) if int(((kv[i][1]["labels"]==1)|(kv[i][1]["labels"]==2)).sum())>=4]; k_idx=int(rng.choice(ki))
cs=CityscapesCOCO(Path.home()/"datasets/cityscapes/extracted/leftImg8bit","nvme/datasets/aldi_labels/cityscapes_val_instances.json",split="val",min_box_area=25)
ci=[i for i in range(len(cs)) if int(((cs[i][1]["labels"]==1)|(cs[i][1]["labels"]==2)).sum())>=4]; c_idx=int(rng.choice(ci)); crel=cs[c_idx][1]["frame_rel"]
rt=rtts_load("/mnt/features/downloads/rtts"); ri=[i for i,(f,b,l) in enumerate(rt) if len(l)>=4]; r_idx=int(rng.choice(ri))
imgs=[("KITTI (clear, real)",Path.home()/"datasets/kitti/extracted/training/image_2"/kv[k_idx][1]["frame_rel"]),
      ("Cityscapes (clear, real)",Path.home()/"datasets/cityscapes/extracted/leftImg8bit/val"/crel),
      ("Foggy Cityscapes β=.02 (synthetic fog on real photos)",Path("/mnt/features/cityscapes_foggy/b0.02/leftImg8bit/val")/crel),
      ("RTTS (real haze photos)",rt[r_idx][0])]
print("images:",[str(p) for _,p in imgs])
dev=torch.device("cuda"); dets={lab:load_detector_from_ckpt(f"checkpoints/anchind_{town}_{d}/last.pt",dev) for lab,d in (("trained on clear simulator images","d00_ref"),("trained on mild simulated fog (fog 25)","d05_fog25"))}
COL={1:"#ffd700",2:"#00e5ff",3:"#ff40ff"}
ar=[Image.open(p).size[1]/Image.open(p).size[0] for _,p in imgs]
fig=plt.figure(figsize=(7.0,sum(3.45*a for a in ar)+2.4)); gs=fig.add_gridspec(len(imgs)+1,2,height_ratios=[3.45*a for a in ar]+[2.0],hspace=.28,wspace=.03,top=.94,bottom=.06)
for r,(name,p) in enumerate(imgs):
    im=Image.open(p).convert("RGB"); x=torch.from_numpy(np.asarray(im,dtype=np.float32)/255.).permute(2,0,1).to(dev)
    for c,(lab,det) in enumerate(dets.items()):
        with torch.no_grad(): pr=det([x])[0]
        k=pr["scores"]>=.5; ax=fig.add_subplot(gs[r,c]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for (x1,y1,x2,y2),l in zip(pr["boxes"][k].cpu().numpy(),pr["labels"][k].cpu().numpy()): ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,lw=.8,ec=COL.get(int(l),"w")))
        a_=mean[name][c]; short={"KITTI (clear, real)":"KITTI, clear","Cityscapes (clear, real)":"Cityscapes, clear","Foggy Cityscapes β=.02 (synthetic fog on real photos)":"same street in fog","RTTS (real haze photos)":"RTTS, real haze"}[name]
        ax.set_title(f"{short}: dataset mAP {a_.mean():.1f}",fontsize=6.5)
        if r==0: ax.text(.5,1.24,lab,transform=ax.transAxes,ha="center",fontsize=7.5,fontweight="bold")
        for s in ax.spines.values(): s.set_visible(False)
ax=fig.add_subplot(gs[-1,:]); x=np.arange(len(SRC)); w=.36
for j,(lab,col) in enumerate((("trained on clear simulator images","#7f7f7f"),("trained on mild simulated fog (fog 25)","#1f77b4"))):
    v=[mean[k][j] for k in SRC]; ax.bar(x+(j-.5)*w,[a.mean() for a in v],w,color=col,label=lab)
    for i,a in enumerate(v): ax.scatter(np.full(8,x[i]+(j-.5)*w)+rng.uniform(-.08,.08,8),a,s=4,color="k",zorder=3,linewidths=0)
for i,k in enumerate(SRC):
    d=mean[k][1]-mean[k][0]; ax.text(x[i],max(mean[k][0].max(),mean[k][1].max())+1,f"{d.mean():+.1f} pp\n({int((d>0).sum())}/8 towns better)",ha="center",fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(["KITTI\n(clear, real)","Cityscapes\n(clear, real)","Foggy Cityscapes β=.02\n(synthetic fog on real photos)","RTTS\n(real haze photos)"]); ax.set_ylabel("real-world mAP (%)"); ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.3),ncol=2,fontsize=6.5)
ax.set_ylim(0,max(a.max() for v in mean.values() for a in v)+7)
fig.suptitle(f"Simulator-trained detectors tested on real images ({town}, chosen by rule; examples drawn at random, seed 0).\nCar yellow, pedestrian cyan, cyclist magenta. Panel mAP = 8-town mean over all images.",fontsize=7,y=1.012)
fig.savefig("figures/cvpr/F8_real_world_test.pdf",bbox_inches="tight"); fig.savefig("figures/cvpr/F8_real_world_test.png",bbox_inches="tight",dpi=150); print("[fig] F8")
