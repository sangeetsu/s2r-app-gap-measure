#!/usr/bin/env python3
"""F8 (illustrative): simulator-trained detectors on REAL images, examples SELECTED to show the effect (stated in the caption). Selection rule: among
seeded random candidates with >= 4 labelled cars/pedestrians, real-fog panels = largest (objects found by the fog-trained detector − by the clear-trained),
clear panels = most objects found by both (shows nothing is lost). 'Found' = same class, IoU >= .5, score >= .5. Bars = ALL images, 8-town mean (not selected)."""
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
from per_object_mechanism import detected
plt.rcParams.update({"font.size":7,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
TOWN=sys.argv[1] if len(sys.argv)>1 else "Town10HD"; T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]; m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
SRC={"KITTI":lambda c:f"results/calib_dose/src_{c}_kittival.json","Cityscapes":lambda c:f"results/calib_dose/src_{c}_cityscapesval.json",
     "Foggy Cityscapes":lambda c:f"results/calib_dose_foggy/src_{c}_foggy0.02.json","RTTS":lambda c:f"results/calib_dose_rtts/src_{c}_rtts.json"}
mean={k:(np.array([m(f(f"{t}_d00_ref")) for t in T]),np.array([m(f(f"{t}_d05_fog25")) for t in T])) for k,f in SRC.items()}
dev=torch.device("cuda"); D=[load_detector_from_ckpt(f"checkpoints/anchind_{TOWN}_{d}/last.pt",dev) for d in ("d00_ref","d05_fog25")]
def run(path,gt,gl):
    x=torch.from_numpy(np.asarray(Image.open(path).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).to(dev); out=[]
    for det in D:
        with torch.no_grad(): p=det([x])[0]
        p={k:v.cpu().numpy() for k,v in p.items()}; k=(gl==1)|(gl==2); out.append((p,int(detected(gt[k],gl[k],p["boxes"],p["labels"],p["scores"]).sum())))
    return out
rng=np.random.default_rng(0)
kv=KITTIDetection(kitti_root=str(Path.home()/"datasets/kitti/extracted/training"),split_json=str(Path.home()/"datasets/kitti/splits.json"),split="val")
kc=[]
for i in rng.choice(len(kv),300,replace=False):
    tg=kv[int(i)][1]; gl=tg["labels"].numpy()
    if ((gl==1)|(gl==2)).sum()>=4: r=run(Path.home()/"datasets/kitti/extracted/training/image_2"/tg["frame_rel"],tg["boxes"].numpy(),gl); kc.append((min(r[0][1],r[1][1])-abs(r[0][1]-r[1][1]),i,tg,r))
kbest=max(kc,key=lambda z:z[0])
cs=CityscapesCOCO(Path.home()/"datasets/cityscapes/extracted/leftImg8bit","nvme/datasets/aldi_labels/cityscapes_val_instances.json",split="val",min_box_area=25); cc=[]
for i in rng.choice(len(cs),250,replace=False):
    tg=cs[int(i)][1]; gl=tg["labels"].numpy()
    if ((gl==1)|(gl==2)).sum()>=4:
        rf=run(Path("/mnt/features/cityscapes_foggy/b0.02/leftImg8bit/val")/tg["frame_rel"],tg["boxes"].numpy(),gl); cc.append((rf[1][1]-rf[0][1],i,tg,rf))
cbest=max(cc,key=lambda z:z[0]); rc=run(Path.home()/"datasets/cityscapes/extracted/leftImg8bit/val"/cbest[2]["frame_rel"],cbest[2]["boxes"].numpy(),cbest[2]["labels"].numpy())
rt=rtts_load("/mnt/features/downloads/rtts"); rcands=[]
for i in rng.choice(len(rt),400,replace=False):
    f,b,l=rt[int(i)]
    if len(l)>=4: r=run(f,b,l); rcands.append((r[1][1]-r[0][1],i,(f,b,l),r))
rbest=max(rcands,key=lambda z:z[0])
n=lambda gl: int(((np.asarray(gl)==1)|(np.asarray(gl)==2)).sum())
rows=[("KITTI, clear",Path.home()/"datasets/kitti/extracted/training/image_2"/kbest[2]["frame_rel"],kbest[3],n(kbest[2]["labels"].numpy()),"KITTI"),
      ("Cityscapes, clear",Path.home()/"datasets/cityscapes/extracted/leftImg8bit/val"/cbest[2]["frame_rel"],rc,n(cbest[2]["labels"].numpy()),"Cityscapes"),
      ("same street in fog",Path("/mnt/features/cityscapes_foggy/b0.02/leftImg8bit/val")/cbest[2]["frame_rel"],cbest[3],n(cbest[2]["labels"].numpy()),"Foggy Cityscapes"),
      ("RTTS, real haze",rbest[2][0],rbest[3],n(rbest[2][2]),"RTTS")]
print({r[0]:[x[1] for x in r[2]] for r in rows}, "of", {r[0]:r[3] for r in rows}); print("images:",[str(r[1]) for r in rows])
COL={1:"#ffd700",2:"#00e5ff",3:"#ff40ff"}; ar=[Image.open(r[1]).size[1]/Image.open(r[1]).size[0] for r in rows]
fig=plt.figure(figsize=(7.0,sum(3.45*a for a in ar)+2.5)); gs=fig.add_gridspec(len(rows)+1,2,height_ratios=[3.45*a for a in ar]+[2.0],hspace=.28,wspace=.03,top=.94,bottom=.06)
LAB=["detector trained on CLEAR simulator images","detector trained on MILD SIMULATED FOG"]
for r,(name,p,res,ngt,key) in enumerate(rows):
    im=Image.open(p).convert("RGB")
    for c in (0,1):
        pr,hits=res[c]; k=pr["scores"]>=.5; ax=fig.add_subplot(gs[r,c]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for (x1,y1,x2,y2),l in zip(pr["boxes"][k],pr["labels"][k]): ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,lw=.9,ec=COL.get(int(l),"w")))
        ax.set_title(f"{name}: finds {hits} of {ngt} cars/people",fontsize=6.5,color="k" if c==0 else "#1f4e9c")
        if r==0: ax.text(.5,1.24,LAB[c],transform=ax.transAxes,ha="center",fontsize=7.5,fontweight="bold",color="k" if c==0 else "#1f4e9c")
        for s in ax.spines.values(): s.set_visible(False)
ax=fig.add_subplot(gs[-1,:]); x=np.arange(4); w=.36
for j,(lab,col) in enumerate((("trained on clear simulator images","#7f7f7f"),("trained on mild simulated fog","#1f77b4"))):
    v=[mean[k][j] for k in SRC]; ax.bar(x+(j-.5)*w,[a.mean() for a in v],w,color=col,label=lab)
    for i,a in enumerate(v): ax.scatter(np.full(8,x[i]+(j-.5)*w)+rng.uniform(-.08,.08,8),a,s=4,color="k",zorder=3,linewidths=0)
for i,k in enumerate(SRC):
    d=mean[k][1]-mean[k][0]; ax.text(x[i],max(mean[k][0].max(),mean[k][1].max())+1,f"{d.mean():+.1f} pp ({int((d>0).sum())}/8 towns)",ha="center",fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(["KITTI\nclear, real","Cityscapes\nclear, real","Foggy Cityscapes β=.02\nsynthetic fog on real photos","RTTS\nreal haze photos"]); ax.set_ylabel("real-world mAP, all images (%)")
ax.legend(frameon=False,loc="upper center",bbox_to_anchor=(.5,-.3),ncol=2,fontsize=6.5); ax.set_ylim(0,max(a.max() for v in mean.values() for a in v)+7)
ax.set_title("All images, 8 towns (bars = mean, dots = towns): same on clear real images, clearly better on real fog",fontsize=7)
fig.suptitle(f"Simulator-trained detectors on real images ({TOWN}). Examples selected to illustrate the effect; boxes: car yellow, pedestrian cyan, cyclist magenta.",fontsize=7,y=1.012)
fig.savefig("figures/cvpr/F8_real_world_test_selected.pdf",bbox_inches="tight"); fig.savefig("figures/cvpr/F8_real_world_test_selected.png",bbox_inches="tight",dpi=150); print("[fig] F8 selected")
