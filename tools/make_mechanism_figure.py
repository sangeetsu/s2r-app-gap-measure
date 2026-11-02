#!/usr/bin/env python3
"""F9 (mechanism, ledger §15.82): (a) loss rate by displacement quintile per family; (b) ROC of displacement vs contrast, luminance, size;
F9_mechanism_example: Town03 frame 852 under fog 50 — each GT object the clean-trained detector finds in the clean render, colored by its feature displacement,
dashed red = lost in the fog render. Panel (c) is one illustrative frame (the F1 frame); (a)/(b) use all 36,216 cases."""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from per_object_mechanism import detected
plt.rcParams.update({"font.size":7,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
z=np.load("results/calib_dose/per_object.npz"); y=z["lost"].astype(int)
fig=plt.figure(figsize=(7.0,2.5)); gs=fig.add_gridspec(1,2,wspace=.3)
ax=fig.add_subplot(gs[0,0]); q=np.quantile(z["disp"],np.linspace(0,1,6)); FAM={"all":None,"fog":[4,5,6],"sun / night":[0,1,2,3],"wet / rain":[7,8]}; COLF={"all":"k","fog":"#d62728","sun / night":"#1f77b4","wet / rain":"#2ca02c"}
for f,ds in FAM.items():
    i0=np.ones(len(y),bool) if ds is None else np.isin(z["dose"].astype(int),ds); r=[]
    for b in range(5):
        i=i0&(z["disp"]>=q[b])&(z["disp"]<=q[b+1]); r.append(100*y[i].mean() if i.sum()>30 else np.nan)
    ax.plot(range(1,6),r,"-o",ms=3,color=COLF[f],lw=1.6 if f=="all" else .9,label=f)
ax.set_xticks(range(1,6)); ax.set_xticklabels([f"Q{b+1}\n{q[b]:.0f}–{q[b+1]:.0f}" for b in range(5)]); ax.set_xlabel("per-object feature displacement (quintiles, all doses)"); ax.set_ylabel("objects lost by the detector (%)")
ax.set_title("(a) loss rate vs feature displacement"); ax.legend(frameon=False,fontsize=6)
ax=fig.add_subplot(gs[0,1])
for k,lab,col in (("disp","feature displacement (DINOv3)","k"),("dcontrast","|Δ contrast| in box","#ff7f0e"),("height","small object (−height)","#9467bd"),("dlum","|Δ brightness| in box","#8c564b")):
    x=-z["height"] if k=="height" else z[k]; fpr,tpr,_=roc_curve(y,x); ax.plot(fpr,tpr,color=col,lw=1.2 if k=="disp" else .9,label=f"{lab}: AUC {roc_auc_score(y,x):.2f}")
ax.plot([0,1],[0,1],":",color="gray",lw=.6); ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate"); ax.set_title("(b) which objects are lost? (36,216 cases)"); ax.legend(frameon=False,fontsize=5.8,loc="lower right")
# (c) one frame
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.detector import load_detector_from_ckpt
from linear_probe_object_class import bbox_to_feat_region
from domain_matrix_probe import gather, CARLA_IMG_HW
ROOT="/mnt/features/carla_doses"; t="Town03"; k=852; dose="d06_fog50"; dev=torch.device("cuda")
lk=SpatialFeatureLookup.from_npz(f"{ROOT}/spatial_features_dinov3_80cells.npz"); gh,gw=lk.target_hw
dsr=CARLADetection(root=ROOT,cells=[f"{t}_d00_ref"]); dsd=CARLADetection(root=ROOT,cells=[f"{t}_{dose}"]); idx=[i for i,r in enumerate(dsr.records) if f"frame_{k:06d}" in str(r[2])][0]
Xp,_=gather(lk,torch.utils.data.Subset(dsr,range(0,800)),lambda d,i:CARLA_IMG_HW,np.random.default_rng(7),4,"probe"); sd=Xp.std(0)+1e-6
det=load_detector_from_ckpt(f"checkpoints/anchind_{t}_d00_ref/last.pt",dev); (ir,tr),(idd,td)=dsr[idx],dsd[idx]; gt=tr["boxes"].numpy(); gl=tr["labels"].numpy()
with torch.no_grad(): pr=det([ir.to(dev)])[0]; pd=det([idd.to(dev)])[0]
hr=detected(gt,gl,*[pr[x].cpu().numpy() for x in ("boxes","labels","scores")]); hd=detected(gt,gl,*[pd[x].cpu().numpy() for x in ("boxes","labels","scores")])
zr=np.array(lk.z_mmap[lk.path_to_idx[tr["frame_rel"]]],dtype=np.float32); zd=np.array(lk.z_mmap[lk.path_to_idx[td["frame_rel"]]],dtype=np.float32); H,W=ir.shape[-2:]
disp=[]
for b in gt:
    i1,j1,i2,j2=bbox_to_feat_region(tuple(b),(int(H),int(W)),(gh,gw)); disp.append(float(np.linalg.norm((zd[i1:i2,j1:j2].reshape(-1,zd.shape[-1]).mean(0)-zr[i1:i2,j1:j2].reshape(-1,zr.shape[-1]).mean(0))/sd)))
fig.savefig("figures/cvpr/F9_mechanism.pdf",bbox_inches="tight"); fig.savefig("figures/cvpr/F9_mechanism.png",bbox_inches="tight",dpi=150)
norm=plt.Normalize(q[0],q[-2]); cmap=cm.viridis; fig2,axs=plt.subplots(1,2,figsize=(7.0,1.55)); fig2.subplots_adjust(wspace=.03)
for c,(img,title) in enumerate(((ir,"clean render"),(idd,"same frame under fog 50"))):
    ax=axs[c]; ax.imshow(img.permute(1,2,0).numpy()); ax.set_xticks([]); ax.set_yticks([])
    for j,b in enumerate(gt):
        if not hr[j]: continue
        col=cmap(norm(disp[j])); ax.add_patch(plt.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,lw=1.4,ec=col))
        if c==1 and not hd[j]: ax.add_patch(plt.Rectangle((b[0]-3,b[1]-3),b[2]-b[0]+6,b[3]-b[1]+6,fill=False,lw=1.0,ec="#ff3030",ls="--"))
    ax.set_title((f"{title}: objects the clean-trained detector finds" if c==0 else f"{title}: red dashed = lost by the detector"),fontsize=6.5)
    for s in ax.spines.values(): s.set_visible(False)
sm=cm.ScalarMappable(norm=norm,cmap=cmap); cb=fig2.colorbar(sm,ax=list(axs),orientation="horizontal",fraction=.08,pad=.04,aspect=60); cb.set_label("per-object feature displacement (box color)",fontsize=6); cb.ax.tick_params(labelsize=6)
print("frame objects (found in clean):",int(hr.sum()),"lost under fog:",int((hr&~hd).sum()),"displacements lost:",[round(d,1) for d,a,b_ in zip(disp,hr,hd) if a and not b_],"kept:",[round(d,1) for d,a,b_ in zip(disp,hr,hd) if a and b_])
fig2.savefig("figures/cvpr/F9_mechanism_example.pdf",bbox_inches="tight"); fig2.savefig("figures/cvpr/F9_mechanism_example.png",bbox_inches="tight",dpi=150); print("[fig] F9 + example")
