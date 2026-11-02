# Figure F11 (Chapter 6, generative real twins): label survival on selected frames; reads the pilot survival files under /mnt/features/cosmos.
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as pa
from PIL import Image
O="/mnt/features/cosmos/sol_eval"; X="/mnt/features/cosmos/sol_results_20261022/results/realtwin"; C="/mnt/features/carla_doses"
dirs={"D":lambda t:f"/mnt/features/cosmos/pilot/{t}_d50_realtwin","M1":lambda t:f"{X}/M1/{t}_d54_edge_depth_seg"}
def boxes(t,f):
    out=[]
    for ln in open(f"{C}/{t}_d00_ref/{f}.txt"):
        p=ln.split()
        if p and p[0] in ("Car","Pedestrian"):
            x1,y1,x2,y2=map(float,p[4:8])
            if y2-y1>=25 and float(p[1])<=.5: out.append((p[0],x1,y1,x2,y2))
    return out
cand=[]
for t in ("Town03","Town05"):
    S={v:{r["name"]:r for r in json.load(open(f"{O}/surv_{v}_{t}.json"))["rows"]} for v in ("D","M1")}
    for n,rD in S["D"].items():
        rM=S["M1"][n]; o=np.array(rD["found"]["orig"]); 
        if o.sum()<3: continue
        d=int(np.array(rD["found"]["rest"])[o].sum()); m=int(np.array(rM["found"]["rest"])[o].sum())
        cand.append((m-d, o.sum(), t, n))
cand.sort(reverse=True); pick=[]
for c in cand:
    if c[0]<=0: break
    if sum(1 for p in pick if p[2]==c[2])<2 and len(pick)<3: pick.append(c)
print("picked", pick)
fig,axes=plt.subplots(len(pick),3,figsize=(15,1.95*len(pick)+0.5))
for i,(_,_,t,n) in enumerate(pick):
    rD=json.load(open(f"{O}/surv_D_{t}.json")); rM=json.load(open(f"{O}/surv_M1_{t}.json"))
    fD=[r for r in rD["rows"] if r["name"]==n][0]; fM=[r for r in rM["rows"] if r["name"]==n][0]; B=boxes(t,n)
    for j,(lab,img,found) in enumerate([("CARLA",f"{C}/{t}_d00_ref/{n}.jpg",fD["found"]["orig"]),("Edge only",f"{dirs['D'](t)}/{n}.jpg",fD["found"]["rest"]),("Edge + depth + segmentation",f"{dirs['M1'](t)}/{n}.jpg",fM["found"]["rest"])]):
        ax=axes[i,j]; ax.imshow(Image.open(img)); ax.set_xticks([]); ax.set_yticks([])
        for k,(cl,x1,y1,x2,y2) in enumerate(B):
            if not fD["found"]["orig"][k]: continue
            col="#2ecc40" if (j==0 or found[k]) else "#ff2d2d"
            ax.add_patch(pa.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,lw=1.6,ec=col))
        if i==0: ax.set_title(lab,fontsize=12)
        if j==0: ax.set_ylabel(f"{t}\n{n[-4:]}",fontsize=9)
fig.text(0.5,0.005,"Boxes: labeled objects a COCO-trained judge finds in the CARLA frame; green = still found after the restyle, red = lost. Selected examples.",ha="center",fontsize=9)
import os; os.makedirs("figures/cvpr",exist_ok=True)
plt.tight_layout(rect=(0,0.03,1,1)); fig.savefig("figures/cvpr/F11_real_twin.pdf"); fig.savefig("figures/cvpr/F11_real_twin.png",dpi=110)
