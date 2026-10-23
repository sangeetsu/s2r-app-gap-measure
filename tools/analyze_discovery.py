#!/usr/bin/env python3
"""Label-free DISCOVERY-box gap (no target labels) vs anchored drops; reference = labeled dIV_fg."""
import json, glob, numpy as np
from scipy import stats
R="results/calib_anchored"
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
cells=sorted({f.split("eval_")[1].split("__on__")[0] for f in glob.glob(f"{R}/eval_*.json")})
anchor={s:m(f"{R}/eval_{s}__on__{s}.json") for s in cells if glob.glob(f"{R}/eval_{s}__on__{s}.json")}; cells=[c for c in cells if c in anchor]
D=json.load(open(f"{R}/discovery_box_gap_s42.json")); seeds=[json.load(open(f))["matrix"] for f in sorted(glob.glob(f"{R}/probe_matrix_anchored_s*.json"))]
def corr(name,x,y):
    x,y=np.array(x,float),np.array(y,float); ok=~np.isnan(x)&~np.isnan(y)
    if ok.sum()<4: print(f"  {name:<30} n/a"); return
    r,p=stats.pearsonr(x[ok],y[ok]); rho,ps=stats.spearmanr(x[ok],y[ok]); print(f"  {name:<30} r={r:+.3f} (p={p:.3g})  rho={rho:+.3f} (p={ps:.3g})  n={ok.sum()}")
print("discovery sanity (fraction of discovered boxes hitting a GT box, IoU>=.5):")
for T in ["KITTI","Cityscapes"]+cells:
    v=[D[s][T]["disc_gt_hit"] for s in cells if T in D[s]]; n=[D[s][T]["n_boxes"] for s in cells if T in D[s]]
    if v: print(f"  {T:<22} hit {100*np.mean(v):4.0f}%  boxes {int(np.mean(n))}")
pairs=[(S,T) for S in cells for T in cells if S!=T and T in D[S] and glob.glob(f"{R}/eval_{S}__on__{T}.json")]
drop=[anchor[S]-m(f"{R}/eval_{S}__on__{T}.json") for S,T in pairs]
print(f"\n=== {len(pairs)} CARLA pairs, anchored drop ===")
corr("dIV_fg (labeled reference)",[np.mean([s[S][T]["dIV_fg"] for s in seeds]) for S,T in pairs],drop)
corr("gap_atc  DISCOVERY, label-free",[D[S][T]["gap_atc"] for S,T in pairs],drop)
corr("gap_conf DISCOVERY, label-free",[D[S][T]["gap_conf"] for S,T in pairs],drop)
for Tgt,key in [("KITTI","kittival"),("Cityscapes","cityscapesval")]:
    fs={s:m(f"{R}/src_{s}_{key}.json") for s in cells if glob.glob(f"{R}/src_{s}_{key}.json") and Tgt in D[s]}
    if len(fs)<6: continue
    y=[anchor[s]-fs[s] for s in fs]; print(f"=== {Tgt}: n={len(fs)} sources, anchored drop ===")
    corr("dIV_fg (labeled reference)",[np.mean([sd[s][Tgt]["dIV_fg"] for sd in seeds]) for s in fs],y)
    corr("gap_atc  DISCOVERY, label-free",[D[s][Tgt]["gap_atc"] for s in fs],y)
    corr("gap_conf DISCOVERY, label-free",[D[s][Tgt]["gap_conf"] for s in fs],y)
