#!/usr/bin/env python3
"""Join anchored drops with (a) 3-seed dIV_fg and (b) label-free dense-probe ATC gap."""
import json, glob, numpy as np
from scipy import stats
R="results/calib_anchored"
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
cells=sorted({f.split("eval_")[1].split("__on__")[0] for f in glob.glob(f"{R}/eval_*.json")})
anchor={s:m(f"{R}/eval_{s}__on__{s}.json") for s in cells if glob.glob(f"{R}/eval_{s}__on__{s}.json")}; cells=[c for c in cells if c in anchor]
seeds=[json.load(open(f))["matrix"] for f in sorted(glob.glob(f"{R}/probe_matrix_anchored_s*.json"))]
atc=json.load(open(f"{R}/dense_probe_atc_s42.json"))
def corr(name,x,y):
    x,y=np.array(x,float),np.array(y,float); ok=~np.isnan(x)&~np.isnan(y)
    r,p=stats.pearsonr(x[ok],y[ok]); rho,ps=stats.spearmanr(x[ok],y[ok]); print(f"  {name:<34} r={r:+.3f} (p={p:.3g})  rho={rho:+.3f} (p={ps:.3g})  n={ok.sum()}")
pairs=[(S,T) for S in cells for T in cells if S!=T and glob.glob(f"{R}/eval_{S}__on__{T}.json")]
drop=[anchor[S]-m(f"{R}/eval_{S}__on__{T}.json") for S,T in pairs]
print(f"=== 56 CARLA pairs, anchored drop ({len(seeds)} probe seeds) ===")
corr("dIV_fg seed-mean (needs target boxes)",[np.mean([s[S][T]["dIV_fg"] for s in seeds]) for S,T in pairs],drop)
corr("gap_atc LABEL-FREE",[atc[S][T]["gap_atc"] for S,T in pairs],drop)
corr("gap_true (dense probe, labeled)",[atc[S][T]["gap_true"] for S,T in pairs],drop)
corr("ATC est vs true patch acc (ATC validity)",[atc[S][T]["atc_est_acc"] for S,T in pairs],[atc[S][T]["true_acc"] for S,T in pairs])
for Tgt,key in [("KITTI","kittival"),("Cityscapes","cityscapesval")]:
    fs={s:m(f"{R}/src_{s}_{key}.json") for s in cells if glob.glob(f"{R}/src_{s}_{key}.json")}
    if len(fs)<6 or Tgt not in atc[cells[0]]: continue
    y=[anchor[s]-fs[s] for s in fs]; print(f"=== {Tgt}: n={len(fs)} sources, anchored drop ===")
    corr("dIV_fg seed-mean",[np.mean([sd[s][Tgt]["dIV_fg"] for sd in seeds]) for s in fs],y)
    corr("gap_atc LABEL-FREE",[atc[s][Tgt]["gap_atc"] for s in fs],y)
    corr("gap_true",[atc[s][Tgt]["gap_true"] for s in fs],y)
