#!/usr/bin/env python3
"""Score the pre-registered rendering-quality intervention (ledger §15.40)."""
import json, glob, os, numpy as np
from pathlib import Path
from scipy import stats
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
def nbox(root,c): return sum(1 for f in sorted(glob.glob(f"{root}/{c}/frame_*.txt"))[:800] for l in open(f) if l.split() and l.split()[0] in ("Car","Pedestrian","Cyclist"))
R="results/calib_imnet_lowq"; G="results/calib_imnet"; OLD=f"{Path.home()}/datasets/carla_extracted"; NEW="/mnt/features/carla_new"; LOW="/mnt/features/carla_lowq"
D=json.load(open("results/domain_distance_dinov3_all.json"))["metrics"]["image"]; F=json.load(open(f"{R}/fid_to_real_low.json"))["cells"] if os.path.exists(f"{R}/fid_to_real_low.json") else {}
cells=[c for c in ["Town01_ClearNoon","Town02_ClearNoon","Town03_ClearNoon","Town04_ClearNoon","Town05_ClearNoon","Town10HD_ClearNoon","Town06_ClearNoon","Town07_ClearNoon"] if os.path.exists(f"{R}/src_{c}_kittival.json")]
H="results/calib_imnet_towns2"; N2="/mnt/features/carla_new2"; P2=json.load(open("results/calib_imnet/prereg_predictions_towns2.json"))["cells"]
Gd=lambda c: H if os.path.exists(f"{H}/src_{c}_kittival.json") else G
er=lambda c: OLD if os.path.isdir(f"{OLD}/{c}") else (NEW if os.path.isdir(f"{NEW}/{c}") else N2)
print(f"{'cell':<20}{'boxes E/L':>12}  {'own E/L':>11}  {'E-det on L':>10} {'L-det on E':>10}  {'KITTI E/L':>11} {'CS E/L':>11}  {'FID->KITTI E/L':>15} {'FID->CS E/L':>13}")
dk,dc,fk,fc=[],[],[],[]
for c in cells:
    bE,bL=nbox(er(c),c),nbox(LOW,c); oE=m(f"{Gd(c)}/eval_{c}__on__{c}.json"); oL=m(f"{R}/eval_{c}__on__{c}_low.json")
    eL=m(f"{R}/eval_epic_{c}__on__{c}_low.json"); lE=m(f"{R}/eval_{c}__on__{c}_epic.json")
    kE,kL=m(f"{Gd(c)}/src_{c}_kittival.json"),m(f"{R}/src_{c}_kittival.json"); cE,cL=m(f"{Gd(c)}/src_{c}_cityscapesval.json"),m(f"{R}/src_{c}_cityscapesval.json")
    fE={T:((D.get(f"{T}||{c}") or D.get(f"{c}||{T}") or {}).get("fid") or P2[c][T]["fid"]) for T in ("KITTI","Cityscapes")}; fL={T:F.get(c,{}).get(T,{}).get("fid",np.nan) for T in ("KITTI","Cityscapes")}
    dk.append(kE-kL); dc.append(cE-cL); fk.append(fL["KITTI"]-fE["KITTI"]); fc.append(fL["Cityscapes"]-fE["Cityscapes"])
    print(f"{c:<20}{bE:>6}/{bL:<5}  {oE:5.1f}/{oL:<5.1f}  {eL:10.1f} {lE:10.1f}  {kE:5.1f}/{kL:<5.1f} {cE:5.1f}/{cL:<5.1f}  {fE['KITTI']:7.2f}/{fL['KITTI']:<7.2f} {fE['Cityscapes']:6.2f}/{fL['Cityscapes']:<6.2f}")
for T,d,f in (("KITTI",dk,fk),("Cityscapes",dc,fc)):
    d=np.array(d); f=np.array(f); t=stats.ttest_1samp(d,0) if len(d)>=3 else None
    print(f"\n{T}: Epic - Low real mAP = {d.mean():+.2f} ± {d.std(ddof=1) if len(d)>1 else 0:.2f} pp over {len(d)} towns"+(f"; paired t p={t.pvalue:.4f}" if t else ""))
    ok=~np.isnan(f)
    if ok.sum()>=4: r=stats.pearsonr(f[ok],-d[ok]); print(f"  r(ΔFID to {T} [Low-Epic], ΔmAP [Low-Epic]) = {r[0]:+.3f} (p={r[1]:.3f}); mean ΔFID {f[ok].mean():+.2f}")
