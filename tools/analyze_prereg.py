#!/usr/bin/env python3
"""Score the pre-registered FID->real-mAP predictions on held-out towns (2026-10-14). Observed values come from detectors trained
AFTER the prediction file was written (timestamps printed)."""
import json, glob, os, numpy as np
from scipy import stats
import sys
PF=sys.argv[1] if len(sys.argv)>1 else "results/calib_imnet/prereg_predictions_towns2.json"; R=sys.argv[2] if len(sys.argv)>2 else "results/calib_imnet_towns2"
P=json.load(open(PF))
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
print(f"predictions written {P['created']}")
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    rows=[(c,v[T]["pred_mAP"],m(f"{R}/src_{c}_{key}.json"),v[T]["pred_band_1sd"]) for c,v in P["cells"].items() if os.path.exists(f"{R}/src_{c}_{key}.json")]
    if not rows: continue
    print(f"\n=== {T}: {len(rows)} held-out cells (detector result files written {min(os.path.getmtime(f'{R}/src_{c}_{key}.json') for c,*_ in rows):.0f}+ epoch s) ===")
    for c,pr,ob,sd in rows: print(f"  {c:<24} predicted {pr:5.1f}  observed {ob:5.1f}  error {ob-pr:+5.1f}  {'within 1 sd' if abs(ob-pr)<=sd else ''}")
    pr=np.array([r[1] for r in rows]); ob=np.array([r[2] for r in rows]); sd=rows[0][3]
    print(f"  mean |error| {np.abs(ob-pr).mean():.2f} pp  (calibration residual sd {sd:.2f}); within 4 pp {100*(np.abs(ob-pr)<=4).mean():.0f}%")
    if len(rows)>=4:
        r,p=stats.pearsonr(pr,ob); rho,ps=stats.spearmanr(pr,ob); print(f"  predicted vs observed: r={r:+.3f} (p={p:.3f}) rho={rho:+.3f} (p={ps:.3f})")
    towns=sorted({c.split('_')[0] for c,*_ in rows})
    if len(towns)>=3:
        tp=[pr[[c.startswith(t+'_') for c,*_ in rows]].mean() for t in towns]; to=[ob[[c.startswith(t+'_') for c,*_ in rows]].mean() for t in towns]
        print("  town means (pred/obs): "+", ".join(f"{t} {a:.1f}/{b:.1f}" for t,a,b in zip(towns,tp,to)))

# ---- competing pre-registered models (FID vs training-box count vs both) ----
CF=PF.replace(".json","_competing.json")
if os.path.exists(CF):
    Cm=json.load(open(CF)); print(f"\ncompeting models file written {Cm['created']}")
    for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
        cells=[c for c in Cm["cells"] if os.path.exists(f"{R}/src_{c}_{key}.json")]
        if not cells: continue
        ob=np.array([m(f"{R}/src_{c}_{key}.json") for c in cells]); print(f"=== {T} ({len(cells)} held-out cells): mean |error| / r(pred, obs) ===")
        for name in ("fid","boxes","fid+boxes"):
            pr=np.array([Cm["cells"][c].get(f"{T}:{name}", Cm["cells"][c].get(f"{T}:{name} (prereg 08:12)")) for c in cells])
            rr=f"{stats.pearsonr(pr,ob)[0]:+.3f}" if len(cells)>=4 else "n/a"
            print(f"  {name:<10} |err| {np.abs(ob-pr).mean():5.2f}  r {rr}   "+"  ".join(f"{c.split('_')[0][-2:]}{c.split('_')[1][5:8]}:{p_:.1f}/{o:.1f}" for c,p_,o in zip(cells,pr,ob)))
