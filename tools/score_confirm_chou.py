#!/usr/bin/env python3
"""Pre-registered confirmation of the exploratory geometry model (ledger §15.43). Step 1 writes predictions (frozen coefficients, no outcomes
read); step 2 reads outcomes and scores rule (a) r(pred,obs)>0, p<.05 on both targets and (b) MAE below boxes-only on both."""
import json, glob, os, time, numpy as np
from scipy import stats
F=json.load(open("results/manifold_markers/frozen_model.json")); K=json.load(open("results/manifold_markers/markers_confirm.json")); R="results/calib_imnet_confirm"; ROOT="/mnt/features/carla_confirm"
def nbox(c): return sum(1 for f in sorted(glob.glob(f"{ROOT}/{c}/frame_*.txt"))[:800] for l in open(f) if l.split() and l.split()[0] in ("Car","Pedestrian","Cyclist"))
pred={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"cells":{}}
for n,r in K.items():
    c=n.split("_",1)[1]; lb=float(np.log(nbox(c))); row={"log_boxes":lb,"has_markers":"D_eff" in r}
    for T,mdl in F["models"].items():
        b=mdl["boxes_only"]; row[f"{T}:boxes_only"]=b["intercept"]+b["log_boxes"]*lb
        if "D_eff" in r: f=mdl["full"]; row[f"{T}:full"]=f["intercept"]+f["log_boxes"]*lb+f["alpha"]*r["alpha"]+f["D_eff"]*r["D_eff"]
    pred["cells"][c]=row
json.dump(pred,open("results/manifold_markers/confirm_predictions.json","w"),indent=1); print(f"predictions written {pred['created']} for {len(pred['cells'])} cells (outcomes not yet read)")
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
ok=True
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    cells=[c for c,v in pred["cells"].items() if v["has_markers"] and os.path.exists(f"{R}/src_{c}_{key}.json")]
    ob=np.array([m(f"{R}/src_{c}_{key}.json") for c in cells]); pf=np.array([pred["cells"][c][f"{T}:full"] for c in cells]); pb=np.array([pred["cells"][c][f"{T}:boxes_only"] for c in cells])
    r,p=stats.pearsonr(pf,ob); rb,pbv=stats.pearsonr(pb,ob); maef,maeb=np.abs(ob-pf).mean(),np.abs(ob-pb).mean()
    pa=(r>0 and p<.05); pb_=maef<maeb; ok=ok and pa and pb_
    print(f"\n=== {T}: {len(cells)} fresh cells with markers ===")
    for c,o,a,b in zip(cells,ob,pf,pb): print(f"  {c:<22} observed {o:5.1f}  full-model {a:5.1f}  boxes-only {b:5.1f}")
    print(f"  full model: r={r:+.3f} (p={p:.4f}) MAE {maef:.2f} | boxes-only: r={rb:+.3f} (p={pbv:.4f}) MAE {maeb:.2f} | rule (a) {'PASS' if pa else 'FAIL'}  rule (b) {'PASS' if pb_ else 'FAIL'}")
print(f"\nPRE-REGISTERED CONFIRMATION: {'PASSED' if ok else 'FAILED'}")
