#!/usr/bin/env python3
"""Score candidate B (ledger §15.45): frozen mAP ~ log(boxes) + nnd_T_to_S. Usage: score_confirm_B.py NND_JSON ROOT RESULTS_DIR PRED_OUT [alpha]"""
import json, glob, os, sys, time, numpy as np
from scipy import stats
NJ,ROOT,R,PO=sys.argv[1:5]; ALPHA=float(sys.argv[5]) if len(sys.argv)>5 else .025
F=json.load(open("results/manifold_markers/frozen_model_B.json")); N=json.load(open(NJ))
def nbox(c): return sum(1 for f in sorted(glob.glob(f"{ROOT}/{c}/frame_*.txt"))[:800] for l in open(f) if l.split() and l.split()[0] in ("Car","Pedestrian","Cyclist"))
pred={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"cells":{}}
for c,v in N.items():
    lb=float(np.log(nbox(c))); row={"log_boxes":lb}
    for T,mdl in F["models"].items():
        f,b=mdl["full"],mdl["boxes_only"]; row[f"{T}:full"]=f["intercept"]+f["log_boxes"]*lb+f["nnd"]*v[T]; row[f"{T}:boxes_only"]=b["intercept"]+b["log_boxes"]*lb
    pred["cells"][c]=row
json.dump(pred,open(PO,"w"),indent=1); print(f"predictions written {pred['created']} ({len(pred['cells'])} cells; outcomes not yet read)")
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
ok=True
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    cells=[c for c in pred["cells"] if os.path.exists(f"{R}/src_{c}_{key}.json")]
    if len(cells)<4: print(f"{T}: only {len(cells)} cells with outcomes"); ok=False; continue
    ob=np.array([m(f"{R}/src_{c}_{key}.json") for c in cells]); pf=np.array([pred["cells"][c][f"{T}:full"] for c in cells]); pb=np.array([pred["cells"][c][f"{T}:boxes_only"] for c in cells])
    r,p=stats.pearsonr(pf,ob); rb,pbv=stats.pearsonr(pb,ob); mf,mb=np.abs(ob-pf).mean(),np.abs(ob-pb).mean(); pa=(r>0 and p<ALPHA); pbb=mf<mb; ok=ok and pa and pbb
    print(f"\n=== {T}: {len(cells)} cells ===")
    for c,o,a,b in zip(cells,ob,pf,pb): print(f"  {c:<22} observed {o:5.1f}  B-model {a:5.1f}  boxes-only {b:5.1f}")
    print(f"  B-model r={r:+.3f} (p={p:.4f}) MAE {mf:.2f} | boxes-only r={rb:+.3f} (p={pbv:.4f}) MAE {mb:.2f} | rule (a, p<{ALPHA}) {'PASS' if pa else 'FAIL'} rule (b) {'PASS' if pbb else 'FAIL'}")
print(f"\nCANDIDATE B: {'PASSED' if ok else 'FAILED'}")
