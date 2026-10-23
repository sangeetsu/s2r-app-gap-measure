#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ledger §15.77: reverse-direction directed metrics (ref -> dose) with exactly the objects / standardisation of dose_metrics.py; the forward
quantities are recomputed as a reproduction check. Output results/calib_dose/dose_reverse_metrics.json."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from dose_metrics import objs
from domain_matrix_probe import gather, CARLA_IMG_HW
from latent_geometry import knn_radius
ROOT="/mnt/features/carla_doses"; OUTF="results/calib_dose/dose_reverse_metrics.json"
def main():
    dev=torch.device("cuda"); lk=SpatialFeatureLookup.from_npz(f"{ROOT}/spatial_features_dinov3_80cells.npz"); out=json.load(open(OUTF)) if os.path.exists(OUTF) else {}
    cells=sorted(d.name for d in Path(ROOT).iterdir() if d.is_dir() and any(d.glob("frame_*.jpg"))); towns=sorted({c.split("_d")[0] for c in cells})
    for town in towns:
        todo=[c for c in cells if c.startswith(town+"_d") and not c.endswith("d00_ref") and c not in out]
        if not todo: continue
        ref=f"{town}_d00_ref"; dsr=CARLADetection(root=ROOT,cells=[ref]); n=len(dsr); _,Or=objs(lk,dsr,range(n))
        sel=np.sort(np.random.default_rng(42).choice(len(Or),min(3000,len(Or)),replace=False))
        Xp,yp=gather(lk,torch.utils.data.Subset(dsr,range(0,800 if n>=1000 else int(0.75*n))),lambda d,i:CARLA_IMG_HW,np.random.default_rng(7),4,ref+" probe"); mu,sd=Xp.mean(0),Xp.std(0)+1e-6
        S=torch.from_numpy((Or[sel]-mu)/sd).to(dev); rS=knn_radius(S)
        for c in todo:
            ds=CARLADetection(root=ROOT,cells=[c]); _,Od=objs(lk,ds,range(n)); T=torch.from_numpy((Od[sel]-mu)/sd).to(dev); rT=knn_radius(T)
            Dts=torch.cdist(T,S); dm=Dts.min(1).values; dr=Dts.min(0).values
            out[c]={"nnd_dose_to_ref":float(dm.median()/rS.median()),"cov_dose_by_ref":float((dm<=rS[Dts.argmin(1)]).float().mean()),
                    "nnd_ref_to_dose":float(dr.median()/rT.median()),"cov_ref_by_dose":float((dr<=rT[Dts.argmin(0)]).float().mean())}
            json.dump(out,open(OUTF,"w"),indent=1); print(f"[reverse] {c:<22} "+" ".join(f"{k}={v:.3f}" for k,v in out[c].items()),flush=True)
if __name__=="__main__": raise SystemExit(main())
