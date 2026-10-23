#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Label-free FID (DINOv3 image level) between CARLA cells and the real targets, replicating tools/domain_distance_matrix.py
exactly (same feature pooling, first-1000-frames selection, pair order Cityscapes<KITTI<Town*, _cap seeds 1/2, fid()).
With --predict, applies the pre-registered calibration lines (results/calib_imnet/prereg_fid_line.json) BEFORE any detector
is trained on the new cells, and writes a timestamped prediction file.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.carla import CARLADetection
from s2rgap.data.kitti import KITTIDetection
from domain_distance_matrix import fid, cos_mean, _cap, frame_and_box_feats
def log(m): print(f"[fid2real] {m}", flush=True)
def img_feats(lookup, rels, max_frames=1000):
    f,_=frame_and_box_feats(lookup, rels, {}, {}, max_frames); return f
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--features",required=True); p.add_argument("--cells",nargs="*",default=None)
    p.add_argument("--out",required=True); p.add_argument("--predict",action="store_true")
    p.add_argument("--kitti-features",default=str(Path.home()/"datasets/kitti/spatial_features_dinov3.npz")); p.add_argument("--cs-features",default=str(Path.home()/"datasets/cityscapes/spatial_features_dinov3_val.npz"))
    p.add_argument("--kitti-root",default=str(Path.home()/"datasets/kitti/extracted/training")); p.add_argument("--kitti-splits",default=str(Path.home()/"datasets/kitti/splits.json"))
    a=p.parse_args()
    lk=SpatialFeatureLookup.from_npz(a.kitti_features); kv=KITTIDetection(kitti_root=a.kitti_root,split_json=a.kitti_splits,split="val")
    krels=[kv[i][1]["frame_rel"] for i in range(min(1000,len(kv)))]; K=img_feats(lk,krels)
    lcs=SpatialFeatureLookup.from_npz(a.cs_features); CS=np.array([np.array(lcs.z_mmap[i],dtype=np.float32).reshape(-1,lcs.z_mmap.shape[-1]).mean(0) for i in range(min(1000,len(lcs)))],np.float32)
    log(f"KITTI {len(K)} frames, Cityscapes {len(CS)} frames")
    lc=SpatialFeatureLookup.from_npz(a.features)
    cells=a.cells or sorted(d.name for d in Path(a.root).iterdir() if d.is_dir() and any(d.glob("frame_*.jpg")))   # Town* and SHIFT_* cells
    out={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"root":a.root,"cells":{}}
    pre=json.load(open(REPO/"results/calib_imnet/prereg_fid_line.json")) if a.predict else None
    for c in cells:
        ds=CARLADetection(root=a.root,cells=[c]); rels=[r[2] for r in ds.records]; C=img_feats(lc,rels)
        if len(C)<100: log(f"{c}: only {len(C)} frames with features - skipped (singular covariance)"); continue
        row={"n_frames":int(len(C))}
        for T,X in (("Cityscapes",CS),("KITTI",K)):   # matrix order: (T, cell) since "Cityscapes"/"KITTI" < "Town*"
            A,B=_cap(X,1),_cap(C,2); row[T]={"fid":float(fid(A,B)),"cos_mean":float(cos_mean(A,B))}
            if pre: L=pre["lines"][T]; row[T]["pred_mAP"]=L["intercept"]+L["slope"]*row[T]["fid"]; row[T]["pred_band_1sd"]=L["resid_sd"]
        out["cells"][c]=row; log(f"{c:<24} FID KITTI {row['KITTI']['fid']:6.2f}  Cityscapes {row['Cityscapes']['fid']:6.2f}"+(f"  -> predicted mAP KITTI {row['KITTI']['pred_mAP']:.1f}, CS {row['Cityscapes']['pred_mAP']:.1f}" if pre else ""))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(out,indent=1)); log(f"saved -> {a.out}")
if __name__=="__main__": raise SystemExit(main())
