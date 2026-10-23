#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ledger §15.78 (M): per-object mechanism. For each GT object in a held-out frame (records 800:end) that the town's seed-42 reference detector detects in
the reference render (same class, IoU >= .5, score >= .5): lost in the dose render? vs per-object DINOv3 displacement (dose_metrics.py standardisation),
|Δ RMS contrast| and |Δ mean luminance| inside the box, box height. Output results/calib_dose/per_object.npz; mode analyze -> mechanism.txt."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
ROOT="/mnt/features/carla_doses"; OUT="results/calib_dose/per_object.npz"
T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]; D=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
def iou(a,b):
    x1=np.maximum(a[:,None,0],b[None,:,0]); y1=np.maximum(a[:,None,1],b[None,:,1]); x2=np.minimum(a[:,None,2],b[None,:,2]); y2=np.minimum(a[:,None,3],b[None,:,3])
    inter=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None); ar=lambda z:(z[:,2]-z[:,0])*(z[:,3]-z[:,1]); return inter/(ar(a)[:,None]+ar(b)[None,:]-inter+1e-9)
def detected(gt,gl,pb,pl,ps):
    k=ps>=.5; pb,pl=pb[k],pl[k]
    if len(pb)==0 or len(gt)==0: return np.zeros(len(gt),bool)
    I=iou(gt,pb); return np.array([(I[j][pl==gl[j]]>=.5).any() if (pl==gl[j]).any() else False for j in range(len(gt))])
def collect():
    from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
    from s2rgap.data.carla import CARLADetection
    from s2rgap.detector import load_detector_from_ckpt
    from linear_probe_object_class import bbox_to_feat_region
    from domain_matrix_probe import gather, CARLA_IMG_HW
    dev=torch.device("cuda"); lk=SpatialFeatureLookup.from_npz(f"{ROOT}/spatial_features_dinov3_80cells.npz"); gh,gw=lk.target_hw; rows=[]
    for t in T:
        ref=f"{t}_d00_ref"; dsr=CARLADetection(root=ROOT,cells=[ref]); n=len(dsr); det=load_detector_from_ckpt(f"checkpoints/anchind_{ref}/last.pt",dev)
        Xp,_=gather(lk,torch.utils.data.Subset(dsr,range(0,800)),lambda d,i:CARLA_IMG_HW,np.random.default_rng(7),4,ref+" probe"); sd=Xp.std(0)+1e-6
        dss={d:CARLADetection(root=ROOT,cells=[f"{t}_{d}"]) for d in D}
        for i in range(800,n):
            im_r,tg=dsr[i]; gt=tg["boxes"].numpy(); gl=tg["labels"].numpy()
            if len(gt)==0: continue
            with torch.no_grad(): pr=det([im_r.to(dev)])[0]
            hit=detected(gt,gl,pr["boxes"].cpu().numpy(),pr["labels"].cpu().numpy(),pr["scores"].cpu().numpy())
            if not hit.any(): continue
            H,W=int(im_r.shape[-2]),int(im_r.shape[-1]); zr=np.array(lk.z_mmap[lk.path_to_idx[tg["frame_rel"]]],dtype=np.float32); gr=im_r.mean(0).numpy()
            regs=[bbox_to_feat_region(tuple(b),(H,W),(gh,gw)) for b in gt]
            for d in D:
                im_d,td=dss[d][i]; assert td["frame_rel"].split("/")[-1]==tg["frame_rel"].split("/")[-1]
                with torch.no_grad(): pd=det([im_d.to(dev)])[0]
                hd=detected(gt,gl,pd["boxes"].cpu().numpy(),pd["labels"].cpu().numpy(),pd["scores"].cpu().numpy())
                zd=np.array(lk.z_mmap[lk.path_to_idx[td["frame_rel"]]],dtype=np.float32); gd=im_d.mean(0).numpy()
                for j in np.where(hit)[0]:
                    i1,j1,i2,j2=regs[j]; fr=zr[i1:i2,j1:j2].reshape(-1,zr.shape[-1]); fd=zd[i1:i2,j1:j2].reshape(-1,zd.shape[-1])
                    if len(fr)==0: continue
                    x1,y1,x2,y2=[int(round(v)) for v in gt[j]]; x1,y1=max(x1,0),max(y1,0); pa=gr[y1:max(y2,y1+1),x1:max(x2,x1+1)]; pb_=gd[y1:max(y2,y1+1),x1:max(x2,x1+1)]
                    rows.append((T.index(t),D.index(d),int(gl[j]),float(gt[j][3]-gt[j][1]),float(np.linalg.norm((fd.mean(0)-fr.mean(0))/sd)),abs(float(pa.std()-pb_.std())),abs(float(pa.mean()-pb_.mean())),int(not hd[j])))
        print(f"[mech] {t}: {len(rows)} object-dose rows so far",flush=True)
    a=np.array(rows,float); np.savez(OUT,town=a[:,0],dose=a[:,1],label=a[:,2],height=a[:,3],disp=a[:,4],dcontrast=a[:,5],dlum=a[:,6],lost=a[:,7]); print("[mech] saved",OUT,a.shape)
def analyze():
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    z=np.load(OUT); y=z["lost"].astype(int); g=z["town"].astype(int); rng=np.random.default_rng(0)
    def boot(fn):
        v=[]
        for _ in range(1000):
            ii=np.concatenate([np.where(g==k)[0] for k in rng.choice(8,8,replace=True)]); 
            try: v.append(fn(ii))
            except Exception: pass
        return np.percentile(v,[2.5,97.5])
    print(f"{len(y)} object-dose rows (objects detected in the reference render); lost in dose render: {y.mean()*100:.1f}%")
    for k in ("disp","dcontrast","dlum","height"):
        x=z[k] if k!="height" else -z["height"]; auc=roc_auc_score(y,x); ci=boot(lambda ii: roc_auc_score(y[ii],x[ii]))
        print(f"  AUC {k:<10} {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"+("  (smaller boxes = higher score)" if k=="height" else "")+(f"  -> M1 {'PASS' if ci[0]>.5 else 'FAIL'}" if k=="disp" else ""))
    fam={"sun":[0,1,2,3],"fog":[4,5,6],"wet+rain":[7,8]}
    for f,ds in fam.items():
        i=np.isin(z["dose"].astype(int),ds); print(f"    {f:<9} AUC disp {roc_auc_score(y[i],z['disp'][i]):.3f}  dcontrast {roc_auc_score(y[i],z['dcontrast'][i]):.3f}  lost {100*y[i].mean():.1f}%  n={i.sum()}")
    X=np.column_stack([np.log(z["height"]),z["dcontrast"],z["dlum"],z["disp"]]); mu,sd_=X.mean(0),X.std(0); Xs=(X-mu)/sd_
    lr=LogisticRegression(max_iter=1000).fit(Xs,y); coef=lr.coef_[0]; ci=boot(lambda ii: LogisticRegression(max_iter=1000).fit(Xs[ii],y[ii]).coef_[0][3])
    lr0=LogisticRegression(max_iter=1000).fit(Xs[:,:3],y); a0=roc_auc_score(y,lr0.predict_proba(Xs[:,:3])[:,1]); a1=roc_auc_score(y,lr.predict_proba(Xs)[:,1])
    print(f"  logistic (standardised): log height {coef[0]:+.2f}, |Δcontrast| {coef[1]:+.2f}, |Δlum| {coef[2]:+.2f}, displacement {coef[3]:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] -> M2 {'PASS' if ci[0]>0 else 'FAIL'}")
    print(f"  in-sample AUC: baselines only {a0:.3f} -> + displacement {a1:.3f}")
    for b in (0,1,2,3,4):
        q=np.quantile(z["disp"],[b/5,(b+1)/5]); i=(z["disp"]>=q[0])&(z["disp"]<=q[1]); print(f"    displacement quintile {b+1}: [{q[0]:.1f}, {q[1]:.1f}] lost {100*y[i].mean():.1f}%")
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=["collect","analyze"]); a=p.parse_args(); collect() if a.mode=="collect" else analyze()
