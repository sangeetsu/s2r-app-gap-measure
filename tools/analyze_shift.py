#!/usr/bin/env python3
"""Score the pre-registered SHIFT replication (ledger §15.48): S1 appearance at near-matched counts, S2 FID increment beyond count,
S3 causal count effect (learning curve)."""
import json, os, numpy as np
from pathlib import Path
from scipy import stats
R="results/calib_shift"; ROOT=Path("/mnt/features/shift_cells"); META=json.load(open(ROOT/"cells_meta.json")); FID=json.load(open(f"{R}/fid_to_real.json"))["cells"]
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
def nbox(cell,n): return sum(1 for k in range(n) for l in open(ROOT/cell/f"frame_{k:06d}.txt") if l.strip())
cells=sorted(META); tod=np.array([META[c]["timeofday"] for c in cells]); wx=np.array([META[c]["weather"] for c in cells]); lb=np.log([META[c]["train_boxes"] for c in cells])
def dummies(v): u=sorted(set(v)); return np.column_stack([(v==k).astype(float) for k in u[1:]])
def rss(y,X): b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b; return float(e@e)
def ftest(y,Xr,Xf):
    r0,r1=rss(y,Xr),rss(y,Xf); df1=Xf.shape[1]-Xr.shape[1]; df2=len(y)-Xf.shape[1]; F=((r0-r1)/df1)/(r1/df2); return F, stats.f.sf(F,df1,df2), df1, df2
one=np.ones((len(cells),1))
for key,T in [("kittival","KITTI"),("cityscapesval","Cityscapes")]:
    y=np.array([m(f"{R}/src_{c}_{key}.json") for c in cells]); x=np.array([FID[c][T]["fid"] for c in cells]); tot=float(((y-y.mean())**2).sum())
    Xb=np.column_stack([one,lb]); Xt=np.column_stack([Xb,dummies(tod)]); Xtw=np.column_stack([Xt,dummies(wx)])
    print(f"=== {T}: 15 condition cells, real mAP {y.mean():.1f} ± {y.std(ddof=1):.1f} (range {y.min():.1f}-{y.max():.1f}); train boxes {int(np.exp(lb).min())}-{int(np.exp(lb).max())}")
    print(f"  S1 shares of variance (sequential): log boxes {100*(tot-rss(y,Xb))/tot:.0f}%, + time-of-day {100*(rss(y,Xb)-rss(y,Xt))/tot:.0f}%, + weather {100*(rss(y,Xt)-rss(y,Xtw))/tot:.0f}%, residual {100*rss(y,Xtw)/tot:.0f}%")
    for name,Xr,Xf in [("time-of-day | boxes",Xb,Xt),("weather | boxes,time",Xt,Xtw),("time+weather | boxes",Xb,Xtw)]:
        F,p,d1,d2=ftest(y,Xr,Xf); print(f"     F({d1},{d2}) {name:<22} = {F:.2f}, p={p:.3f}")
    r0=stats.pearsonr(x,y); F,p,_,_=ftest(y,Xb,np.column_stack([Xb,x])); rx=x-Xb@np.linalg.lstsq(Xb,x,rcond=None)[0]; ry=y-Xb@np.linalg.lstsq(Xb,y,rcond=None)[0]; pr=stats.pearsonr(rx,ry)
    print(f"  S2 FID: raw r={r0[0]:+.3f} (p={r0[1]:.3f}); increment beyond log boxes: partial r={pr[0]:+.3f}, F p={p:.3f} -> {'REPLICATES (no increment)' if p>=.05 else 'DOES NOT replicate (FID adds)'}")
    X,Y=[],[]
    for c in ["SHIFT_daytime_clear","SHIFT_night_rainy","SHIFT_dawn-dusk_foggy"]:
        xs=[np.log(nbox(c,n)) for n in (100,200,400)]; ys=[m(f"{R}/src_{c}_n{n}_{key}.json") if n<400 else m(f"{R}/src_{c}_{key}.json") for n in (100,200,400)]
        print(f"     {c:<24} "+"  ".join(f"n{n}: {v:5.1f} ({int(np.exp(b))} boxes)" for n,v,b in zip((100,200,400),ys,xs)))
        X+=list(np.array(xs)-np.mean(xs)); Y+=list(np.array(ys)-np.mean(ys))
    X,Y=np.array(X),np.array(Y); s=float(X@Y/(X@X)); se=np.sqrt(((Y-s*X)**2).sum()/(len(X)-3-1)/(X@X)); t=s/se; p=2*stats.t.sf(abs(t),len(X)-4)
    print(f"  S3 within-cell slope {s:+.2f} pp per log(boxes) (SE {se:.2f}, p={p:.4f}) -> {'REPLICATES' if s>0 and p<.05 else 'DOES NOT replicate'}")
