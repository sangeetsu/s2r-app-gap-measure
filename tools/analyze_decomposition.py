#!/usr/bin/env python3
"""Decomposition analysis (ledger §15.27): how much of the anchored drop the two anchors explain, and what each
pairwise measure adds beyond them (LOO by cell; partial correlations). Any number of cells; measures auto-detected."""
import json, glob, os, numpy as np
from scipy import stats
R=os.environ.get("CALIB_DIR","results/calib_anchored")   # detector-dependent results (evals, detector-box, BoS/AoL)
PMD="results/calib_anchored"                              # detector-INDEPENDENT probe matrices live here for both grids
print(f"[decomp] detector results from {R}")
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
cells=sorted({f.split("eval_")[1].split("__on__")[0] for f in glob.glob(f"{R}/eval_*.json")})
anchor={s:m(f"{R}/eval_{s}__on__{s}.json") for s in cells if glob.glob(f"{R}/eval_{s}__on__{s}.json")}; cells=[c for c in cells if c in anchor]
pairs=[(S,T) for S in cells for T in cells if S!=T and glob.glob(f"{R}/eval_{S}__on__{T}.json")]
y=np.array([anchor[S]-m(f"{R}/eval_{S}__on__{T}.json") for S,T in pairs]); src=np.array([anchor[S] for S,T in pairs]); tgt=np.array([anchor[T] for S,T in pairs])
def ld(pat): return [json.load(open(f)) for f in sorted(glob.glob(pat))]
meas={}
pm=ld(f"{PMD}/probe_matrix_anchored_all_s*.json") or ld(f"{PMD}/probe_matrix_anchored_s*.json")
if pm:
    mats=[p["matrix"] for p in pm]
    meas["dIV_fg (labeled)"]=lambda S,T: np.mean([M[S][T]["dIV_fg"] for M in mats if S in M and T in M[S]] or [np.nan])
ens=ld(f"{R}/detector_box_gap_all_s42.json") or ld(f"{R}/detector_box_gap_ens.json")
if ens:
    E=ens[0]; TAU="0.7"; ok_=lambda S,T: S in E and T in E[S] and TAU in E[S][T]
    meas["FM veto gap (label-free)"]=lambda S,T: (E[S][T][TAU]["probe_pbg"]-E[S][f"in@{TAU}"]["probe_pbg"]) if ok_(S,T) else np.nan
    meas["DoC (label-free)"]=lambda S,T: (E[S][f"in@{TAU}"]["det_conf"]-E[S][T][TAU]["det_conf"]) if ok_(S,T) else np.nan
    meas["FM-det disagreement (label-free)"]=lambda S,T: (E[S][f"in@{TAU}"]["agree"]-E[S][T][TAU]["agree"]) if ok_(S,T) else np.nan
dd=ld("results/domain_distance_dinov3_all.json") or ld("results/domain_distance_dinov3.json")
if dd:
    D=dd[0]["metrics"]["image"]; meas["FID (label-free)"]=lambda S,T: (D.get(f"{S}||{T}") or D.get(f"{T}||{S}") or {"fid":np.nan})["fid"]
ba=ld(f"{R}/bos_aol.json") or ld(f"{R}/bos_aol_8cells.json")
if ba:
    B=ba[0]
    for k,lab in [("bos_stability","BoS stability gap (label-free)"),("aol_mean_agreement","AoL agreement gap (label-free)")]:
        meas[lab]=(lambda k: (lambda S,T: (B[S][S][k]-B[S][T][k]) if S in B and S in B[S] and T in B[S] and k in B[S][T] else np.nan))(k)
X={k:np.array([f(S,T) for S,T in pairs],float) for k,f in meas.items()}
def loo(cols,yv=y):
    A=np.column_stack([np.ones(len(yv))]+cols); ok=~np.isnan(A).any(1); errs=[]
    for h in cells:
        inh=np.array([h in pr for pr in pairs]); te=inh&ok; tr=(~inh)&ok
        if te.sum()==0 or tr.sum()<len(cols)+2: continue
        b=np.linalg.lstsq(A[tr],yv[tr],rcond=None)[0]; errs+=list(np.abs(yv[te]-A[te]@b))
    e=np.array(errs); return e.mean(),100*(e<=4).mean(),1-np.mean(e**2)/yv[ok].var(),int(ok.sum())
def partial(x,zs,yv):
    Z=np.column_stack([np.ones(len(yv))]+zs); ok=~np.isnan(x)&~np.isnan(Z).any(1); x,Z,yy=x[ok],Z[ok],yv[ok]
    rx=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]; ry=yy-Z@np.linalg.lstsq(Z,yy,rcond=None)[0]; r,p=stats.pearsonr(rx,ry); return r,p,int(ok.sum())
print(f"{len(cells)} anchored cells, {len(pairs)} directional pairs; drop mean {y.mean():.1f} sd {y.std():.1f} pp; LOO by cell\n")
print(f"  {'model (predicting the anchored drop)':<56} {'|err|':>6} {'<=4pp':>6} {'LOO R2':>7} {'n':>4}")
for name,cols in [("source anchor",[src]),("target anchor",[tgt]),("both anchors",[src,tgt])]+[(f"both anchors + {k}",[src,tgt,X[k]]) for k in X]+[(f"source anchor + {k}  [deployable]",[src,X[k]]) for k in X]:
    e,p4,r2,n=loo(cols); print(f"  {name:<56} {e:6.2f} {p4:5.0f}% {r2:+7.3f} {n:4d}")
print(f"\n  {'measure':<38} {'raw r':>18} {'partial | both anchors':>28} {'partial | source anchor':>26}")
for k,x in X.items():
    ok=~np.isnan(x)
    if ok.sum()<6: continue
    r0=stats.pearsonr(x[ok],y[ok]); pb=partial(x,[src,tgt],y); ps=partial(x,[src],y)
    print(f"  {k:<38} {r0[0]:+.3f} (p={r0[1]:.0e})    {pb[0]:+.3f} (p={pb[1]:.0e}) n={pb[2]:<5d} {ps[0]:+.3f} (p={ps[1]:.0e})")
for Tgt,key in [("KITTI","kittival"),("Cityscapes","cityscapesval")]:
    fs=[s for s in cells if glob.glob(f"{R}/src_{s}_{key}.json")]
    if len(fs)<6: continue
    mt=np.array([m(f"{R}/src_{s}_{key}.json") for s in fs]); a=np.array([anchor[s] for s in fs]); d=a-mt
    print(f"\n=== {Tgt}: n={len(fs)} sources; mAP(S->{Tgt}) {mt.mean():.1f} ± {mt.std():.1f} pp; anchored drop {d.mean():.1f} ± {d.std():.1f} ===")
    print(f"  {'predictor':<38} {'r with mAP(S->T)':>18} {'r with drop':>13} {'partial r | anchor':>22}")
    print(f"  {'source anchor':<38} {stats.pearsonr(a,mt)[0]:+18.3f} {stats.pearsonr(a,d)[0]:+13.3f}")
    for k,f in meas.items():
        x=np.array([f(s,Tgt) for s in fs],float); ok=~np.isnan(x)
        if ok.sum()<6: continue
        pr=partial(x,[a],d); print(f"  {k:<38} {stats.pearsonr(x[ok],mt[ok])[0]:+18.3f} {stats.pearsonr(x[ok],d[ok])[0]:+13.3f} {pr[0]:+14.3f} (p={pr[1]:.2f})")
