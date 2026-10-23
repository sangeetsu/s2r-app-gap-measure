#!/usr/bin/env python3
"""Detector-box label-free signals (tools/detector_box_gap.py) vs anchored drops; reference = labeled dIV_fg."""
import json, glob, sys, numpy as np
from scipy import stats
R="results/calib_anchored"; FILES=(sys.argv[1] if len(sys.argv)>1 else f"{R}/detector_box_gap_s42.json").split(","); TAU=sys.argv[2] if len(sys.argv)>2 else "0.5"
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
cells=sorted({f.split("eval_")[1].split("__on__")[0] for f in glob.glob(f"{R}/eval_*.json")})
anchor={s:m(f"{R}/eval_{s}__on__{s}.json") for s in cells if glob.glob(f"{R}/eval_{s}__on__{s}.json")}; cells=[c for c in cells if c in anchor]
DS=[json.load(open(f)) for f in FILES]; D=DS[0]; seeds=[json.load(open(f))["matrix"] for f in sorted(glob.glob(f"{R}/probe_matrix_anchored_s*.json"))]
def sig(S,T):
    vals=[sig1(d,S,T) for d in DS]; return {k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
def sig1(D,S,T):
    i=D[S][f"in@{TAU}"]; t=D[S][T][TAU]
    return {"gap_atc (probe ATC, label-free)":i["acc_in"]-t["probe_atc"],"gap_conf (probe conf, label-free)":i["acc_in"]-t["probe_conf"],
            "gap_agree (FM-det agreement, lf)":i["agree"]-t["agree"],"gap_bg (probe rejects det box, lf)":t["probe_bg"]-i["probe_bg"],"gap_pbg (soft FM veto, mean P(bg), lf)":t.get("probe_pbg",np.nan)-i.get("probe_pbg",np.nan),
            "DoC (det confidence, lf)":i["det_conf"]-t["det_conf"],"log count ratio (dets/frame, lf)":np.log(i["dets_pf"]/max(1e-6,t["dets_pf"])),
            "true probe gap (LABELED diag)":i["acc_in"]-t["true_acc"],"det precision drop (LABELED diag)":i["det_prec"]-t["det_prec"]}
def rep(name,x,y,hold=None):
    x,y=np.array(x,float),np.array(y,float); ok=~np.isnan(x)&~np.isnan(y)
    if ok.sum()<4: return
    r,p=stats.pearsonr(x[ok],y[ok]); rho,ps=stats.spearmanr(x[ok],y[ok]); s=f"  {name:<36} r={r:+.3f} (p={p:.1e}) rho={rho:+.3f}"
    if hold is not None:
        errs=[]
        for h in set(hold):
            tr=np.array([hh!=h for hh in hold])&ok; te=np.array([hh==h for hh in hold])&ok
            if tr.sum()<3 or te.sum()==0: continue
            sl,ic,*_=stats.linregress(x[tr],y[tr]); errs+=list(np.abs(y[te]-(ic+sl*x[te])))
        errs=np.array(errs); s+=f"  LOO|err| {errs.mean():.2f} <=4pp {100*(errs<=4).mean():.0f}%"
    print(s)
srcs=[c for c in cells if c in D]
pairs=[(S,T) for S in srcs for T in cells if S!=T and T in D[S] and TAU in D[S][T] and glob.glob(f"{R}/eval_{S}__on__{T}.json")]
drop=[anchor[S]-m(f"{R}/eval_{S}__on__{T}.json") for S,T in pairs]; hold=[S for S,T in pairs]
print(f"=== {len(pairs)} CARLA pairs ({len(srcs)} sources), anchored drop, tau_det={TAU}; {len(FILES)} probe run(s) averaged; LOO by source ===")
if seeds: rep("dIV_fg (LABELED reference)",[np.mean([s[S][T]["dIV_fg"] for s in seeds if S in s and T in s[S]]) for S,T in pairs],drop,hold)
for k in sig(*pairs[0]): rep(k,[sig(S,T)[k] for S,T in pairs],drop,hold)
for Tgt,key in [("KITTI","kittival"),("Cityscapes","cityscapesval")]:
    fs={s:m(f"{R}/src_{s}_{key}.json") for s in srcs if glob.glob(f"{R}/src_{s}_{key}.json") and Tgt in D[s] and TAU in D[s][Tgt]}
    if len(fs)<6: continue
    y=[anchor[s]-fs[s] for s in fs]; print(f"=== {Tgt}: n={len(fs)} sources, anchored drop ===")
    if seeds: rep("dIV_fg (LABELED reference)",[np.mean([sd[s][Tgt]["dIV_fg"] for sd in seeds if s in sd]) for s in fs],y)
    for k in sig(next(iter(fs)),Tgt): rep(k,[sig(s,Tgt)[k] for s in fs],y)
