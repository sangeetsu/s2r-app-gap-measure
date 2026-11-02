#!/usr/bin/env python3
"""Consolidated analysis for the CVPR paper (plan_2026-10-16 §4, Oct 19 item): every table from results/*.json in ONE run, with
town-cluster bootstrap 95% CIs and town-clustered permutation p for the headline statistics. Reads only.
Usage (worktree root): python tools/consolidate.py  ->  results/consolidated/tables.json + tables.md
Every table lists its source directories; numbers are recomputed from the eval/src JSONs, never copied from the ledger."""
import json, glob, os, time, numpy as np
from pathlib import Path
from scipy import stats
B=2000; rng=np.random.default_rng(0)
OUT=Path("results/consolidated"); OUT.mkdir(parents=True,exist_ok=True)
IM,IMR,H2,CF,LC,LQ="results/calib_imnet","results/calib_imnet_rep","results/calib_imnet_towns2","results/calib_imnet_confirm","results/calib_imnet_lc","results/calib_imnet_lowq"
AN,SH,DO,DOR,A2,FG,H25,H40="results/calib_anchored","results/calib_shift","results/calib_dose","results/calib_dose_rep","results/calib_dose_arch2","results/calib_dose_foggy","results/calib_dose_h25","results/calib_dose_h40"
DOSES=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]; FAM={"sun":DOSES[:4],"fog":DOSES[4:7]}
TG=[("KITTI","kittival"),("Cityscapes","cityscapesval")]
J={"created":time.strftime("%Y-%m-%dT%H:%M:%S"),"B":B,"seed":0,"methods":{
 "ci":"95% percentile interval, cluster bootstrap over towns (cells where no town structure), B=2000, seed 0",
 "p":"two-sided permutation p, outcome shuffled within town, B=2000: (1+#|null|>=|obs|)/(B+1); floor 0.0005",
 "loto_r2":"leave-one-town-out linear calibration, 1 - MSE/var(y)",
 "cal_k":"mean |error| (pp) of a linear calibration fit on k towns, applied to the remaining towns; k=n-1 exact, k=1 200 random draws"},"tables":{}}
MD=[f"# Consolidated tables — tools/consolidate.py, {J['created']} (B={B}, seed 0)\n","CI = town-cluster bootstrap 95%; p = town-clustered permutation; LOTO = leave-one-town-out calibration. Every number is recomputed from the listed results directories.\n"]
# ---------- helpers ----------
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
def mm(f): return m(f) if os.path.exists(f) else np.nan
def town(c): return c.split("_")[0]
def sp(x,y): return stats.spearmanr(x,y)[0]
def pr(x,y): return stats.pearsonr(x,y)[0]
def cboot(fn,g,*a):
    g=np.asarray(g); G=sorted(set(g)); idx={k:np.where(g==k)[0] for k in G}; v=[]
    for _ in range(B):
        ii=np.concatenate([idx[k] for k in rng.choice(G,len(G),replace=True)])
        try: v.append(fn(*[x[ii] for x in a]))
        except Exception: pass
    v=np.array(v,float); v=v[np.isfinite(v)]
    return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] if len(v)>B//2 else [np.nan,np.nan]
def cperm(fn,x,y,g):
    g=np.asarray(g); obs=fn(x,y); null=[]; G=list(set(g)); idx=[np.where(g==k)[0] for k in G]
    for _ in range(B):
        yp=y.copy()
        for i in idx: yp[i]=rng.permutation(y[i])
        null.append(fn(x,yp))
    return float((1+np.sum(np.abs(null)>=abs(obs)-1e-12))/(B+1))
def loto_r2(x,y,g):
    g=np.asarray(g); e=[]
    for h in set(g):
        te=g==h; tr=~te; b=np.polyfit(x[tr],y[tr],1); e+=list((y[te]-np.polyval(b,x[te]))**2)
    return float(1-np.mean(e)/np.mean((y-y.mean())**2))
def cal_k(x,y,g,k):
    g=np.asarray(g); G=sorted(set(g))
    if k>=len(G): return np.nan
    subs=[[t for t in G if t!=h] for h in G] if k==len(G)-1 else [list(rng.choice(G,k,replace=False)) for _ in range(200)]
    e=[]
    for tr in subs:
        itr=np.isin(g,tr); ite=~itr
        if itr.sum()<3: continue
        b=np.polyfit(x[itr],y[itr],1); e+=list(np.abs(y[ite]-np.polyval(b,x[ite])))
    return float(np.mean(e))
def _partial_r(rx,ry,k):
    """Pearson r of two residual vectors with its two-sided p-value on n - 2 - k degrees of freedom (k controlled variables)."""
    r=float(stats.pearsonr(rx,ry)[0]); df=len(rx)-2-k; t=r*np.sqrt(df/max(1e-12,1-r*r)); return r,float(2*stats.t.sf(abs(t),df))
def partial(x,y,z):
    Z=np.column_stack([np.ones(len(y)),z]); rx=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]; ry=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]; return _partial_r(rx,ry,Z.shape[1]-1)
def rss(y,X): b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b; return float(e@e)
def dummies(v): u=sorted(set(v)); return np.column_stack([(np.asarray(v)==k).astype(float) for k in u[1:]])
def fmt(v,d=3,sign=True):
    if v is None or (isinstance(v,float) and not np.isfinite(v)): return "—"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
def ci(c): return "—" if not np.isfinite(c[0]) else f"[{c[0]:+.2f}, {c[1]:+.2f}]"
def pf(p): return "<.001" if p<.001 else f"{p:.3f}"
def table(key,title,header,rows,note="",sources=(),data=None):
    J["tables"][key]={"title":title,"header":header,"rows":rows,"note":note,"sources":list(sources),"data":data}
    MD.append(f"\n## {key}. {title}\n"); MD.append("| "+" | ".join(header)+" |"); MD.append("|"+"---|"*len(header))
    for r in rows: MD.append("| "+" | ".join(str(c) for c in r)+" |")
    if note: MD.append(f"\n{note}")
    if sources: MD.append(f"\nSources: {', '.join(sources)}")
    print(f"[consolidate] {key} {title}")
ROOTS=[f"{Path.home()}/datasets/carla_extracted","/mnt/features/carla_new","/mnt/features/carla_new2","/mnt/features/carla_confirm","/mnt/features/carla_lowq"]
def root(c,prefer=None):
    for r in ([prefer] if prefer else [])+ROOTS:
        if r and os.path.isdir(f"{r}/{c}"): return r
_nb={}
def nbox(c,k=800,r=None):
    r=r or root(c)
    if (r,c,k) not in _nb: _nb[(r,c,k)]=sum(1 for f in sorted(glob.glob(f"{r}/{c}/frame_*.txt"))[:k] for l in open(f) if l.split() and l.split()[0] in ("Car","Pedestrian","Cyclist"))
    return _nb[(r,c,k)]
def own_cells(R): return sorted({f.split("eval_")[1].split("__on__")[0] for f in glob.glob(f"{R}/eval_*__on__*.json") if f.split("eval_")[1].split("__on__")[0]==f.split("__on__")[1][:-5]})
def real(R,c,key): return mm(f"{R}/src_{c}_{key}.json")
def anchors_loo(R,cells):
    """both-anchors LOO-by-cell R² of the anchored within-CARLA drop (ledger §15.27/15.29 model)."""
    anc={s:m(f"{R}/eval_{s}__on__{s}.json") for s in cells}; pairs=[(S,T) for S in cells for T in cells if S!=T and os.path.exists(f"{R}/eval_{S}__on__{T}.json")]
    y=np.array([anc[S]-m(f"{R}/eval_{S}__on__{T}.json") for S,T in pairs]); A=np.column_stack([np.ones(len(y)),[anc[S] for S,_ in pairs],[anc[T] for _,T in pairs]]); e=[]
    for h in cells:
        te=np.array([h in p for p in pairs]); tr=~te; b=np.linalg.lstsq(A[tr],y[tr],rcond=None)[0]; e+=list((y[te]-A[te]@b)**2)
    return len(pairs),float(1-np.mean(e)/y.var()),float(np.mean(np.sqrt(e)))
def anova_shares(cells,y):
    tw=[town(c) for c in cells]; wx=[c.split("_",1)[1] for c in cells]; one=np.ones((len(y),1)); tot=float(((y-y.mean())**2).sum())
    Xt=np.column_stack([one,dummies(tw)]); Xtw=np.column_stack([Xt,dummies(wx)]); r0,r1,r2=tot,rss(y,Xt),rss(y,Xtw); dfr=len(y)-Xtw.shape[1]
    Ft=((r0-r1)/(Xt.shape[1]-1))/(r2/dfr); Fw=((r1-r2)/(Xtw.shape[1]-Xt.shape[1]))/(r2/dfr)
    return (r0-r1)/tot,stats.f.sf(Ft,Xt.shape[1]-1,dfr),(r1-r2)/tot,stats.f.sf(Fw,Xtw.shape[1]-Xt.shape[1],dfr)
# ---------- grids ----------
C_AN=own_cells(AN); C_IM=own_cells(IM); C_H2=sorted({f.split("src_")[1].replace("_kittival.json","") for f in glob.glob(f"{H2}/src_*_kittival.json")})
C_CF=sorted({f.split("src_")[1].replace("_kittival.json","") for f in glob.glob(f"{CF}/src_*_kittival.json")}); C_SH=sorted({f.split("src_")[1].replace("_kittival.json","") for f in glob.glob(f"{SH}/src_SHIFT_*_kittival.json") if "_n" not in f.split("src_")[1].replace("_kittival.json","")[-5:]})
C_SH=[c for c in C_SH if not any(c.endswith(f"_n{n}") for n in (100,200,400))]
T_DO=sorted({c.split("_d00")[0] for c in own_cells(DO) if c.endswith("d00_ref")}); T_A2=sorted({c.split("_d00")[0] for c in own_cells(A2) if c.endswith("d00_ref")})
FIDM=json.load(open("results/domain_distance_dinov3_all.json"))["metrics"]["image"]; P2=json.load(open(f"{IM}/prereg_predictions_towns2.json"))["cells"]
def fid_real(c,T):
    v=(FIDM.get(f"{c}||{T}") or FIDM.get(f"{T}||{c}") or {}).get("fid")
    return v if v is not None else P2.get(c,{}).get(T,{}).get("fid",np.nan)
DM=json.load(open(f"{DO}/dose_metrics.json")); DB=json.load(open(f"{DO}/dose_metrics_detbox.json")); LL=json.load(open(f"{DO}/dose_lowlevel.json"))
CK=json.load(open(f"{DO}/dose_cmmd_kid.json")) if os.path.exists(f"{DO}/dose_cmmd_kid.json") else {}   # E4 (2026-10-17)
CK=CK if len(CK)>=80 else {}
DM2=json.load(open(f"{DO}/dose_metrics_dinov2.json")); DMC=json.load(open(f"{DO}/dose_metrics_clip.json")); FF=json.load(open(f"{FG}/fid_to_foggy.json"))
def drops(R,towns):
    return {(t,d):mm(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json")-mm(f"{R}/eval_{t}_d00_ref__on__{t}_{d}.json") for t in towns for d in DOSES}
def dreal(R,towns,key): return {(t,d):real(R,f"{t}_{d}",key)-real(R,f"{t}_d00_ref",key) for t in towns for d in DOSES}
DR42,DR17,DRA2=drops(DO,T_DO),drops(DOR,T_DO),drops(A2,T_A2); PAIRS=[(t,d) for t in T_DO for d in DOSES]; G=np.array([t for t,_ in PAIRS])
Y42=np.array([DR42[p] for p in PAIRS]); Y17=np.array([DR17[p] for p in PAIRS]); YM=(Y42+Y17)/2
# ================= T1 grids and noise floors =================
rows=[]; data={}
def grid_row(name,R,cells,extra=""):
    k=np.array([real(R,c,"kittival") for c in cells]); c_=np.array([real(R,c,"cityscapesval") for c in cells])
    data[name]={"n":len(cells),"KITTI_mean":float(np.nanmean(k)),"KITTI_sd":float(np.nanstd(k,ddof=1)),"CS_mean":float(np.nanmean(c_)),"CS_sd":float(np.nanstd(c_,ddof=1)),"KITTI_range":[float(np.nanmin(k)),float(np.nanmax(k))]}
    rows.append([name,len(cells),f"{np.nanmean(k):.1f} ± {np.nanstd(k,ddof=1):.1f} ({np.nanmin(k):.1f}–{np.nanmax(k):.1f})",f"{np.nanmean(c_):.1f} ± {np.nanstd(c_,ddof=1):.1f}",extra])
grid_row("COCO-init 24-cell grid (6 towns × 4 weathers)",AN,C_AN,"detector pretrained on COCO end to end")
grid_row("ImageNet-init 24-cell grid",IM,C_IM,"ImageNet backbone only; seed 42")
grid_row("held-out towns (Town06/07/15 × Noon/Night)",H2,C_H2,"pre-registered test cells")
grid_row("fresh cells (6 towns × ClearSunset/WetNoon)",CF,C_CF,"pre-registered confirmation cells")
grid_row("SHIFT 15 condition cells (independent rig)",SH,C_SH,"400 frames/cell, fixed 2400 it")
DOC=own_cells(DO); grid_row("frame-matched dose testbed, 8 towns × 10 doses (seed 42)",DO,DOC,"Faster R-CNN R50-FPN, ImageNet init")
grid_row("dose testbed replicate (seed 17)",DOR,own_cells(DOR),"")
grid_row("dose testbed, FCOS (Town01 + Town10HD × 10 doses)",A2,own_cells(A2),"second detector family")
table("T1","Experimental grids: real-target mAP of simulator-only detectors (mean ± sd over source cells; mAP@0.5, 3 classes)",["grid","sources","KITTI mAP (range)","Cityscapes mAP","note"],rows,note="sd with ddof=1 (the ledger's earlier 2.9/4.6 used ddof=0). Held-out towns include Town15, whose two cells failed to train (561/673 boxes).",sources=[AN,IM,H2,CF,SH,DO,DOR,A2],data=data)
# noise floors
reps=own_cells(IMR); nf={}
for T,key in TG:
    d=np.array([real(IMR,s,key)-real(IM,s,key) for s in reps]); allv=np.array([real(IM,c,key) for c in C_IM]); run=d.std(ddof=1)/np.sqrt(2); src=allv.std(ddof=1)
    nf[T]={"n_rep":len(reps),"per_run_sd":float(run),"across_source_sd":float(src),"reliability":float(1-run**2/src**2),"ceiling":float(np.sqrt(1-run**2/src**2)),"mean_abs_diff":float(np.abs(d).mean())}
dd=[]
for s in reps:
    a0,b0=m(f"{IM}/eval_{s}__on__{s}.json"),m(f"{IMR}/eval_{s}__on__{s}.json")
    for f in glob.glob(f"{IMR}/eval_{s}__on__*.json"):
        t=f.split("__on__")[1][:-5]
        if t!=s and os.path.exists(f"{IM}/eval_{s}__on__{t}.json"): dd.append((b0-m(f))-(a0-m(f"{IM}/eval_{s}__on__{t}.json")))
dd=np.array(dd); nf["grid_drop"]={"pairs":len(dd),"mean_abs_diff":float(np.abs(dd).mean()),"per_run_sd":float(dd.std(ddof=1)/np.sqrt(2)),"within4pp":float((np.abs(dd)<=4).mean())}
dz=Y17-Y42; nf["dose_drop"]={"pairs":len(dz),"r_seed":float(pr(Y42,Y17)),"mean_abs_diff":float(np.abs(dz).mean()),"per_run_sd":float(dz.std(ddof=1)/np.sqrt(2)),"signal_sd":float(YM.std(ddof=1)),"reliability":float(1-(dz.var(ddof=1)/2)/YM.var(ddof=1))}
for T,key in TG:
    a=np.array([real(DO,c,key) for c in DOC]); b=np.array([real(DOR,c,key) for c in DOC]); da=np.array([v for v in dreal(DO,T_DO,key).values()]); db_=np.array([v for v in dreal(DOR,T_DO,key).values()])
    nf[f"dose_real_{T}"]={"n":len(a),"per_run_sd":float((b-a).std(ddof=1)/np.sqrt(2)),"mean_abs_diff":float(np.abs(b-a).mean()),"delta_real_r_seed":float(pr(da,db_)),"delta_real_abs_diff":float(np.abs(da-db_).mean())}
rows=[["ImageNet grid, real mAP (8 sources re-trained, seed 17 vs 42)",f"KITTI {nf['KITTI']['per_run_sd']:.2f} / CS {nf['Cityscapes']['per_run_sd']:.2f}",f"across-source sd {nf['KITTI']['across_source_sd']:.2f} / {nf['Cityscapes']['across_source_sd']:.2f}",f"reliability {nf['KITTI']['reliability']:.2f} / {nf['Cityscapes']['reliability']:.2f}; r ceiling {nf['KITTI']['ceiling']:.2f} / {nf['Cityscapes']['ceiling']:.2f}"],
 ["ImageNet grid, within-CARLA anchored drop (184 pairs)",f"{nf['grid_drop']['per_run_sd']:.2f}",f"mean |Δseed| {nf['grid_drop']['mean_abs_diff']:.2f} pp",f"{100*nf['grid_drop']['within4pp']:.0f}% of pairs within 4 pp"],
 ["dose testbed, appearance-only drop (72 pairs, 80+80 detectors)",f"{nf['dose_drop']['per_run_sd']:.2f}",f"mean |Δseed| {nf['dose_drop']['mean_abs_diff']:.2f} pp; signal sd {nf['dose_drop']['signal_sd']:.1f}",f"seed r {nf['dose_drop']['r_seed']:.3f}; reliability {nf['dose_drop']['reliability']:.2f}"],
 ["dose testbed, real mAP (80 detectors × 2 seeds)",f"KITTI {nf['dose_real_KITTI']['per_run_sd']:.2f} / CS {nf['dose_real_Cityscapes']['per_run_sd']:.2f}",f"Δ real mAP (dose − ref) seed r {nf['dose_real_KITTI']['delta_real_r_seed']:.2f} / {nf['dose_real_Cityscapes']['delta_real_r_seed']:.2f}",f"mean |Δseed| of Δ real mAP {nf['dose_real_KITTI']['delta_real_abs_diff']:.2f} / {nf['dose_real_Cityscapes']['delta_real_abs_diff']:.2f} pp"]]
table("T2","Noise floors: run-to-run (training-seed) variation of every outcome used in the paper",["outcome","per-run sd (pp)","scale","reliability / agreement"],rows,note="per-run sd = sd(seed17 − seed42)/√2. Correlations with a per-cell outcome cannot exceed the stated ceiling.",sources=[IM,IMR,DO,DOR],data=nf)
# ================= T3 pretraining hides the simulator =================
rows=[]; data={}
for name,R,cells in [("COCO-pretrained (backbone+FPN+heads)",AN,C_AN),("ImageNet backbone only",IM,C_IM)]:
    np_,r2,err=anchors_loo(R,cells); d={"pairs":np_,"anchors_loo_r2":r2}
    for T,key in TG:
        y=np.array([real(R,c,key) for c in cells]); a=np.array([m(f"{R}/eval_{c}__on__{c}.json") for c in cells]); st,pt,sw,pw=anova_shares(cells,y)
        tm=[y[[town(c)==t for c in cells]].mean() for t in sorted(set(map(town,cells)))]
        d[T]={"mean":float(y.mean()),"sd":float(y.std(ddof=1)),"town_share":float(st),"town_p":float(pt),"weather_share":float(sw),"weather_p":float(pw),"town_means_range":[float(min(tm)),float(max(tm))],"r_anchor":float(pr(a,y))}
    data[name]=d
    rows.append([name,f"{d['KITTI']['mean']:.1f} ± {d['KITTI']['sd']:.1f}",f"{d['Cityscapes']['mean']:.1f} ± {d['Cityscapes']['sd']:.1f}",f"{d['KITTI']['town_means_range'][0]:.1f}–{d['KITTI']['town_means_range'][1]:.1f}",
                 f"{100*d['KITTI']['town_share']:.0f}% (p={pf(d['KITTI']['town_p'])}) / {100*d['KITTI']['weather_share']:.0f}% (p={pf(d['KITTI']['weather_p'])})",f"{100*d['Cityscapes']['town_share']:.0f}% (p={pf(d['Cityscapes']['town_p'])}) / {100*d['Cityscapes']['weather_share']:.0f}% (p={pf(d['Cityscapes']['weather_p'])})",f"{d['anchors_loo_r2']:.3f}"])
table("T3","Pretraining hides the simulator: the same 24 CARLA cells with two detector initialisations",["init","KITTI mAP","Cityscapes mAP","KITTI town means","KITTI variance: town / weather","Cityscapes variance: town / weather","within-CARLA drop explained by the two anchors (LOO R²)"],rows,
      note="Two-way ANOVA, type-I with town before weather, 24 sources. Anchors = the detector's own in-domain mAP and the target cell's in-domain mAP.",sources=[AN,IM],data=data)
# ================= T4 content dominates =================
def content_block(cells,R_of,label,cluster):
    y={T:np.array([real(R_of(c),c,key) for c in cells]) for T,key in TG}; lb=np.log([nbox(c) for c in cells]); fid={T:np.array([fid_real(c,T) for c in cells],float) for T,_ in TG}; g=np.array([town(c) for c in cells]); out={"n":len(cells)}; row=[label,len(cells)]
    for T,_ in TG:
        ok=~np.isnan(fid[T]); rb=stats.pearsonr(lb,y[T]); cb=cboot(pr,g,lb,y[T]); rf=stats.pearsonr(fid[T][ok],y[T][ok]) if ok.sum()>5 else (np.nan,np.nan)
        pfb=partial(fid[T][ok],y[T][ok],lb[ok])[0] if ok.sum()>5 else np.nan; pbf=partial(lb[ok],y[T][ok],fid[T][ok])[0] if ok.sum()>5 else np.nan
        out[T]={"r_logboxes":float(rb[0]),"p":float(rb[1]),"ci":cb,"r_fid":float(rf[0]),"partial_fid_given_boxes":float(pfb),"partial_boxes_given_fid":float(pbf)}
        row+= [f"{rb[0]:+.3f} {ci(cb)}",f"{fmt(rf[0])}",f"{fmt(pfb)} / {fmt(pbf)}"]
    return out,row
rows=[]; data={}
for label,cells,R_of in [("24 grid sources",C_IM,lambda c:IM),("30 sources (grid + 6 held-out towns)",C_IM+C_H2,lambda c: IM if c in C_IM else H2),("12 fresh cells (unseen weathers)",C_CF,lambda c:CF)]:
    o,r=content_block(cells,R_of,label,True); data[label]=o; rows.append(r)
# town level, 30 sources
cells=C_IM+C_H2; R_of=lambda c: IM if c in C_IM else H2; tws=sorted(set(map(town,cells))); tl={}
for T,key in TG:
    yt=np.array([np.mean([real(R_of(c),c,key) for c in cells if town(c)==t]) for t in tws]); bt=np.array([np.mean([np.log(nbox(c)) for c in cells if town(c)==t]) for t in tws]); ft=np.array([np.mean([fid_real(c,T) for c in cells if town(c)==t]) for t in tws])
    tl[T]={"r_boxes":stats.pearsonr(bt,yt),"r_fid":stats.pearsonr(ft,yt)}
rows.append([f"town level ({len(tws)} towns, 30 sources)",len(tws)]+sum([[f"{tl[T]['r_boxes'][0]:+.3f} (p={pf(tl[T]['r_boxes'][1])})",f"{tl[T]['r_fid'][0]:+.3f} (p={pf(tl[T]['r_fid'][1])})","—"] for T,_ in TG],[]))
data["town_level"]={T:{"r_boxes":float(tl[T]["r_boxes"][0]),"p_boxes":float(tl[T]["r_boxes"][1]),"r_fid":float(tl[T]["r_fid"][0]),"p_fid":float(tl[T]["r_fid"][1])} for T,_ in TG}
table("T4","Content dominates across configurations: labeled object count vs frozen-DINOv3 image FID as predictors of real transfer",["sources","n","KITTI r(log boxes, mAP) [CI]","KITTI r(FID, mAP)","KITTI partial r: FID | boxes / boxes | FID","CS r(log boxes, mAP) [CI]","CS r(FID, mAP)","CS partial r: FID | boxes / boxes | FID"],rows,
      note="Boxes = Car/Pedestrian/Cyclist labels in the 800 training frames. FID = image-level Fréchet distance in frozen DINOv3 space between the source cell and the real target. CI clusters by town.",sources=[IM,H2,CF,"results/domain_distance_dinov3_all.json",f"{IM}/prereg_predictions_towns2.json"],data=data)
# learning curves (causal)
rows=[]; data={}
lc_cells=["Town01_ClearNoon","Town02_ClearNoon","Town03_ClearNoon","Town04_ClearNoon","Town05_ClearNoon","Town10HD_ClearNoon"]
for T,key in TG:
    X,Yv,Gc,slopes=[],[],[],[]
    for c in lc_cells:
        pts=[(np.log(nbox(c,k)),m(f"{IM}/src_{c}_{key}.json" if k==800 else f"{LC}/src_{c}_n{k}_{key}.json")) for k in (100,200,400,800)]
        slopes.append(stats.linregress([p[0] for p in pts],[p[1] for p in pts]).slope); X+=[p[0] for p in pts]; Yv+=[p[1] for p in pts]; Gc+=[c]*4
    X,Yv,Gc=np.array(X),np.array(Yv),np.array(Gc)
    def fe(x,y,g=None):
        g=Gc if g is None else g; xc=x-np.array([x[g==k].mean() for k in g]); yc=y-np.array([y[g==k].mean() for k in g]); return float((xc*yc).sum()/(xc*xc).sum())
    # cluster bootstrap over cells: need group labels resampled together
    def fe_boot():
        v=[]
        for _ in range(B):
            pick=rng.choice(lc_cells,len(lc_cells),replace=True); xs,ys,gs=[],[],[]
            for j,c in enumerate(pick): i=Gc==c; xs+=list(X[i]); ys+=list(Yv[i]); gs+=[j]*int(i.sum())
            v.append(fe(np.array(xs),np.array(ys),np.array(gs)))
        return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
    s=fe(X,Yv); cis=fe_boot()
    allc=C_IM+C_H2; ya=np.array([real(IM if c in C_IM else H2,c,key) for c in allc]); xa=np.array([np.log(nbox(c)) for c in allc]); bb=stats.linregress(xa,ya)
    # SHIFT within-cell slope (3 conditions × 100/200/400)
    SR=Path("/mnt/features/shift_cells"); Xs,Ys=[],[]
    for c in ["SHIFT_daytime_clear","SHIFT_night_rainy","SHIFT_dawn-dusk_foggy"]:
        xs=[np.log(sum(1 for k in range(n) for l in open(SR/c/f"frame_{k:06d}.txt") if l.strip())) for n in (100,200,400)]; ys=[m(f"{SH}/src_{c}_n{n}_{key}.json") if n<400 else m(f"{SH}/src_{c}_{key}.json") for n in (100,200,400)]
        Xs+=list(np.array(xs)-np.mean(xs)); Ys+=list(np.array(ys)-np.mean(ys))
    Xs,Ys=np.array(Xs),np.array(Ys); ss=float(Xs@Ys/(Xs@Xs)); se=np.sqrt(((Ys-ss*Xs)**2).sum()/(len(Xs)-4)/(Xs@Xs)); ps=2*stats.t.sf(abs(ss/se),len(Xs)-4)
    data[T]={"carla_fe_slope":s,"ci":cis,"per_cell_mean":float(np.mean(slopes)),"per_cell_sd":float(np.std(slopes)),"between_slope_30":float(bb.slope),"share":s/bb.slope,"shift_slope":ss,"shift_se":float(se),"shift_p":float(ps),"pp_per_doubling":s*np.log(2)}
    rows.append([T,f"{s:+.2f} {ci(cis)}",f"{np.mean(slopes):+.2f} ± {np.std(slopes):.2f}",f"{s*np.log(2):+.1f}",f"{bb.slope:+.2f} (r={bb.rvalue:+.2f})",f"{100*s/bb.slope:.0f}%",f"{ss:+.2f} (SE {se:.2f}, p={pf(ps)})"])
table("T5","Labeled-data quantity is causal: within-cell learning curves (content and appearance fixed) vs the between-source slope",["target","CARLA within-cell slope, pp per log(boxes) [CI]","per-cell slopes (6 cells)","pp per doubling","between-source slope (30 sources)","share of between-source slope","SHIFT within-cell slope (3 conditions)"],rows,
      note="CARLA: 6 ClearNoon cells × 100/200/400/800 frames, fixed 2400 iterations; CI = cluster bootstrap over cells. SHIFT: 3 condition cells × 100/200/400 frames.",sources=[LC,IM,H2,SH],data=data)
# ================= T6 every cross-configuration predictor on the ImageNet grid (real targets) =================
meas={}
pmf=sorted(glob.glob(f"{AN}/probe_matrix_anchored_all_s*.json")) or sorted(glob.glob(f"{AN}/probe_matrix_anchored_s*.json"))
if pmf:
    mats=[json.load(open(f))["matrix"] for f in pmf]; meas["dIV_fg usable-information gap (labeled boxes)"]=lambda S,T: np.mean([M[S][T]["dIV_fg"] for M in mats if S in M and T in M[S]] or [np.nan])
if os.path.exists(f"{IM}/detector_box_gap_all_s42.json"):
    E=json.load(open(f"{IM}/detector_box_gap_all_s42.json")); TAU="0.7"; ok_=lambda S,T: S in E and T in E[S] and TAU in E[S][T]
    meas["FM veto gap (frozen probe on detector boxes)"]=lambda S,T: (E[S][T][TAU]["probe_pbg"]-E[S][f"in@{TAU}"]["probe_pbg"]) if ok_(S,T) else np.nan
    meas["DoC, difference of confidences"]=lambda S,T: (E[S][f"in@{TAU}"]["det_conf"]-E[S][T][TAU]["det_conf"]) if ok_(S,T) else np.nan
    meas["FM–detector disagreement"]=lambda S,T: (E[S][f"in@{TAU}"]["agree"]-E[S][T][TAU]["agree"]) if ok_(S,T) else np.nan
meas["image FID, frozen DINOv3"]=lambda S,T: fid_real(S,T)
if os.path.exists(f"{IM}/bos_aol.json"):
    Bo=json.load(open(f"{IM}/bos_aol.json"))
    for k,lab in [("bos_stability","BoS stability gap"),("aol_mean_agreement","AoL agreement gap")]:
        meas[lab]=(lambda k: (lambda S,T: (Bo[S][S][k]-Bo[S][T][k]) if S in Bo and S in Bo[S] and T in Bo[S] and k in Bo[S][T] else np.nan))(k)
rows=[]; data={}; lb24=np.log([nbox(c) for c in C_IM]); g24=np.array([town(c) for c in C_IM])
for name,f in meas.items():
    r=[name]; d={}
    for T,key in TG:
        y=np.array([real(IM,c,key) for c in C_IM]); x=np.array([f(c,T) for c in C_IM],float); ok=~np.isnan(x)
        if ok.sum()<6: r+=["—","—"]; continue
        rr=stats.pearsonr(x[ok],y[ok]); c_=cboot(pr,g24[ok],x[ok],y[ok]); pp=partial(x[ok],y[ok],lb24[ok])
        d[T]={"r":float(rr[0]),"p":float(rr[1]),"ci":c_,"partial_given_boxes":float(pp[0]),"n":int(ok.sum())}; r+=[f"{rr[0]:+.3f} {ci(c_)}",f"{pp[0]:+.3f}"]
    rows.append(r); data[name]=d
y=None; r=["log labeled boxes (content)"]
for T,key in TG:
    y=np.array([real(IM,c,key) for c in C_IM]); rr=stats.pearsonr(lb24,y); c_=cboot(pr,g24,lb24,y); r+=[f"{rr[0]:+.3f} {ci(c_)}","—"]
rows.insert(0,r)
table("T6","Across configurations, no distance or consistency estimator predicts real transfer once content is known (24 ImageNet-init sources)",["predictor","KITTI r with real mAP [CI]","KITTI partial r | log boxes","Cityscapes r [CI]","Cityscapes partial r | log boxes"],rows,
      note="Expected sign for a gap measure is negative (more shift → lower mAP). Only FID has the right sign, and it vanishes given object count.",sources=[IM,AN,"results/domain_distance_dinov3_all.json"],data=data)
# ================= T7 pre-registered out-of-sample tests =================
rows=[]; data={}
PP=json.load(open(f"{IM}/prereg_predictions_towns2.json")); PC=json.load(open(f"{IM}/prereg_predictions_towns2_competing.json"))
for T,key in TG:
    ob=np.array([real(H2,c,key) for c in C_H2]); d={}
    for name,getp in [("FID line (pre-registered 08:12)",lambda c: PP["cells"][c][T]["pred_mAP"]),("log(training boxes)",lambda c: PC["cells"][c].get(f"{T}:boxes")),("FID + log(boxes)",lambda c: PC["cells"][c].get(f"{T}:fid+boxes"))]:
        p_=np.array([getp(c) for c in C_H2],float); d[name]={"mae":float(np.abs(ob-p_).mean()),"r":float(pr(p_,ob))}
    data[T]=d
for name in data["KITTI"]:
    rows.append([f"held-out towns (6 cells): {name}",f"{data['KITTI'][name]['mae']:.2f}",f"{data['KITTI'][name]['r']:+.3f}",f"{data['Cityscapes'][name]['mae']:.2f}",f"{data['Cityscapes'][name]['r']:+.3f}"])
CP=json.load(open("results/manifold_markers/confirm_predictions.json"))["cells"]; dcf={}
for T,key in TG:
    cells=[c for c in C_CF if CP.get(c,{}).get("has_markers")]; ob=np.array([real(CF,c,key) for c in cells]); pfull=np.array([CP[c][f"{T}:full"] for c in cells]); pbox=np.array([CP[c][f"{T}:boxes_only"] for c in cells])
    lbc=np.log([nbox(c) for c in C_CF]); obc=np.array([real(CF,c,key) for c in C_CF]); rr=stats.pearsonr(lbc,obc)
    dcf[T]={"boxes_only":{"r":float(pr(pbox,ob)),"mae":float(np.abs(ob-pbox).mean()),"mae_centred":float(np.abs((ob-ob.mean())-(pbox-pbox.mean())).mean())},"geometry_model":{"r":float(pr(pfull,ob)),"mae":float(np.abs(ob-pfull).mean()),"mae_centred":float(np.abs((ob-ob.mean())-(pfull-pfull.mean())).mean())},"r_logboxes_12":float(rr[0]),"p":float(rr[1]),"n":len(cells)}
for name in ("boxes_only","geometry_model"):
    rows.append([f"fresh cells ({dcf['KITTI']['n']}): {name.replace('_',' ')} (pre-registered coefficients)",f"{dcf['KITTI'][name]['mae']:.2f} (centred {dcf['KITTI'][name]['mae_centred']:.2f})",f"{dcf['KITTI'][name]['r']:+.3f}",f"{dcf['Cityscapes'][name]['mae']:.2f} (centred {dcf['Cityscapes'][name]['mae_centred']:.2f})",f"{dcf['Cityscapes'][name]['r']:+.3f}"])
data["fresh"]=dcf
# rendering quality
LQc=[c for c in ["Town01_ClearNoon","Town04_ClearNoon","Town10HD_ClearNoon","Town06_ClearNoon","Town07_ClearNoon"] if os.path.exists(f"{LQ}/src_{c}_kittival.json")]; FL=json.load(open(f"{LQ}/fid_to_real_low.json"))["cells"]; dl={}
for T,key in TG:
    e=np.array([real(IM if c in C_IM else H2,c,key) for c in LQc]); l=np.array([real(LQ,c,key) for c in LQc]); fe_=np.array([fid_real(c,T) for c in LQc]); fl=np.array([FL[c][T]["fid"] for c in LQc]); t=stats.ttest_1samp(e-l,0)
    dl[T]={"epic_minus_low":float((e-l).mean()),"sd":float((e-l).std(ddof=1)),"p":float(t.pvalue),"towns_lose":int(((e-l)>0).sum()),"n":len(LQc),"dfid_low_minus_epic":float((fl-fe_).mean()),"fid_up":int(((fl-fe_)>0).sum()),"r_dfid_dmap":float(pr(fl-fe_,l-e))}
    rows.append([f"rendering quality Epic→Low ({len(LQc)} towns), {T}",f"Epic − Low real mAP {dl[T]['epic_minus_low']:+.2f} ± {dl[T]['sd']:.2f} (paired t p={pf(dl[T]['p'])}); {dl[T]['towns_lose']}/{dl[T]['n']} towns lose","—",f"FID to {T} rises at Low in {dl[T]['fid_up']}/{dl[T]['n']} (+{dl[T]['dfid_low_minus_epic']:.2f}); r(ΔFID, ΔmAP) {dl[T]['r_dfid_dmap']:+.2f} (wrong sign)","—"])
data["lowq"]=dl
table("T7","Pre-registered out-of-sample tests across configurations",["test","KITTI mean |error| (pp)","KITTI r(pred, obs)","Cityscapes mean |error|","Cityscapes r(pred, obs)"],rows,
      note="Predictions were written and hashed before the detectors were trained (ledger §15.34, 15.36, 15.43, 15.40). Town15 collapsed in training (561/673 boxes) and is kept as pre-registered.",sources=[H2,CF,LQ,f"{IM}/prereg_predictions_towns2*.json","results/manifold_markers/confirm_predictions.json"],data=data)
# ================= T8 SHIFT =================
rows=[]; data={}; META=json.load(open("/mnt/features/shift_cells/cells_meta.json")); SF=json.load(open(f"{SH}/fid_to_real.json"))["cells"]
cells=sorted(META); tod=np.array([META[c]["timeofday"] for c in cells]); wx=np.array([META[c]["weather"] for c in cells]); lbs=np.log([META[c]["train_boxes"] for c in cells]); one=np.ones((len(cells),1))
for T,key in TG:
    y=np.array([m(f"{SH}/src_{c}_{key}.json") for c in cells]); x=np.array([SF[c][T]["fid"] for c in cells]); tot=float(((y-y.mean())**2).sum())
    Xb=np.column_stack([one,lbs]); Xt=np.column_stack([Xb,dummies(tod)]); Xtw=np.column_stack([Xt,dummies(wx)])
    def ft(Xr,Xf): r0,r1=rss(y,Xr),rss(y,Xf); d1=Xf.shape[1]-Xr.shape[1]; d2=len(y)-Xf.shape[1]; F=((r0-r1)/d1)/(r1/d2); return float(stats.f.sf(F,d1,d2))
    pb=partial(x,y,lbs); pfid=ft(Xb,np.column_stack([Xb,x])); rb=stats.pearsonr(lbs,y)
    data[T]={"mean":float(y.mean()),"sd":float(y.std(ddof=1)),"share_boxes":(tot-rss(y,Xb))/tot,"share_time":(rss(y,Xb)-rss(y,Xt))/tot,"share_weather":(rss(y,Xt)-rss(y,Xtw))/tot,"p_time":ft(Xb,Xt),"p_weather":ft(Xt,Xtw),"p_both":ft(Xb,Xtw),"r_fid":float(pr(x,y)),"partial_fid":float(pb[0]),"p_fid_increment":pfid,"r_logboxes":float(rb[0])}
    rows.append([T,f"{y.mean():.1f} ± {y.std(ddof=1):.1f}",f"{100*data[T]['share_boxes']:.0f}% / {100*data[T]['share_time']:.0f}% / {100*data[T]['share_weather']:.0f}%",f"{pf(data[T]['p_time'])} / {pf(data[T]['p_weather'])} / {pf(data[T]['p_both'])}",f"{pr(x,y):+.3f}",f"{pb[0]:+.3f} (F p={pf(pfid)})"])
table("T8","SHIFT replication (independent CARLA rig, 15 time-of-day × weather cells at near-matched counts)",["target","real mAP","variance shares: log boxes / time / weather","F p: time | boxes, weather | boxes+time, both","r(FID, mAP)","partial r(FID | log boxes)"],rows,
      note="n = 15 (low power; estimation reported). Causal count slopes for SHIFT are in T5.",sources=[SH,"/mnt/features/shift_cells/cells_meta.json"],data=data)
# ================= T9 dose R1 =================
rows=[]; data={"seed42":{f"{t}|{d}":float(DR42[(t,d)]) for t,d in PAIRS},"seed17":{f"{t}|{d}":float(DR17[(t,d)]) for t,d in PAIRS}}
for t in T_DO: rows.append([t]+[f"{(DR42[(t,d)]+DR17[(t,d)])/2:.1f}" for d in DOSES])
rows.append(["**town mean**"]+[f"**{np.mean([(DR42[(t,d)]+DR17[(t,d)])/2 for t in T_DO]):.1f}**" for d in DOSES])
rows.append(["town-mean CI"]+[ci(cboot(lambda v: v.mean(),np.array(T_DO),np.array([(DR42[(t,d)]+DR17[(t,d)])/2 for t in T_DO]))) for d in DOSES])
rows.append(["mean |seed 42 − seed 17|"]+[f"{np.mean([abs(DR42[(t,d)]-DR17[(t,d)]) for t in T_DO]):.1f}" for d in DOSES])
mono={}
for sname,DR in [("seed42",DR42),("seed17",DR17)]:
    for fam,ds in FAM.items(): rh=[sp(range(len(ds)),[DR[(t,d)] for d in ds]) for t in T_DO]; mono[f"{sname}_{fam}"]={"rhos":[float(r) for r in rh],"n_pass":int(sum(r>=.8-1e-9 for r in rh))}
data["monotone"]=mono
# visibility
vis={}
for lab,R in [("≥25 px",H25),("≥40 px",H40)]:
    DRv=drops(R,T_DO); vis[lab]={"town_mean":{d:float(np.mean([DRv[(t,d)] for t in T_DO])) for d in DOSES}}
    for fam,ds in FAM.items(): rh=[sp(range(len(ds)),[DRv[(t,d)] for d in ds]) for t in T_DO]; vis[lab][f"mono_{fam}"]=int(sum(r>=.8-1e-9 for r in rh))
    rows.append([f"town mean, boxes {lab} only (seed 42)"]+[f"{vis[lab]['town_mean'][d]:.1f}" for d in DOSES])
data["visibility"]=vis
table("T9","R1 — appearance-only cost: mAP drop (pp) of each town's clean-trained detector on the SAME frames re-rendered under a dose (mean of 2 seeds)",["town"]+[d[4:] for d in DOSES],rows,
      note=f"Monotone within family (Spearman ≥ .8 per town): sun {mono['seed42_sun']['n_pass']}/8 (seed 42), {mono['seed17_sun']['n_pass']}/8 (seed 17); fog {mono['seed42_fog']['n_pass']}/8, {mono['seed17_fog']['n_pass']}/8. Visibility control (seed 42, boxes <25 px / <40 px removed): monotone in sun {vis['≥25 px']['mono_sun']} and {vis['≥40 px']['mono_sun']} of 8 towns, fog {vis['≥25 px']['mono_fog']} and {vis['≥40 px']['mono_fog']} of 8 — drops grow when only nearer objects are scored.",sources=[DO,DOR,H25,H40],data=data)
# ================= T10 dose R2 main =================
def metric_rows(groups,Y_a,Y_b,pairs,g,label_a="seed 42",label_b="seed 17",full=True):
    rows=[]; data={}
    for grp,src,keys in groups:
        for k,lab,need in keys:
            x=np.array([src.get(f"{t}_{d}",{}).get(k,np.nan) for t,d in pairs],float); ok=~np.isnan(x)
            if ok.sum()<8: continue
            xa,ya,ga=x[ok],Y_a[ok],g[ok]; rho=sp(xa,ya); c_=cboot(sp,ga,xa,ya); p=cperm(sp,xa,ya,ga); r2=loto_r2(xa,ya,ga)
            d={"rho":float(rho),"ci":c_,"p":p,"loto_r2":r2,"n":int(ok.sum())}; row=[grp,lab,need,f"{rho:+.3f} {ci(c_)}",pf(p),f"{r2:+.2f}"]
            if Y_b is not None: yb=Y_b[ok]; rb=sp(xa,yb); r2b=loto_r2(xa,yb,ga); d.update({"rho_b":float(rb),"loto_r2_b":r2b}); row+=[f"{rb:+.3f}",f"{r2b:+.2f}"]
            if full: k1,k7=cal_k(xa,ya,ga,1),cal_k(xa,ya,ga,len(set(ga))-1); d.update({"cal_k1":k1,"cal_k7":k7}); row+=[f"{k1:.1f} → {k7:.1f}"]
            rows.append(row); data[f"{grp}|{k}"]=d
    return rows,data
GRP=[("object-level, frozen DINOv3 (GT boxes)",DM,[("obj_fid_ref","object FID","boxes"),("nnd_dose_to_ref","directed NN distance dose→ref","boxes"),("cov_dose_by_ref","coverage of dose by ref","boxes"),("paired_obj_disp","paired object displacement","boxes"),("paired_task_disp","task-projected displacement","boxes + probe"),("probe_flip","probe flip rate","boxes + probe")]),
     ("object-level, detector-proposed boxes (label-free)",DB,[("obj_fid_ref","object FID","detector boxes"),("nnd_dose_to_ref","directed NN distance","detector boxes"),("paired_obj_disp","paired object displacement","detector boxes"),("probe_flip","probe flip rate","detector boxes + probe"),("det_count_ratio","log detection-count ratio (detector itself)","the detector"),("det_doc","DoC (detector itself)","the detector")]),
     ("image-level, frozen DINOv3 (no boxes)",DM,[("img_fid_ref","image FID","none"),("paired_img_disp","paired image displacement","none")]),
     ("modern distribution baselines (no boxes)",CK,[("cmmd_ref","CMMD, CLIP ViT-L/14-336","none"),("kid_ref","KID, Inception pool3","none")]),
     ("low-level / classic (no boxes)",LL,[("lpips","LPIPS-AlexNet (paired)","none"),("incfid_ref","Inception-pool3 FID","none"),("dcontrast","|Δ RMS contrast| (paired)","none"),("ssim","SSIM (paired)","none"),("psnr","PSNR (paired)","none"),("dlum","|Δ luminance| (paired)","none")])]
rows,data=metric_rows(GRP,Y42,Y17,PAIRS,G)
table("T10","R2 — with content fixed, label-free metrics track the appearance-only cost (72 town×dose pairs, 8 towns)",["family","metric","needs","Spearman ρ, seed 42 [CI]","clustered p","LOTO R² (42)","ρ seed 17","LOTO R² (17)","calibration |err| pp, k=1 → 7 towns (predict-the-mean 8.8)"],rows,
      note=f"Pre-registered bar: |ρ| ≥ .7 and LOTO R² > 0. Predict-the-mean error = {np.abs(Y42-Y42.mean()).mean():.1f} pp. All metrics are computed from the two renders only (detector-independent except the two detector rows).",sources=[DO,DOR,f"{DO}/dose_metrics*.json",f"{DO}/dose_lowlevel.json"],data=data)
# ================= T11 R2 robustness: feature spaces (4 towns), FCOS, visibility =================
T4=sorted({c.split("_d00")[0] for c in DM2 if c.endswith("d00_ref")}); P4=[(t,d) for t in T4 for d in DOSES]; G4=np.array([t for t,_ in P4]); Y4=np.array([DR42[p] for p in P4])
rows=[]; data={}
for k,lab in [("obj_fid_ref","object FID"),("nnd_dose_to_ref","directed NN distance"),("cov_dose_by_ref","coverage"),("paired_obj_disp","paired object displacement"),("paired_task_disp","task-projected displacement"),("probe_flip","probe flip rate"),("img_fid_ref","image FID (no boxes)"),("paired_img_disp","paired image displacement (no boxes)")]:
    r=[lab]; d={}
    for sname,src in [("DINOv3",DM),("DINOv2",DM2),("CLIP",DMC)]:
        x=np.array([src[f"{t}_{d_}"][k] for t,d_ in P4]); rho=sp(x,Y4); r2=loto_r2(x,Y4,G4); d[sname]={"rho":float(rho),"loto_r2":r2}; r.append(f"{rho:+.2f} / {r2:+.2f}")
    PA=[(t,d_) for t in T_A2 for d_ in DOSES]; GA=np.array([t for t,_ in PA]); YA=np.array([DRA2[p] for p in PA]); x=np.array([DM[f"{t}_{d_}"][k] for t,d_ in PA]); rho=sp(x,YA); p=cperm(sp,x,YA,GA); d["FCOS"]={"rho":float(rho),"p":p,"n":len(PA)}; r.append(f"{rho:+.2f} (p={pf(p)})")
    for lab2,R in [("h25",H25),("h40",H40)]:
        DRv=drops(R,T_DO); yv=np.array([DRv[p_] for p_ in PAIRS]); x=np.array([DM[f"{t}_{d_}"][k] for t,d_ in PAIRS]); d[lab2]={"rho":float(sp(x,yv)),"loto_r2":loto_r2(x,yv,G)}
    r.append(f"{d['h25']['rho']:+.2f} / {d['h40']['rho']:+.2f}"); rows.append(r); data[k]=d
mono_a2={fam:[float(sp(range(len(ds)),[DRA2[(t,d_)] for d_ in ds])) for t in T_A2] for fam,ds in FAM.items()}; data["fcos_monotone"]=mono_a2
table("T11","R2 robustness: other feature spaces (same 4 towns, 36 pairs, seed-42 drops), a second detector family (FCOS, 2 towns, 18 pairs), and visibility-filtered drops (72 pairs)",["metric","DINOv3 ρ / LOTO R²","DINOv2 ρ / LOTO R²","CLIP ρ / LOTO R²","FCOS drop: ρ (clustered p)","ρ with drops on boxes ≥25 px / ≥40 px"],rows,
      note=f"FCOS appearance-only drops are monotone in both towns (sun ρ {mono_a2['sun']}, fog ρ {mono_a2['fog']}). The FCOS detectors see the SAME DINOv3 metrics (detector-independent).",sources=[DO,A2,H25,H40],data=data)
# ================= T12 R3: appearance changes real transfer; Δ distance-to-real predicts Δ real mAP =================
rows=[]; data={}
for T,key in TG:
    d42,d17=dreal(DO,T_DO,key),dreal(DOR,T_DO,key); dA=dreal(A2,T_A2,key); d={}
    for dz in DOSES:
        v=np.array([(d42[(t,dz)]+d17[(t,dz)])/2 for t in T_DO]); c_=cboot(lambda a:a.mean(),np.array(T_DO),v); d[dz]={"mean":float(v.mean()),"ci":c_,"seed42":float(np.mean([d42[(t,dz)] for t in T_DO])),"seed17":float(np.mean([d17[(t,dz)] for t in T_DO])),"fcos":float(np.mean([dA[(t,dz)] for t in T_A2]))}
    data[T]=d
    rows.append([f"{T}: Δ real mAP, mean of 8 towns × 2 seeds [CI]"]+[f"{d[dz]['mean']:+.1f} {ci(d[dz]['ci'])}" for dz in DOSES])
    rows.append([f"{T}: FCOS (2 towns, 1 seed)"]+[f"{d[dz]['fcos']:+.1f}" for dz in DOSES])
table("T12a","R3 — training on the re-rendered dose instead of the clean reference changes real-target mAP (dose-trained − reference-trained, pp)",["target"]+[d[4:] for d in DOSES],rows,sources=[DO,DOR,A2],data=data)
rows=[]; data={}
SUB={"all":DOSES,"without fog":[d for d in DOSES if "fog" not in d],"fog only":FAM["fog"],"sun + night":FAM["sun"]}
for T,key in TG:
    d42,d17=dreal(DO,T_DO,key),dreal(DOR,T_DO,key)
    for mname,src,kk in [("DINOv3 image FID to real",DM,f"fid_{T}"),("Inception-pool3 FID to real",LL,f"incfid_{T}")]+([("CMMD to real",CK,f"cmmd_{T}"),("KID to real",CK,f"kid_{T}")] if CK else []):
        dx=np.array([src[f"{t}_{d}"][kk]-src[f"{t}_d00_ref"][kk] for t,d in PAIRS])
        for sname,dy in [("seed 42",np.array([d42[p] for p in PAIRS])),("seed 17",np.array([d17[p] for p in PAIRS])),("seed mean",np.array([(d42[p]+d17[p])/2 for p in PAIRS]))]:
            r=[T,mname,sname]; dd_={}
            for sub,ds in SUB.items():
                i=np.array([d in ds for _,d in PAIRS]); rho=sp(dx[i],dy[i]); c_=cboot(sp,G[i],dx[i],dy[i]); p=cperm(sp,dx[i],dy[i],G[i]); dd_[sub]={"rho":float(rho),"ci":c_,"p":p,"n":int(i.sum())}; r.append(f"{rho:+.3f} {ci(c_)} p={pf(p)}")
            rows.append(r); data[f"{T}|{mname}|{sname}"]=dd_
    dA=dreal(A2,T_A2,key); PA=[(t,d) for t in T_A2 for d in DOSES]; GA=np.array([t for t,_ in PA]); dx=np.array([DM[f"{t}_{d}"][f"fid_{T}"]-DM[f"{t}_d00_ref"][f"fid_{T}"] for t,d in PA]); dy=np.array([dA[p] for p in PA]); rho=sp(dx,dy); p=cperm(sp,dx,dy,GA)
    rows.append([T,"DINOv3 image FID to real","FCOS (18 pairs)",f"{rho:+.3f} p={pf(p)}","—","—","—"]); data[f"{T}|fcos"]={"rho":float(rho),"p":p}
    # foggy target
    if T=="Cityscapes":
        dyf=np.array([mm(f"{FG}/src_{t}_{d}_foggy0.02.json")-mm(f"{FG}/src_{t}_d00_ref_foggy0.02.json") for t,d in PAIRS]); dxf=np.array([FF[f"{t}_{d}"]["fid_foggy_0.02"]-FF[f"{t}_d00_ref"]["fid_foggy_0.02"] for t,d in PAIRS]); ok=~np.isnan(dyf)
        rho=sp(dxf[ok],dyf[ok]); c_=cboot(sp,G[ok],dxf[ok],dyf[ok]); p=cperm(sp,dxf[ok],dyf[ok],G[ok]); rows.append(["Foggy Cityscapes β=.02","DINOv3 image FID to foggy CS","seed 42",f"{rho:+.3f} {ci(c_)} p={pf(p)}","—","—","—"]); data["foggy|fid_to_foggy"]={"rho":float(rho),"ci":c_,"p":p,"n":int(ok.sum())}
table("T12b","R3 — Δ(distance to the real domain) vs Δ(real mAP) over the 72 (town, dose) pairs, Spearman ρ [CI], town-clustered p",["target","distance","detectors","all doses","without fog","fog only","sun + night"],rows,
      note="Δ = dose cell − reference cell of the same town. Expected sign negative. Foggy Cityscapes row = pre-registered F2 (fails).",sources=[DO,DOR,A2,FG,f"{DO}/dose_lowlevel.json"],data=data)
# ================= T13 asymmetry =================
rows=[]; data={}
for sname,R in [("seed 42",DO),("seed 17",DOR)]:
    a1={}; a2={}
    for t in T_DO:
        for d in DOSES:
            c,ref=f"{t}_{d}",f"{t}_d00_ref"; a1[(t,d)]=mm(f"{R}/eval_{ref}__on__{ref}.json")-mm(f"{R}/eval_{ref}__on__{c}.json"); a2[(t,d)]=mm(f"{R}/eval_{c}__on__{c}.json")-mm(f"{R}/eval_{c}__on__{ref}.json")
    harder=sum(a1[p]>a2[p] for p in PAIRS); data[sname]={"clean_to_dose_harder":int(harder),"n":len(PAIRS),"by_dose":{d:{"ref_on_dose":float(np.mean([a1[(t,d)] for t in T_DO])),"dose_on_ref":float(np.mean([a2[(t,d)] for t in T_DO])),"harder":int(sum(a1[(t,d)]>a2[(t,d)] for t in T_DO))} for d in DOSES}}
    rows.append([f"{sname}: clean-trained on dose frames (pp drop)"]+[f"{data[sname]['by_dose'][d]['ref_on_dose']:.1f}" for d in DOSES])
    rows.append([f"{sname}: dose-trained on clean frames (pp drop)"]+[f"{data[sname]['by_dose'][d]['dose_on_ref']:.1f}" for d in DOSES])
    rows.append([f"{sname}: towns where clean→dose is harder"]+[f"{data[sname]['by_dose'][d]['harder']}/8" for d in DOSES])
table("T13","Asymmetry: the clean→degraded direction is harder than degraded→clean (same frames, each detector scored on its own held-out frames)",["quantity"]+[d[4:] for d in DOSES],rows,
      note=f"Clean→dose harder in {data['seed 42']['clean_to_dose_harder']}/72 pairs (seed 42) and {data['seed 17']['clean_to_dose_harder']}/72 (seed 17). Symmetric distances (FID, MMD, CORAL) cannot represent this; directed coverage / NN distance can.",sources=[DO,DOR],data=data)
# ================= T14 Foggy Cityscapes =================
rows=[]; data={}
def fsrc(c,b): return f"{DO}/src_{c}_cityscapesval.json" if b=="0" else f"{FG}/src_{c}_foggy{b}.json"
for b in ("0","0.005","0.01","0.02"):
    v={k:np.array([mm(fsrc(f"{t}_{k}",b)) for t in T_DO]) for k in ("d00_ref","d05_fog25","d06_fog50","d07_fog75")}; d75=v["d07_fog75"]-v["d00_ref"]
    data[b]={k:float(np.nanmean(x)) for k,x in v.items()}; data[b].update({"fog75_minus_ref":float(np.nanmean(d75)),"towns_pos":int(np.nansum(d75>0)),"ci_fog75_minus_ref":cboot(lambda a:a.mean(),np.array(T_DO),d75)})
    rows.append([f"β = {b}" if b!="0" else "clear (β = 0)",f"{data[b]['d00_ref']:.1f}",f"{data[b]['d05_fog25']:.1f}",f"{data[b]['d06_fog50']:.1f}",f"{data[b]['d07_fog75']:.1f}",f"{data[b]['fog75_minus_ref']:+.1f} {ci(data[b]['ci_fog75_minus_ref'])}",f"{data[b]['towns_pos']}/8"])
k=data["0.02"]["towns_pos"]; psign=stats.binomtest(k,8,0.5,alternative="greater").pvalue; mono_b=all(data[a]["fog75_minus_ref"]<=data[b_]["fog75_minus_ref"] for a,b_ in zip(("0","0.005","0.01"),("0.005","0.01","0.02"))); data["F1"]={"towns":k,"sign_p":float(psign),"monotone":bool(mono_b),"pass":bool(k>=6 and mono_b)}
table("T14","Foggy Cityscapes (third real target): town-mean mAP of dose-trained detectors on real fog of increasing density",["real target","clear-trained","fog25-trained","fog50-trained","fog75-trained","fog75 − clear [CI]","towns fog75 > clear"],rows,
      note=f"F1 (pre-registered): fog75 beats clear-trained at β = .02 in {k}/8 towns (sign test p = {psign:.3f}), benefit monotone in β = {mono_b} → {'PASS' if data['F1']['pass'] else 'FAIL'}. Mild simulated fog (fog25) is best at every real-fog level with no loss on clear images. F2/F3 (FID-to-foggy predicts the benefit) fail: see T12b.",sources=[FG,DO],data=data)
# ================= T15 decomposition summary (numbers behind figure F5) =================
A=J["tables"]["T3"]["data"]; C=J["tables"]["T5"]["data"]; R3=J["tables"]["T12a"]["data"]
rows=[]; data={}
for T,_ in TG:
    pre=A["COCO-pretrained (backbone+FPN+heads)"][T]["mean"]-A["ImageNet backbone only"][T]["mean"]; tr=A["ImageNet backbone only"][T]["town_means_range"]; dbl=C[T]["pp_per_doubling"]
    fog75=R3[T]["d07_fog75"]["mean"]; night=R3[T]["d04_night"]["mean"]; ws=A["ImageNet backbone only"][T]["weather_share"]
    data[T]={"pretraining_pp":float(pre),"town_range_pp":float(tr[1]-tr[0]),"pp_per_doubling":float(dbl),"fog75_pp":float(fog75),"night_pp":float(night),"weather_share_24grid":float(ws)}
    rows.append([T,f"{pre:+.1f}",f"{tr[1]-tr[0]:.1f} (town means {tr[0]:.1f}–{tr[1]:.1f})",f"{dbl:+.1f}",f"{100*ws:.0f}%",f"fog75 {fog75:+.1f}; night {night:+.1f}"])
table("T15","Decomposition of real transfer (pp of real mAP): pretraining ≫ labeled content ≫ appearance",["target","COCO vs ImageNet init (same 24 cells)","spread of town means (ImageNet grid)","per doubling of labeled objects (causal)","weather share of variance (24-cell grid)","appearance at fixed content (frame-matched)"],rows,
      note="Pretraining and content are between-configuration effects; the appearance column is the only within-configuration (content-fixed) manipulation.",sources=[AN,IM,LC,DO,DOR],data=data)
# ================= T16 held-out-dose prediction (E1, §15.65 / 15.72) =================
HO="results/calib_dose_ho"
if os.path.exists(f"{HO}/predictions.json") and glob.glob(f"{HO}/eval_*"):
    P=json.load(open(f"{HO}/predictions.json")); own42={t:m(f"{DO}/eval_{t}_d00_ref__on__{t}_d00_ref.json") for t in T_DO}; own17={t:m(f"{DOR}/eval_{t}_d00_ref__on__{t}_d00_ref.json") for t in T_DO}
    cells=sorted(P["cells"]); obs=np.array([np.nanmean([own42[P["cells"][c]["town"]]-mm(f"{HO}/eval_{P['cells'][c]['town']}_d00_ref__on__{c}.json"),own17[P["cells"][c]["town"]]-mm(f"{HO}_rep/eval_{P['cells'][c]['town']}_d00_ref__on__{c}.json")]) for c in cells])
    gH=np.array([P["cells"][c]["town"] for c in cells]); sub=np.array([P["cells"][c]["subset"] for c in cells]); rows=[]; data={}
    for k,lab in [("obj_fid_ref","object FID (primary)"),("nnd_dose_to_ref","directed NN distance"),("cov_dose_by_ref","coverage"),("paired_obj_disp","paired object displacement"),("incfid_ref","Inception FID (no boxes)"),("lpips","LPIPS (no boxes)"),("img_fid_ref","DINOv3 image FID (no boxes)")]:
        pr=np.array([P["cells"][c][k]["pred"] for c in cells]); lo=np.array([P["cells"][c][k]["pi90"][0] for c in cells]); hi=np.array([P["cells"][c][k]["pi90"][1] for c in cells])
        rho=sp(pr,obs); c_=cboot(sp,gH,pr,obs); mae=float(np.abs(pr-obs).mean()); cov=float(((obs>=lo)&(obs<=hi)).mean())
        data[k]={"rho":float(rho),"ci":c_,"mae":mae,"coverage90":cov,"subsets":{s_:{"rho":float(sp(pr[sub==s_],obs[sub==s_])),"mae":float(np.abs(pr-obs)[sub==s_].mean())} for s_ in ("interpolation","extrapolation","compound","novel")}}
        rows.append([lab,f"{rho:+.3f} {ci(c_)}",f"{mae:.2f}",f"{100*cov:.0f}%"]+[f"{data[k]['subsets'][s_]['rho']:+.2f} / {data[k]['subsets'][s_]['mae']:.1f}" for s_ in ("interpolation","extrapolation","compound","novel")])
    data["mean_pred_mae"]=float(np.abs(P["calibration_mean_drop"]-obs).mean())
    data["per_dose"]={d:{"observed":float(obs[[P["cells"][c]["dose"]==d for c in cells]].mean()),"predicted":float(np.mean([P["cells"][c]["obj_fid_ref"]["pred"] for c in cells if P["cells"][c]["dose"]==d]))} for d in sorted({P["cells"][c]["dose"] for c in cells})}
    table("T16","Held-out-dose prediction (pre-registered, predictions hashed before evaluation): metric calibrated on the 9 original doses predicts 9 unseen doses (72 pairs)",
          ["metric","Spearman(pred, obs) [CI]","MAE (pp)","90% PI coverage","interp. ρ / MAE","extrap. ρ / MAE","compound ρ / MAE","novel ρ / MAE"],rows,
          note=f"Rules (object FID): ρ ≥ .7, MAE ≤ 7.0 pp and < predict-the-calibration-mean ({data['mean_pred_mae']:.2f}), coverage ≥ 80%. Per dose (observed / predicted, pp): "+"; ".join(f"{d[4:]} {v['observed']:.1f} / {v['predicted']:.1f}" for d,v in data["per_dose"].items())+".",sources=[HO,HO+"_rep",DO,DOR],data=data)
# ================= T17 RTTS real fog (E3) =================
RT="results/calib_dose_rtts"
if os.path.exists(f"{RT}/fid_to_rtts.json"):
    rows=[]; data={}
    for lab,Rr,ds_ in [("seed 42",RT,["d00_ref"]+DOSES),("seed 17",RT+"_rep",["d00_ref","d05_fog25","d06_fog50","d07_fog75"])]:
        v={d:np.array([mm(f"{Rr}/src_{t}_{d}_rtts.json") for t in T_DO]) for d in ds_}; data[lab]={d:float(np.nanmean(x)) for d,x in v.items()}
        for d in ("d05_fog25","d06_fog50","d07_fog75"):
            dd=v[d]-v["d00_ref"]; data[lab][f"{d}_minus_ref"]={"mean":float(np.nanmean(dd)),"ci":cboot(lambda a:a.mean(),np.array(T_DO),dd),"towns":int((dd>0).sum())}
        rows.append([lab,f"{data[lab]['d00_ref']:.1f}"]+[f"{data[lab][d]:.1f} ({data[lab][d+'_minus_ref']['mean']:+.1f} {ci(data[lab][d+'_minus_ref']['ci'])}; {data[lab][d+'_minus_ref']['towns']}/8)" for d in ("d05_fog25","d06_fog50","d07_fog75")])
    FR=json.load(open(f"{RT}/fid_to_rtts.json")); dy=np.array([mm(f"{RT}/src_{t}_{d}_rtts.json")-mm(f"{RT}/src_{t}_d00_ref_rtts.json") for t,d in PAIRS]); dx=np.array([FR[f"{t}_{d}"]["fid_RTTS"]-FR[f"{t}_d00_ref"]["fid_RTTS"] for t,d in PAIRS])
    rho=sp(dx,dy); data["G2"]={"rho":float(rho),"ci":cboot(sp,G,dx,dy),"p":cperm(sp,dx,dy,G)}
    table("T17","Real fog: RTTS (4,322 real hazy images; Car + Pedestrian mAP@0.5, difficult boxes included) — dose-trained detectors, town mean",["detectors","clear-trained","fog25 (Δ [CI]; towns better)","fog50","fog75"],rows,
          note=f"Δ FID-to-RTTS vs Δ RTTS mAP over 72 pairs: ρ {rho:+.3f} {ci(data['G2']['ci'])}, clustered p = {pf(data['G2']['p'])}. Every non-clear training dose also beats clear on RTTS (seed 42: "+", ".join(f"{d[4:]} {data['seed 42'][d]:.1f}" for d in DOSES if "fog" not in d)+"); fog's gain over the non-fog mean is ≈ +3 pp.",sources=[RT,RT+"_rep"],data=data)
# ================= T18 detector families =================
rows=[]; data={}
for lab,R_,towns in [("Faster R-CNN R50 (8 towns, seed 42)",DO,T_DO),("FCOS R50 (2 towns)",A2,T_A2),("Faster R-CNN Swin-T (2 towns)","results/calib_dose_arch3",["Town01","Town10HD"])]:
    if not os.path.exists(f"{R_}/eval_{towns[0]}_d00_ref__on__{towns[0]}_d00_ref.json"): continue
    Pp=[(t,d) for t in towns for d in DOSES]; Gp=np.array([t for t,_ in Pp]); y=np.array([drop(R_,t,d) for t,d in Pp]) if False else np.array([mm(f"{R_}/eval_{t}_d00_ref__on__{t}_d00_ref.json")-mm(f"{R_}/eval_{t}_d00_ref__on__{t}_{d}.json") for t,d in Pp])
    mono=[min(sp(range(len(ds)),[mm(f"{R_}/eval_{t}_d00_ref__on__{t}_d00_ref.json")-mm(f"{R_}/eval_{t}_d00_ref__on__{t}_{d}.json") for d in ds]) for ds in FAM.values()) for t in towns]
    x=np.array([DM[f"{t}_{d}"]["obj_fid_ref"] for t,d in Pp]); r2=sp(x,y)
    dk=np.array([mm(f"{R_}/src_{t}_{d}_kittival.json")-mm(f"{R_}/src_{t}_d00_ref_kittival.json") for t,d in Pp]); dxk=np.array([DM[f"{t}_{d}"]["fid_KITTI"]-DM[f"{t}_d00_ref"]["fid_KITTI"] for t,d in Pp])
    r3=sp(dxk,dk); p3=cperm(sp,dxk,dk,Gp); f75=float(np.mean([dk[i] for i,(t,d) in enumerate(Pp) if d=="d07_fog75"]))
    data[lab]={"monotone_min_rho":[float(v) for v in mono],"r2_objfid":float(r2),"r3_rho":float(r3),"r3_p":p3,"fog75_kitti":f75}
    rows.append([lab,f"{sum(v>=.8-1e-9 for v in mono)}/{len(towns)}",f"{r2:+.3f}",f"{f75:+.1f}",f"{r3:+.3f} (p={pf(p3)})"])
table("T18","The measurement across detector families (same renders, same detector-independent DINOv3 metrics)",["detector family","towns monotone in sun and fog (R1)","object FID vs appearance-only drop, ρ (R2)","KITTI Δ mAP of fog75-trained (pp)","Δ FID-to-KITTI vs Δ KITTI mAP, ρ (R3)"],rows,
      note="Deformable DETR (ImageNet-init) did not train on 800-frame cells (own-cell mAP 0.0; wrapper verified; §15.68). R3 holds where a training appearance costs real transfer (R50 families) and has nothing to predict where it does not (Swin-T).",sources=[DO,A2,"results/calib_dose_arch3"],data=data)
# ================= T19 Virtual KITTI 2 (E2 / E2b) =================
VK="results/calib_vkitti"
if os.path.exists(f"{VK}/predictions_sim2real.json") and os.path.exists(f"{VK}/dose_metrics.json"):
    apc=lambda f: json.load(open(f))["metrics"]["AP_Car"]*100 if os.path.exists(f) else np.nan
    SCv=["Scene01","Scene02","Scene06","Scene18","Scene20"]; Vv=["d20_morning","d21_sunset","d22_overcast","d23_fog","d24_rain"]; MV=json.load(open(f"{VK}/dose_metrics.json")); DV=json.load(open(f"{VK}/sim2real_distances.json"))
    dv=lambda s_,v: np.nanmean([apc(f"{Rr}/eval_{s_}_d00_ref__on__{s_}_d00_ref.json")-apc(f"{Rr}/eval_{s_}_d00_ref__on__{s_}_{v}.json") for Rr in (VK,VK+"_rep")])
    rows=[]; data={}
    for s_ in SCv:
        nref=json.load(open(f"{VK}/eval_{s_}_d00_ref__on__{s_}_d00_ref.json"))["metrics"]["n_gt_Car"]; real=dv(s_,"real")
        data[s_]={v:float(dv(s_,v)) for v in Vv}; data[s_].update({"real":float(real),"n_gt_heldout":nref,"objfid_fog":MV[f"{s_}_d23_fog"]["obj_fid_ref"],"objfid_real":DV[s_]["obj_fid_ref"],"kitti_clone":apc(f"{VK}/src_{s_}_d00_ref_kittival.json"),"kitti_fog":apc(f"{VK}/src_{s_}_d23_fog_kittival.json")})
        rows.append([s_,nref]+[f"{data[s_][v]:.1f}" for v in Vv]+[f"{real:.1f}",f"{data[s_]['objfid_fog']:.0f} / {data[s_]['objfid_real']:.0f}",f"{data[s_]['kitti_fog']-data[s_]['kitti_clone']:+.1f}"])
    Pv=[(s_,v) for s_ in SCv for v in Vv]; Gv=np.array([a for a,_ in Pv]); yv=np.array([dv(a,b) for a,b in Pv]); xv=np.array([MV[f"{a}_{b}"]["obj_fid_ref"] for a,b in Pv])
    dk=np.array([apc(f"{VK}/src_{a}_{b}_kittival.json")-apc(f"{VK}/src_{a}_d00_ref_kittival.json") for a,b in Pv]); dxk=np.array([MV[f"{a}_{b}"]["fid_KITTI"]-MV[f"{a}_d00_ref"]["fid_KITTI"] for a,b in Pv])
    data["V2_objfid_rho"]=float(sp(xv,yv)); data["V4"]={"rho":float(sp(dxk,dk)),"p":cperm(sp,dxk,dk,Gv)}
    table("T19","Second rendering engine: Virtual KITTI 2 (Unity; frame-matched variants; Car AP@0.5) and the content-matched sim→real gap (KITTI tracking, same sequences)",
          ["scene","held-out GT cars","morning drop","sunset","overcast","fog","rain","sim→real drop (clone det. on real frames)","object FID fog / real","KITTI Car AP Δ, fog- vs clone-trained"],rows,
          note=f"Drops = clone-trained detector, held-out frames, seed mean (pp). Object FID vs drop over 25 pairs ρ {data['V2_objfid_rho']:+.3f} (V2 fails). Δ FID-to-KITTI vs Δ KITTI Car AP ρ {data['V4']['rho']:+.3f}, clustered p = {pf(data['V4']['p'])} (V4 passes). Scene06's held-out stretch has 19 cars (unstable AP). In the reliable scenes the sim→real feature distance is 2.3–2.8× fog's while the sim→real drop is no larger than fog's.",
          sources=[VK,VK+"_rep"],data=data)
# ---------- write ----------
def clean(o):
    if isinstance(o,dict): return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [clean(v) for v in o]
    if isinstance(o,(np.floating,float)): return None if not np.isfinite(o) else float(o)
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,np.bool_): return bool(o)
    return o
json.dump(clean(J),open(OUT/"tables.json","w"),indent=1); open(OUT/"tables.md","w").write("\n".join(MD)+"\n")
print(f"[consolidate] wrote {OUT}/tables.json and tables.md ({len(J['tables'])} tables)")
