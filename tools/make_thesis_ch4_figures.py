#!/usr/bin/env python3
"""Thesis Chapter 4 figures F12 (pre-registered FID line + held-out towns) and F13 (within-cell learning curves).

Every plotted value is read from the results repository's results/ folder, or, for training-object counts, counted from the label files with the
same function the ledger's analysis scripts used (tools/analyze_learning_curve.py, tools/prereg_competing.py, tools/analyze_shift.py). Before
plotting, the script checks the values against the stored records (prereg files, learning_curve.txt, calib_shift/analysis.txt, consolidated tables
T5/T7) and stops on any mismatch. Style (rcParams, town colours) is copied from tools/make_figures.py of the results repository.
    python make_thesis_ch4_figures.py [RESULTS_REPO] [OUT_DIR]
Data (paths relative to RESULTS_REPO):
  F12  results/calib_imnet/prereg_fid_line.json (frozen line, 2026-10-14T08:12:25; ledger §15.34)
       results/calib_imnet/prereg_predictions_towns2.json (held-out FID + FID-line predictions, 15:00:30)
       results/calib_imnet/prereg_predictions_towns2_competing.json (training-object counts of held-out cells, sha256 5c41685d…; §15.36)
       results/domain_distance_dinov3_all.json (FID of the 24 grid cells), results/calib_imnet/src_*_{kittival,cityscapesval}.json (grid mAP)
       results/calib_imnet_towns2/src_*_{kittival,cityscapesval}.json (held-out mAP; §15.37), results/consolidated/tables.json T7 (scores)
  F13  results/calib_imnet_lc/src_*_n{100,200,400}_*.json + results/calib_imnet/src_*ClearNoon_*.json (n800) (§15.39; T5)
       results/calib_shift/src_SHIFT_*_{n100_,n200_,}*.json (§15.48-15.49; T5); between-source points = 24 grid + 6 held-out cells (§15.38)
"""
import json, os, sys, glob, hashlib, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
from scipy import stats

REPO=sys.argv[1] if len(sys.argv)>1 else "."
OUT=os.path.abspath(sys.argv[2] if len(sys.argv)>2 else "figures/thesis")
os.makedirs(OUT,exist_ok=True); os.chdir(REPO)
# ---- style: identical rcParams and town colours to tools/make_figures.py ----
plt.rcParams.update({"font.size":7,"axes.titlesize":7.5,"axes.labelsize":7,"legend.fontsize":6,"xtick.labelsize":6.5,"ytick.labelsize":6.5,"pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
T8=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]; TC={t:plt.cm.tab10(i) for i,t in enumerate(T8)}; TC["Town15"]=plt.cm.tab10(8)
G,H,L,S="results/calib_imnet","results/calib_imnet_towns2","results/calib_imnet_lc","results/calib_shift"
TJ=json.load(open("results/consolidated/tables.json"))["tables"]
TARGETS=[("KITTI","kittival"),("Cityscapes","cityscapesval")]
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
def save(fig,name): fig.savefig(f"{OUT}/{name}.pdf",bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png",bbox_inches="tight",dpi=200); plt.close(fig); print("[fig]",f"{OUT}/{name}.pdf")
def close(a,b,tol,what):
    if abs(a-b)>tol: raise SystemExit(f"MISMATCH {what}: {a} vs {b}")

# ---- training-object counts (same definitions as the ledger's analysis scripts) ----
def croot(c):
    for r in (f"{Path.home()}/datasets/carla_extracted","/mnt/features/carla_new","/mnt/features/carla_new2","/mnt/features/carla_new3"):
        if os.path.isdir(f"{r}/{c}"): return r
    raise SystemExit(f"no label directory for {c}")
def nbox_carla(c,k): return sum(1 for f in sorted(glob.glob(f"{croot(c)}/{c}/frame_*.txt"))[:k] for l in open(f) if l.split() and l.split()[0] in ("Car","Pedestrian","Cyclist"))
SHROOT=Path("/mnt/features/shift_cells")
def nbox_shift(c,n): return sum(1 for k in range(n) for l in open(SHROOT/c/f"frame_{k:06d}.txt") if l.strip())

# ---- shared data: 24 grid cells + 6 held-out cells ----
LINE=json.load(open(f"{G}/prereg_fid_line.json")); GRID=LINE["fit_on"]; assert len(GRID)==24
PRED=json.load(open(f"{G}/prereg_predictions_towns2.json")); HELD=list(PRED["cells"]); assert len(HELD)==6
CF=f"{G}/prereg_predictions_towns2_competing.json"; COMP=json.load(open(CF))
assert hashlib.sha256(open(CF,"rb").read()).hexdigest().startswith("5c41685d"), "competing-model file hash differs from ledger §15.36"
DM=json.load(open("results/domain_distance_dinov3_all.json"))["metrics"]["image"]
def fid(s,T): return (DM.get(f"{s}||{T}") or DM.get(f"{T}||{s}"))["fid"]
NB={c:nbox_carla(c,800) for c in GRID+HELD}
for c in HELD: assert NB[c]==COMP["cells"][c]["n_boxes"], c          # counts reproduce the pre-registered file
assert (min(NB[c] for c in GRID),max(NB[c] for c in GRID))==(1028,7702)  # ledger §15.36 range
def real(c,key): return m(f"{G if c in GRID else H}/src_{c}_{key}.json")
TOWN=lambda c: c.split("_")[0]
SHORT=lambda c: c.split("_")[0]+" "+("noon" if c.endswith("ClearNoon") else "night")

# ================= F12: pre-registered FID line and the held-out towns =================
# label text positions in data coordinates (FID, mAP) chosen by eye to avoid overlaps: (x, y, ha, va, leader line?)
LABPOS={"KITTI":{"Town06_ClearNight":(14.62,20.56,"right","center",False),"Town06_ClearNoon":(16.55,29.8,"center","bottom",True),
                 "Town07_ClearNight":(15.0,30.6,"center","bottom",True),"Town07_ClearNoon":(16.36,19.31,"left","center",False),
                 "Town15_ClearNight":(15.68,1.81,"left","center",False),"Town15_ClearNoon":(11.85,10.4,"left","top",True)},
        "Cityscapes":{"Town06_ClearNight":(14.54,8.6,"right","center",False),"Town06_ClearNoon":(15.75,16.3,"center","bottom",True),
                      "Town07_ClearNight":(17.1,16.65,"left","center",False),"Town07_ClearNoon":(17.6,8.9,"right","top",True),
                      "Town15_ClearNight":(15.6,2.07,"left","center",False),"Town15_ClearNoon":(10.15,4.6,"left","top",True)}}
def f12():
    fig,axs=plt.subplots(1,2,figsize=(6.9,2.7)); T7=TJ["T7"]["data"]
    for ax,(T,key) in zip(axs,TARGETS):
        ln=LINE["lines"][T]; a,b,sd=ln["intercept"],ln["slope"],ln["resid_sd"]
        xg=np.array([fid(s,T) for s in GRID]); yg=np.array([real(s,key) for s in GRID])
        # checks: the frozen line is the OLS fit of these 24 points; predictions in the prediction file lie on it; scores equal T7
        lr=stats.linregress(xg,yg); close(lr.intercept,a,1e-6,f"{T} intercept"); close(lr.slope,b,1e-6,f"{T} slope"); close(lr.rvalue,ln["r"],1e-6,f"{T} r")
        xh=np.array([PRED["cells"][c][T]["fid"] for c in HELD]); ph=np.array([PRED["cells"][c][T]["pred_mAP"] for c in HELD]); yh=np.array([m(f"{H}/src_{c}_{key}.json") for c in HELD])
        for x_,p_ in zip(xh,ph): close(a+b*x_,p_,1e-6,f"{T} prediction on line")
        s7=T7[T]["FID line (pre-registered 08:12)"]
        close(np.abs(yh-ph).mean(),s7["mae"],1e-9,f"{T} MAE"); close(stats.pearsonr(ph,yh)[0],s7["r"],1e-9,f"{T} r(pred,obs)")
        # line: solid over the fitted FID range, dashed where it extrapolates; band = pre-registered +-1 residual sd
        lo,hi=min(xg.min(),xh.min())-.4,max(xg.max(),xh.max())+.4; f0,f1=ln["fid_range_fit"]
        xx=np.linspace(lo,hi,200); ax.fill_between(xx,a+b*xx-sd,a+b*xx+sd,color="gray",alpha=.15,lw=0)
        ax.plot([f0,f1],[a+b*f0,a+b*f1],"k-",lw=1.0,zorder=2); ax.plot([lo,f0],[a+b*lo,a+b*f0],"k--",lw=.8,zorder=2); ax.plot([f1,hi],[a+b*f1,a+b*hi],"k--",lw=.8,zorder=2)
        for s,x_,y_ in zip(GRID,xg,yg): ax.scatter(x_,y_,s=13,color=TC[TOWN(s)],edgecolor="k",linewidths=.25,zorder=3)
        for c,x_,p_,y_ in zip(HELD,xh,ph,yh):
            ax.plot([x_,x_],[p_,y_],color=TC[TOWN(c)],lw=.7,ls=":",zorder=3); ax.scatter(x_,p_,s=12,facecolor="white",edgecolor=TC[TOWN(c)],linewidths=.8,zorder=4)
            ax.scatter(x_,y_,s=30,marker="D",color=TC[TOWN(c)],edgecolor="k",linewidths=.5,zorder=5)
        for c,x_,y_ in zip(HELD,xh,yh):   # label placement only (text position, alignment, leader line); no data values here
            tx,ty,ha,va,lead=LABPOS[T][c]; txt=f"Town15_ClearNoon: lowest held-out FID,\n{NB[c]} labeled objects, detector failed" if c=="Town15_ClearNoon" else SHORT(c)
            ax.annotate(txt,(x_,y_),xytext=(tx,ty),fontsize=5.5,ha=ha,va=va,zorder=6,arrowprops=dict(arrowstyle="-",lw=.4,color="k",shrinkA=0,shrinkB=2.5) if lead else None)
        ax.text(.98,.98,f"pre-registered: mAP = {a:.2f} − {abs(b):.4f}·FID\n(24 grid sources, r = {ln['r']:+.3f}; band ±1 resid. sd {sd:.2f})\nheld-out: r(pred, obs) = {s7['r']:+.3f}, MAE {s7['mae']:.2f} pp",transform=ax.transAxes,ha="right",va="top",fontsize=5.8)
        ax.set_xlim(lo,hi); ax.set_ylim(min(-1,yh.min()-2),max(yg.max(),ph.max())+(11 if T=="KITTI" else 5.5))
        ax.set_xlabel(f"image FID to {T} (frozen DINOv3, no labels)"); ax.set_ylabel(f"real mAP@0.5 on {T} (%)"); ax.set_title(f"real target: {T}")
    h=[Line2D([],[],marker="o",ls="",color=TC[t],mec="k",mew=.25,label=t,ms=4) for t in ["Town01","Town02","Town03","Town04","Town05","Town10HD"]]
    h+=[Line2D([],[],marker="D",ls="",color=TC[t],mec="k",mew=.5,label=f"{t} (held out)",ms=4) for t in ["Town06","Town07","Town15"]]
    h+=[Line2D([],[],marker="o",ls="",mfc="white",mec="gray",label="pre-registered prediction",ms=4),Line2D([],[],color="k",lw=1,label="FID line (dashed: outside fitted range)"),plt.Rectangle((0,0),1,1,color="gray",alpha=.25,label="±1 residual sd (pre-registered)")]
    fig.legend(handles=h,ncol=6,frameon=False,loc="lower center",bbox_to_anchor=(.5,-.13),handletextpad=.3,columnspacing=.9)
    fig.subplots_adjust(wspace=.25); save(fig,"F12_prereg_fid_line")

# ================= F13: within-cell learning curves vs the between-source relation =================
LC=["Town01_ClearNoon","Town02_ClearNoon","Town03_ClearNoon","Town04_ClearNoon","Town05_ClearNoon","Town10HD_ClearNoon"]
SH=[("SHIFT_daytime_clear","daytime, clear","o"),("SHIFT_night_rainy","night, rainy","s"),("SHIFT_dawn-dusk_foggy","dawn/dusk, foggy","^")]
SHC="#222222"
def lc_txt(path):   # parse "  CELL  n100:  20.7 (232 boxes) ..." lines into {(target,cell,n):(mAP,boxes)}
    out={}; T=None
    for l in open(path):
        if l.startswith("=== "): T=l.split()[1].rstrip(":")
        p=l.split()
        if p and (p[0].startswith("Town") or p[0].startswith("SHIFT")) and "n100:" in p:
            for i,w in enumerate(p):
                if w.startswith("n") and w.endswith(":") and w[1:-1].isdigit(): out[(T,p[0],int(w[1:-1]))]=(float(p[i+1]),int(p[i+2].lstrip("(")))
    return out
def f13():
    T5=TJ["T5"]["data"]; LCT=lc_txt(f"{L}/learning_curve.txt"); SHT=lc_txt(f"{S}/analysis.txt")
    fig,axs=plt.subplots(1,2,figsize=(6.9,2.7))
    for ax,(T,key) in zip(axs,TARGETS):
        # between-source: 30 ImageNet-initialised sources at 800 frames
        allc=GRID+HELD; xa=np.array([NB[c] for c in allc]); ya=np.array([real(c,key) for c in allc]); bb=stats.linregress(np.log(xa),ya)
        close(bb.slope,T5[T]["between_slope_30"],1e-9,f"{T} between-source slope")
        ax.scatter(xa,ya,s=14,facecolor="none",edgecolor="#9a9a9a",linewidths=.6,zorder=1)
        xx=np.array([xa.min()*.9,xa.max()*1.1]); ax.plot(xx,bb.intercept+bb.slope*np.log(xx),color="#9a9a9a",ls="--",lw=.9,zorder=1)
        # CARLA within-cell curves (n800 = the grid cell itself, same 2400 iterations)
        X,Y=[],[]
        for c in LC:
            ns=(100,200,400,800); n_=[nbox_carla(c,k) for k in ns]; y_=[m(f"{G}/src_{c}_{key}.json") if k==800 else m(f"{L}/src_{c}_n{k}_{key}.json") for k in ns]
            for k,nn,yy in zip(ns,n_,y_):
                ty,tb=LCT[(T,c,k)]; close(round(yy,1),ty,1e-9,f"{T} {c} n{k} mAP"); close(nn,tb,1,f"{T} {c} n{k} boxes")   # printed counts are int(exp(log n)): may be 1 low
            ax.plot(n_,y_,"-",marker="o",ms=3,lw=1.0,color=TC[TOWN(c)],mec="k",mew=.25,zorder=3)
            X+=list(np.log(n_)-np.mean(np.log(n_))); Y+=list(np.array(y_)-np.mean(y_))
        close(float(np.dot(X,Y)/np.dot(X,X)),T5[T]["carla_fe_slope"],1e-9,f"{T} CARLA within-cell slope")
        # SHIFT within-cell curves (n400 = the full SHIFT cell)
        X,Y=[],[]
        for c,lab,mk in SH:
            ns=(100,200,400); n_=[nbox_shift(c,k) for k in ns]; y_=[m(f"{S}/src_{c}_n{k}_{key}.json") if k<400 else m(f"{S}/src_{c}_{key}.json") for k in ns]
            for k,nn,yy in zip(ns,n_,y_):
                ty,tb=SHT[(T,c,k)]; close(round(yy,1),ty,1e-9,f"{T} {c} n{k} mAP"); close(nn,tb,1,f"{T} {c} n{k} boxes")
            ax.plot(n_,y_,"--",marker=mk,ms=3.2,lw=.9,color=SHC,mfc="white",mec=SHC,mew=.7,zorder=4)
            X+=list(np.log(n_)-np.mean(np.log(n_))); Y+=list(np.array(y_)-np.mean(y_))
        close(float(np.dot(X,Y)/np.dot(X,X)),T5[T]["shift_slope"],1e-9,f"{T} SHIFT within-cell slope")
        d=T5[T]; ci=d["ci"]
        ax.text(.02,.98,f"slope, pp of real mAP per unit ln N:\nCARLA within-cell {d['carla_fe_slope']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] ({d['pp_per_doubling']:+.1f} per doubling)\nSHIFT within-cell {d['shift_slope']:+.2f}\nbetween-source (30 sources) {d['between_slope_30']:+.2f}",transform=ax.transAxes,ha="left",va="top",fontsize=5.8)
        ax.set_xscale("log"); ax.xaxis.set_major_locator(FixedLocator([200,500,1000,2000,5000])); ax.xaxis.set_minor_locator(NullLocator()); ax.set_xticklabels(["200","500","1,000","2,000","5,000"])
        ax.set_xlim(150,9500); lo,hi=ax.get_ylim(); ax.set_ylim(lo,hi+(hi-lo)*.32)
        ax.set_xlabel("labeled training objects N (log scale)"); ax.set_ylabel(f"real mAP@0.5 on {T} (%)"); ax.set_title(f"real target: {T}")
    h=[Line2D([],[],marker="o",ls="-",color=TC[TOWN(c)],mec="k",mew=.25,ms=3,label=TOWN(c)+" ClearNoon") for c in LC]
    h+=[Line2D([],[],marker=mk,ls="--",color=SHC,mfc="white",mec=SHC,ms=3.2,label="SHIFT "+lab) for _,lab,mk in SH]
    h+=[Line2D([],[],marker="o",ls="--",color="#9a9a9a",mfc="none",ms=3.5,label="30 sources at 800 frames + fit")]
    fig.legend(handles=h,ncol=5,frameon=False,loc="lower center",bbox_to_anchor=(.5,-.16),handletextpad=.3,columnspacing=.9)
    fig.subplots_adjust(wspace=.25); save(fig,"F13_learning_curves")

if __name__=="__main__":
    f12(); f13()
