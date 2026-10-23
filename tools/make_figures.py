#!/usr/bin/env python3
"""Paper figures F1–F6 from results/ (plan_2026-10-16 §4, Oct 20 item). Reads results/consolidated/tables.json for the annotated statistics and the
raw eval/src/metric JSONs for the points. Writes figures/cvpr/F*.pdf and .png (png for hand-checking). No generated artwork: every mark is data."""
import json, os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
plt.rcParams.update({"font.size":7,"axes.titlesize":7.5,"axes.labelsize":7,"legend.fontsize":6,"xtick.labelsize":6.5,"ytick.labelsize":6.5,"pdf.fonttype":42,"ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
OUT="figures/cvpr"; os.makedirs(OUT,exist_ok=True); TJ=json.load(open("results/consolidated/tables.json"))["tables"]
DO,DOR="results/calib_dose","results/calib_dose_rep"; IMG="/mnt/features/carla_doses"
DOSES=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]; NICE={"d00_ref":"reference (clear noon)","d01_sun25":"sun 25°","d02_sun10":"sun 10°","d03_sun3":"sun 3°","d04_night":"night","d05_fog25":"fog 25","d06_fog50":"fog 50","d07_fog75":"fog 75","d08_wet":"wet","d09_rain":"rain"}
FAMC={"sun":"#1f77b4","fog":"#d62728","other":"#2ca02c"}; FAMM={"sun":"o","fog":"^","other":"s"}
def fam(d): return "sun" if d in DOSES[:4] else ("fog" if d in DOSES[4:7] else "other")
def m(f): d=json.load(open(f)); return d.get("metrics",d)["mAP@IoU=0.5"]*100
T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]; PAIRS=[(t,d) for t in T for d in DOSES]; TC={t:plt.cm.tab10(i) for i,t in enumerate(T)}
def drop(R,t,d): return m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{R}/eval_{t}_d00_ref__on__{t}_{d}.json")
D42={p:drop(DO,*p) for p in PAIRS}; D17={p:drop(DOR,*p) for p in PAIRS}; DM={p:(D42[p]+D17[p])/2 for p in PAIRS}
MET=json.load(open(f"{DO}/dose_metrics.json")); LL=json.load(open(f"{DO}/dose_lowlevel.json"))
def dreal(R,t,d,key): return m(f"{R}/src_{t}_{d}_{key}.json")-m(f"{R}/src_{t}_d00_ref_{key}.json")
rng=np.random.default_rng(0)
def bci(v): b=[np.mean(rng.choice(v,len(v))) for _ in range(2000)]; return np.percentile(b,[2.5,97.5])
def save(fig,name): fig.savefig(f"{OUT}/{name}.pdf",bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png",bbox_inches="tight",dpi=150); plt.close(fig); print("[fig]",name)
# ---------------- F1 frame-matched panel ----------------
def f1(town=os.environ.get("F1_TOWN","Town03"),frame=int(os.environ.get("F1_FRAME","852"))):
    L=[l.split() for l in open(f"{IMG}/{town}_d00_ref/frame_{frame:06d}.txt") if l.strip()]
    boxes=[(l[0],*map(float,l[4:8])) for l in L if len(l)>=8]
    xs=[b[1] for b in boxes]+[b[3] for b in boxes]; cx=(min(xs)+max(xs))/2 if xs else 621; W_=760; x0=int(np.clip(cx-W_/2,0,1242-W_))
    panels=["d00_ref","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
    fig,axs=plt.subplots(3,3,figsize=(6.9,3.25)); R9=TJ["T9"]["data"]
    for ax,d in zip(axs.ravel(),panels):
        im=Image.open(f"{IMG}/{town}_{d}/frame_{frame:06d}.jpg").crop((x0,0,x0+W_,375)); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        for c,x1,y1,x2,y2 in boxes: ax.add_patch(plt.Rectangle((x1-x0,y1),x2-x1,y2-y1,fill=False,lw=.5,ec={"Car":"#ffd700","Pedestrian":"#00e5ff","Cyclist":"#ff40ff"}.get(c,"w")))
        cost="" if d=="d00_ref" else f"   (−{(R9['seed42'][f'{town}|{d}']+R9['seed17'][f'{town}|{d}'])/2:.1f} pp)"
        ax.set_title(NICE[d]+cost,fontsize=7,pad=2)
    fig.suptitle(f"{town}, held-out frame {frame}: identical actors, poses and labels ({len(boxes)} boxes: Car yellow, Pedestrian cyan, Cyclist magenta) re-rendered under each dose.\nIn brackets: mAP drop of the town's clean-trained detector on its held-out frames under that dose (mean of 2 seeds).",fontsize=7,y=1.04)
    fig.subplots_adjust(wspace=.02,hspace=.18); save(fig,"F1_frame_matched")
# ---------------- F2 dose response ----------------
def f2():
    fig,axs=plt.subplots(1,3,figsize=(6.9,2.0),sharey=True,gridspec_kw={"width_ratios":[4,3,2]})
    for ax,(name,ds) in zip(axs,[("sun elevation → night",DOSES[:4]),("fog density",DOSES[4:7]),("precipitation",DOSES[7:])]):
        x=np.arange(len(ds))
        for t in T: ax.plot(x,[DM[(t,d)] for d in ds],"-",color=TC[t],lw=.7,alpha=.75,marker="o",ms=2,label=t)
        mu=np.array([np.mean([DM[(t,d)] for t in T]) for d in ds]); ci=np.array([bci(np.array([DM[(t,d)] for t in T])) for d in ds])
        ax.errorbar(x,mu,yerr=[mu-ci[:,0],ci[:,1]-mu],color="k",lw=1.6,capsize=2,marker="s",ms=3,label="town mean [95% CI]",zorder=5)
        ax.set_xticks(x); ax.set_xticklabels([NICE[d] for d in ds]); ax.set_title(name); ax.axhline(0,color="gray",lw=.5,ls=":")
    axs[0].set_ylabel("appearance-only mAP drop (pp)\nclean-trained detector, same frames"); axs[0].legend(ncol=3,frameon=False,loc="upper left",handlelength=1.2,columnspacing=.8)
    fig.subplots_adjust(wspace=.08); save(fig,"F2_dose_response")
# ---------------- F3 metric vs drop ----------------
def f3():
    T10=TJ["T10"]["data"]; spec=[("object-level, frozen DINOv3 (GT boxes)|obj_fid_ref",MET,"obj_fid_ref","object FID (DINOv3, boxes)"),("object-level, frozen DINOv3 (GT boxes)|nnd_dose_to_ref",MET,"nnd_dose_to_ref","directed NN distance (DINOv3, boxes)"),("object-level, frozen DINOv3 (GT boxes)|cov_dose_by_ref",MET,"cov_dose_by_ref","coverage (DINOv3, boxes)"),
          ("low-level / classic (no boxes)|lpips",LL,"lpips","LPIPS (paired, no boxes)"),("low-level / classic (no boxes)|incfid_ref",LL,"incfid_ref","Inception FID (no boxes)"),("image-level, frozen DINOv3 (no boxes)|img_fid_ref",MET,"img_fid_ref","image FID (DINOv3, no boxes)")]
    fig,axs=plt.subplots(2,3,figsize=(6.9,4.0),sharey=True)
    for ax,(key,src,k,title) in zip(axs.ravel(),spec):
        for t,d in PAIRS: ax.scatter(src[f"{t}_{d}"][k],D42[(t,d)],s=11,color=TC[t],marker=FAMM[fam(d)],edgecolor="k",linewidths=.25,alpha=.9)
        s=T10[key]; ax.set_title(title); neg=s['rho']<0; ax.set_ylim(top=44)
        ax.text(.97 if neg else .03,.97,f"ρ = {s['rho']:+.2f} [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]\nseed 17: {s['rho_b']:+.2f}; LOTO R² {s['loto_r2']:.2f}",transform=ax.transAxes,va="top",ha="right" if neg else "left",fontsize=6)
        if ax is axs[1,1]: ax.set_xlabel("metric between the reference render and the dose render of the same frames")
    axs[0,0].set_ylabel("appearance-only mAP drop (pp)"); axs[1,0].set_ylabel("appearance-only mAP drop (pp)")
    h=[Line2D([],[],marker="o",ls="",color=TC[t],label=t,ms=4) for t in T]+[Line2D([],[],marker=FAMM[f],ls="",color="gray",label=f"{f} doses",ms=4) for f in ("sun","fog","other")]
    fig.legend(handles=h,ncol=11,frameon=False,loc="lower center",bbox_to_anchor=(.5,-.03),handletextpad=.2,columnspacing=.8); fig.subplots_adjust(wspace=.08,hspace=.45); save(fig,"F3_metric_vs_drop")
# ---------------- F4 Δ distance-to-real vs Δ real mAP ----------------
def f4():
    fig,axs=plt.subplots(1,2,figsize=(6.9,2.5)); T12=TJ["T12b"]["data"]
    for ax,(Tn,key) in zip(axs,[("KITTI","kittival"),("Cityscapes","cityscapesval")]):
        ax.axhspan(-1.6,1.6,color="gray",alpha=.12,lw=0); ax.axhline(0,color="gray",lw=.5); ax.axvline(0,color="gray",lw=.5)
        for t,d in PAIRS:
            dx=MET[f"{t}_{d}"][f"fid_{Tn}"]-MET[f"{t}_d00_ref"][f"fid_{Tn}"]; dy=(dreal(DO,t,d,key)+dreal(DOR,t,d,key))/2
            ax.scatter(dx,dy,s=12,color=FAMC[fam(d)],marker=FAMM[fam(d)],edgecolor="k",linewidths=.25,alpha=.85)
        s=T12[f"{Tn}|DINOv3 image FID to real|seed mean"]; ax.set_title(f"real target: {Tn}")
        ax.text(.97,.97,f"all doses ρ = {s['all']['rho']:+.2f} [{s['all']['ci'][0]:+.2f}, {s['all']['ci'][1]:+.2f}], p {'< .001' if s['all']['p']<.001 else '= %.3f'%s['all']['p']}\nfog only ρ = {s['fog only']['rho']:+.2f}; without fog ρ = {s['without fog']['rho']:+.2f} (p = {s['without fog']['p']:.2f})",transform=ax.transAxes,ha="right",va="top",fontsize=6)
        ax.set_xlabel(f"Δ FID (DINOv3) to {Tn}, dose − ref cell"); ax.set_ylabel("Δ real mAP (pp), dose- minus\nreference-trained (mean of 2 seeds)"); lo,hi=ax.get_ylim(); ax.set_ylim(lo,hi+(hi-lo)*.25)
    h=[Line2D([],[],marker=FAMM[f],ls="",color=FAMC[f],label={"sun":"sun angle / night","fog":"fog 25/50/75","other":"wet / rain"}[f],ms=4) for f in ("sun","fog","other")]+[plt.Rectangle((0,0),1,1,color="gray",alpha=.25,label="±1 seed sd")]
    axs[0].legend(handles=h,ncol=1,frameon=False,loc="lower left",handletextpad=.3); fig.subplots_adjust(wspace=.32); save(fig,"F4_dfid_vs_dreal")
# ---------------- F5 decomposition ----------------
def f5():
    T15=TJ["T15"]["data"]; T7=TJ["T7"]["data"]["lowq"]; R3=TJ["T12a"]["data"]
    fig,axs=plt.subplots(1,2,figsize=(6.9,2.1),sharey=True)
    for ax,Tn in zip(axs,["KITTI","Cityscapes"]):
        d=T15[Tn]; items=[("pretraining: COCO instead of ImageNet init\n(same 24 cells)",d["pretraining_pp"],None,"#4c4c4c"),("content: best instead of worst town\n(spread of town means, 24-cell grid)",d["town_range_pp"],None,"#1f77b4"),("content: twice the labeled objects\n(within-cell learning curve, causal)",d["pp_per_doubling"],None,"#1f77b4"),
               ("appearance: night instead of clear noon\n(frame-matched, same content)",d["night_pp"],R3[Tn]["d04_night"]["ci"],"#d62728"),("appearance: Low instead of Epic rendering\n(5 towns, p = %.2f KITTI, %.2f Cityscapes; ± sd)"%(T7["KITTI"]["p"],T7["Cityscapes"]["p"]),-T7[Tn]["epic_minus_low"],[-T7[Tn]["epic_minus_low"]-T7[Tn]["sd"],-T7[Tn]["epic_minus_low"]+T7[Tn]["sd"]],"#d62728"),("appearance: heavy fog instead of clear noon\n(frame-matched, fog75, same content)",d["fog75_pp"],R3[Tn]["d07_fog75"]["ci"],"#d62728")]
        y=np.arange(len(items))[::-1]
        for yi,(lab,v,ci,c) in zip(y,items):
            ax.barh(yi,v,color=c,alpha=.85,height=.65)
            if ci: ax.errorbar(v,yi,xerr=[[v-ci[0]],[ci[1]-v]],color="k",lw=.8,capsize=2)
            if v>=0: ax.text((ci[1] if ci else v)+.5,yi,f"{v:+.1f}",va="center",ha="left",fontsize=6)
            else: ax.text((ci[0] if ci else v)-.5,yi,f"{v:+.1f}",va="center",ha="right",fontsize=6)
        ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items]); ax.set_title(f"{Tn}: Δ real mAP (pp)"); ax.axvline(0,color="k",lw=.5); ax.set_xlim(-16,max(24,d["pretraining_pp"]+4))
    fig.subplots_adjust(wspace=.08); save(fig,"F5_decomposition")
# ---------------- F6 asymmetry ----------------
def f6():
    fig,ax=plt.subplots(figsize=(6.9,2.2)); x=np.arange(len(DOSES)); w=.38
    a1={};a2={}
    for R,S in [(DO,{}),(DOR,{})]:
        for t in T:
            for d in DOSES:
                c,ref=f"{t}_{d}",f"{t}_d00_ref"; S[(t,d)]=(m(f"{R}/eval_{ref}__on__{ref}.json")-m(f"{R}/eval_{ref}__on__{c}.json"),m(f"{R}/eval_{c}__on__{c}.json")-m(f"{R}/eval_{c}__on__{ref}.json"))
        (a1 if R==DO else a2).update(S)
    A={p:((a1[p][0]+a2[p][0])/2,(a1[p][1]+a2[p][1])/2) for p in PAIRS}
    m1=[np.mean([A[(t,d)][0] for t in T]) for d in DOSES]; m2=[np.mean([A[(t,d)][1] for t in T]) for d in DOSES]
    ax.bar(x-w/2,m1,w,color="#d62728",alpha=.8,label="clean-trained detector on dose frames"); ax.bar(x+w/2,m2,w,color="#1f77b4",alpha=.8,label="dose-trained detector on clean frames")
    for i,d in enumerate(DOSES):
        ax.scatter(x[i]-w/2+rng.uniform(-.1,.1,8),[A[(t,d)][0] for t in T],s=6,color="k",zorder=3,linewidths=0); ax.scatter(x[i]+w/2+rng.uniform(-.1,.1,8),[A[(t,d)][1] for t in T],s=6,color="k",zorder=3,linewidths=0)
        k=sum(A[(t,d)][0]>A[(t,d)][1] for t in T); ax.text(x[i],max(max(A[(t,d)][0] for t in T),max(A[(t,d)][1] for t in T))+1.2,f"{k}/8",ha="center",fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([NICE[d] for d in DOSES]); ax.set_ylabel("mAP drop on the other rendering (pp)\n(each detector vs its own held-out frames)"); ax.legend(frameon=False,loc="upper left")
    tot=sum(A[p][0]>A[p][1] for p in PAIRS); ax.set_title(f"clean→degraded is the harder direction in {tot}/72 town×dose pairs (seed-mean drops; labels = towns of 8)")
    save(fig,"F6_asymmetry")
# ---------------- F7 held-out-dose prediction ----------------
def f7():
    HO="results/calib_dose_ho"; P=json.load(open(f"{HO}/predictions.json")); T16=TJ["T16"]["data"]["obj_fid_ref"]
    own={R:{t:m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json") for t in T} for R in (DO,DOR)}
    SUBC={"interpolation":"#1f77b4","extrapolation":"#d62728","compound":"#9467bd","novel":"#2ca02c"}
    fig,ax=plt.subplots(figsize=(3.4,3.2))
    for c,v in P["cells"].items():
        t=v["town"]; o=np.mean([own[DO][t]-m(f"{HO}/eval_{t}_d00_ref__on__{c}.json"),own[DOR][t]-m(f"{HO}_rep/eval_{t}_d00_ref__on__{c}.json")]); q=v["obj_fid_ref"]
        ax.errorbar(q["pred"],o,xerr=[[q["pred"]-q["pi90"][0]],[q["pi90"][1]-q["pred"]]],fmt="o",ms=3,color=SUBC[v["subset"]],ecolor=SUBC[v["subset"]],elinewidth=.3,alpha=.8,mec="k",mew=.2)
    lim=[-5,50]; ax.plot(lim,lim,"k:",lw=.6); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("predicted drop (pp), object FID calibrated\non the 9 original doses, 90% PI"); ax.set_ylabel("observed appearance-only drop (pp)")
    ax.text(.03,.97,f"72 held-out pairs, 8 towns\nρ = {T16['rho']:.2f}; MAE {T16['mae']:.1f} pp\n{100*T16['coverage90']:.0f}% inside 90% PI",transform=ax.transAxes,va="top",fontsize=6)
    ax.legend(handles=[Line2D([],[],marker="o",ls="",color=c,label=k+{"interpolation":" (fog35/65, sun6)","extrapolation":" (fog90)","compound":" (night+rain, night+fog, sun10+wet)","novel":" (cloud90, dust50)"}[k],ms=4) for k,c in SUBC.items()],frameon=False,loc="upper center",bbox_to_anchor=(.45,-.2),fontsize=5,ncol=1)
    save(fig,"F7_heldout_prediction")
if __name__=="__main__":
    for f in (f1,f2,f3,f4,f5,f6,f7): f()
