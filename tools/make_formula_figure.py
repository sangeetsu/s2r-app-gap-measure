#!/usr/bin/env python3
"""F10: pre-registered round-3 test (§15.87/15.89): straight line vs the saturating formula drop = A_town (1 − exp(−objFID/τ)); plus the calibration data and curves."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":7,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
H3="results/calib_dose_ho3"; P=json.load(open(f"{H3}/predictions.json")); m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]; TC={t:plt.cm.tab10(i) for i,t in enumerate(T)}
cells=sorted(P["cells"]); o=np.array([np.mean([m(f"results/calib_dose/eval_{P['cells'][c]['town']}_d00_ref__on__{P['cells'][c]['town']}_d00_ref.json")-m(f"{H3}/eval_{P['cells'][c]['town']}_d00_ref__on__{c}.json"),
       m(f"results/calib_dose_rep/eval_{P['cells'][c]['town']}_d00_ref__on__{P['cells'][c]['town']}_d00_ref.json")-m(f"{H3}_rep/eval_{P['cells'][c]['town']}_d00_ref__on__{c}.json")]) for c in cells])
fig,axs=plt.subplots(1,3,figsize=(7.0,2.45),gridspec_kw={"wspace":.42})
ax=axs[0]; A=P["M2"]["ceiling_A"]; tau=P["M2"]["tau"]; xs=np.linspace(0,320,200)
for t in T: ax.plot(xs,A[t]*(1-np.exp(-xs/tau)),color=TC[t],lw=.8)
ax.plot(xs,P["M0"]["intercept"]+P["M0"]["slope"]*xs,"k--",lw=1,label="straight line (M0)")
for c,y in zip(cells,o): ax.scatter(P["cells"][c]["x"],y,s=8,color=TC[P["cells"][c]["town"]],edgecolor="k",linewidths=.2,zorder=3)
ax.set_xlabel("appearance distance $d_\\mathrm{app}$\n(object FID, reference vs dose render)"); ax.set_ylabel("appearance-only drop (pp)"); ax.set_title(f"(a) formula curves per town (τ = {tau:.0f})\ndots = 72 new pairs (not used to fit)",fontsize=6.8); ax.legend(frameon=False,fontsize=6,loc="upper left")
for ax,k,lab in ((axs[1],"M0","(b) straight line"),(axs[2],"M2","(c) calibration formula")):
    pr=np.array([P["cells"][c][k] for c in cells]); e=np.abs(pr-o)
    for c,p_,y in zip(cells,pr,o): ax.scatter(p_,y,s=8,color=TC[P["cells"][c]["town"]],edgecolor="k",linewidths=.2)
    lim=[-3,45]; ax.plot(lim,lim,"k:",lw=.6); ax.fill_between(lim,[l-4 for l in lim],[l+4 for l in lim],color="gray",alpha=.12,lw=0); ax.set_xlim(lim); ax.set_ylim(lim)
    from scipy import stats
    ax.set_title(f"{lab}: MAE {e.mean():.2f} pp, ρ {stats.spearmanr(pr,o)[0]:.2f}\n{100*(e<=4).mean():.0f}% within ±4 pp (band)",fontsize=6.8); ax.set_xlabel("predicted drop (pp), frozen"); ax.set_ylabel("observed drop (pp)")
fig.savefig("figures/cvpr/F10_formula_round3.pdf",bbox_inches="tight"); fig.savefig("figures/cvpr/F10_formula_round3.png",bbox_inches="tight",dpi=150); print("[fig] F10")
