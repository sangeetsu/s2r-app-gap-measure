#!/usr/bin/env python3
"""Score ledger §15.77: does a directed asymmetry index predict which transfer direction is harder? (72 pairs, seed-mean drops)"""
import json, numpy as np
from scipy import stats
T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]; D=["d01_sun25","d02_sun10","d03_sun3","d04_night","d05_fog25","d06_fog50","d07_fog75","d08_wet","d09_rain"]
m=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100; Rv=json.load(open("results/calib_dose/dose_reverse_metrics.json")); F=json.load(open("results/calib_dose/dose_metrics.json"))
P=[(t,d) for t in T for d in D]; G=np.array([t for t,_ in P])
def dd(R,t,d): c,r=f"{t}_{d}",f"{t}_d00_ref"; return (m(f"{R}/eval_{r}__on__{r}.json")-m(f"{R}/eval_{r}__on__{c}.json"))-(m(f"{R}/eval_{c}__on__{c}.json")-m(f"{R}/eval_{c}__on__{r}.json"))
y=np.array([np.mean([dd(R,t,d) for R in ("results/calib_dose","results/calib_dose_rep")]) for t,d in P])
rep=max(max(abs(Rv[f"{t}_{d}"]["nnd_dose_to_ref"]-F[f"{t}_{d}"]["nnd_dose_to_ref"]),abs(Rv[f"{t}_{d}"]["cov_dose_by_ref"]-F[f"{t}_{d}"]["cov_dose_by_ref"])) for t,d in P)
print(f"reproduction of forward metrics: max |diff| {rep:.2e}; clean->dose harder (seed mean) in {(y>0).sum()}/72")
def test(x,lab,gate):
    rho=stats.spearmanr(x,y)[0]; rng=np.random.default_rng(0); null=[]
    for _ in range(2000):
        xp=x.copy()
        for t in T: i=np.where(G==t)[0]; xp[i]=rng.permutation(x[i])
        null.append(stats.spearmanr(xp,y)[0])
    p=(1+np.sum(np.array(null)>=rho))/2001; agree=int((np.sign(x)==np.sign(y)).sum()); pb=stats.binomtest(agree,len(y),.5,alternative="greater").pvalue
    print(f"  {lab:<44} Spearman {rho:+.3f} (one-sided clustered p={p:.4f})  sign agreement {agree}/72 (binomial p={pb:.4f})"+(f"  -> A1 {'PASS' if rho>0 and p<.05 else 'FAIL'}, A2 {'PASS' if pb<.05 else 'FAIL'}" if gate else ""))
    for fam,ds in (("fog",D[4:7]),("sun+night",D[:4]),("wet+rain",D[7:])):
        i=np.array([d in ds for _,d in P]); print(f"      {fam:<10} rho {stats.spearmanr(x[i],y[i])[0]:+.2f}  agree {int((np.sign(x[i])==np.sign(y[i])).sum())}/{i.sum()}")
test(np.array([Rv[f"{t}_{d}"]["cov_ref_by_dose"]-Rv[f"{t}_{d}"]["cov_dose_by_ref"] for t,d in P]),"Δcov = cov(ref by dose) − cov(dose by ref)",True)
test(np.array([Rv[f"{t}_{d}"]["nnd_dose_to_ref"]-Rv[f"{t}_{d}"]["nnd_ref_to_dose"] for t,d in P]),"Δnnd = nnd(dose→ref) − nnd(ref→dose)",False)
