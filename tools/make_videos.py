#!/usr/bin/env python3
"""Supplementary videos (every pixel is a CARLA render; boxes are the town's seed-42 reference detector, score >= .5).
V1: frame-matched testbed — one held-out drive, clean reference + 8 doses, 3x3, with each dose's measured appearance-only drop (Table T9, seed mean).
V2: fog sweep — the same drive at 9 fog densities (0..90), each panel labelled with measured and object-FID-predicted drop."""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src"))
from s2rgap.detector import load_detector_from_ckpt
FONT=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",15); FB=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",17)
COL={1:(255,215,0),2:(0,229,255),3:(255,64,255)}; PW,PH=621,188
def m(f): return json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
def drop_seedmean(t,d,root_res):
    v=[]
    for R,Rr in root_res:
        try: v.append(m(f"{R}/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{Rr}/eval_{t}_d00_ref__on__{t}_{d}.json"))
        except FileNotFoundError: pass
    return float(np.mean(v)) if v else float("nan")
def panel(img_path,boxes,title,sub):
    im=Image.open(img_path).convert("RGB").resize((PW,PH)); dr=ImageDraw.Draw(im); sx,sy=PW/1242,PH/375
    for (x1,y1,x2,y2),l in boxes: dr.rectangle([x1*sx,y1*sy,x2*sx,y2*sy],outline=COL.get(l,(255,255,255)),width=2)
    bar=Image.new("RGB",(PW,26),(20,20,20)); d2=ImageDraw.Draw(bar); d2.text((6,3),title,font=FB,fill=(255,255,255)); d2.text((PW-6-d2.textlength(sub,font=FONT),5),sub,font=FONT,fill=(255,210,120))
    out=Image.new("RGB",(PW,PH+26)); out.paste(bar,(0,0)); out.paste(im,(0,26)); return out
def run(cells,town,frames,outmp4,header,fps=10):
    dev=torch.device("cuda"); det=load_detector_from_ckpt(f"checkpoints/anchind_{town}_d00_ref/last.pt",dev); tmp=Path(tempfile.mkdtemp(dir="/mnt/features"))
    for fi,k in enumerate(frames):
        tiles=[]
        for root,cell,title,sub in cells:
            p=f"{root}/{cell}/frame_{k:06d}.jpg"; x=torch.from_numpy(np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).to(dev)
            with torch.no_grad(): pr=det([x])[0]
            keep=pr["scores"]>=.5; tiles.append(panel(p,list(zip(pr["boxes"][keep].cpu().numpy().tolist(),pr["labels"][keep].cpu().numpy().tolist())),title,sub))
        hb=58 if "\n" in header else 34
        G=Image.new("RGB",(3*PW+8,3*(PH+26)+8+hb),(0,0,0)); dh=ImageDraw.Draw(G); dh.multiline_text((10,6),header.format(frame=k),font=FB,fill=(255,255,255),spacing=4)
        for i,t_ in enumerate(tiles): G.paste(t_,(4+(i%3)*PW,hb+4+(i//3)*(PH+26)))
        G.save(tmp/f"{fi:04d}.png")
    subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(fps),"-i",str(tmp/"%04d.png"),"-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2","-c:v","libx264","-pix_fmt","yuv420p","-crf","20",outmp4],check=True)
    Image.open(tmp/"0000.png").save(outmp4.replace(".mp4","_frame0.png"))
    for f in tmp.iterdir(): f.unlink()
    tmp.rmdir(); print("[video]",outmp4)
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("which",choices=["v1","v2","v2b"]); p.add_argument("--town",default="Town03"); p.add_argument("--start",type=int,default=800); p.add_argument("--n",type=int,default=200); a=p.parse_args()
    t=a.town; S="/mnt/features/carla_doses"; RR=[("results/calib_dose","results/calib_dose"),("results/calib_dose_rep","results/calib_dose_rep")]
    frames=list(range(a.start,a.start+a.n))
    if a.which=="v1":
        D=[("d00_ref","clear noon (reference)"),("d02_sun10","sun 10°"),("d03_sun3","sun 3°"),("d04_night","night"),("d05_fog25","fog 25"),("d06_fog50","fog 50"),("d07_fog75","fog 75"),("d08_wet","wet road"),("d09_rain","rain")]
        cells=[(S,f"{t}_{d}",title,"" if d=="d00_ref" else f"detector loses {drop_seedmean(t,d,RR):.1f} pp") for d,title in D]
        run(cells,t,frames,f"figures/cvpr/videos/V1_frame_matched_{t}.mp4",f"{t}, held-out drive (frame {{frame}}): identical actors and labels re-rendered under each dose; boxes = clear-trained detector")
    elif a.which=="v2b":
        TT=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]; RR2=[("results/calib_dose","results/calib_dose"),("results/calib_dose_rep","results/calib_dose_rep")]
        fogcost={x:np.mean([drop_seedmean(x,d,RR2) for d in ("d05_fog25","d06_fog50","d07_fog75")]) for x in TT}; mu=np.mean(list(fogcost.values())); t=min(TT,key=lambda x:abs(fogcost[x]-mu))
        print("typical town by rule:",t,{k:round(v,1) for k,v in fogcost.items()},"mean",round(mu,1))
        P3=json.load(open("results/calib_dose_ho3/predictions.json")); tau=P3["M2"]["tau"]; A=P3["M2"]["ceiling_A"][t]
        LV=[(0,S,"d00_ref","results/calib_dose","results/calib_dose_rep","results/calib_dose/dose_metrics.json",False),(5,"/mnt/features/carla_doses_ho3","d25_fog5","results/calib_dose_ho3","results/calib_dose_ho3_rep","results/calib_dose_ho3/dose_metrics.json",True),
            (15,"/mnt/features/carla_doses_ho2","d21_fog15","results/calib_dose_ho2","results/calib_dose_ho2_rep","results/calib_dose_ho2/dose_metrics.json",False),(25,S,"d05_fog25","results/calib_dose","results/calib_dose_rep","results/calib_dose/dose_metrics.json",False),
            (30,"/mnt/features/carla_doses_ho3","d26_fog30","results/calib_dose_ho3","results/calib_dose_ho3_rep","results/calib_dose_ho3/dose_metrics.json",True),(50,S,"d06_fog50","results/calib_dose","results/calib_dose_rep","results/calib_dose/dose_metrics.json",False),
            (55,"/mnt/features/carla_doses_ho3","d27_fog55","results/calib_dose_ho3","results/calib_dose_ho3_rep","results/calib_dose_ho3/dose_metrics.json",True),(80,"/mnt/features/carla_doses_ho3","d28_fog80","results/calib_dose_ho3","results/calib_dose_ho3_rep","results/calib_dose_ho3/dose_metrics.json",True),
            (90,"/mnt/features/carla_doses_ho","d12_fog90","results/calib_dose_ho","results/calib_dose_ho_rep","results/calib_dose_ho/dose_metrics.json",False)]
        cells=[]
        for dens,root,d,r42,r17,mf,fresh in LV:
            if d=="d00_ref": cells.append((root,f"{t}_{d}","no fog (the conditions it was trained in)","")); continue
            meas=np.mean([m(f"results/calib_dose/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{r42}/eval_{t}_d00_ref__on__{t}_{d}.json"),m(f"results/calib_dose_rep/eval_{t}_d00_ref__on__{t}_d00_ref.json")-m(f"{r17}/eval_{t}_d00_ref__on__{t}_{d}.json")])
            dist=json.load(open(mf))[f"{t}_{d}"]["obj_fid_ref"]; pred=A*(1-np.exp(-dist/tau))
            cells.append((root,f"{t}_{d}",f"fog {dens}"+("  (fresh)" if fresh else ""),f"measured −{meas:.1f} pp | formula −{pred:.1f}"))
        run(cells,t,frames,f"figures/cvpr/videos/V2_fog_sweep_{t}_labelled.mp4","Detector trained ONLY on clear-noon simulator frames. Shown: later frames of this drive it never saw, under fog it never saw.\nSame cars, same positions, so any loss is caused by the fog. 'formula' = label-free prediction; 'fresh' = fog levels never used to fit it (frame {frame}).")
    else:
        P1=json.load(open("results/calib_dose_ho/predictions.json"))["cells"]; P2=json.load(open("results/calib_dose_ho2/predictions.json"))["cells"]
        HO=[("results/calib_dose","results/calib_dose_ho"),("results/calib_dose_rep","results/calib_dose_ho_rep")]; HO2=[("results/calib_dose","results/calib_dose_ho2"),("results/calib_dose_rep","results/calib_dose_ho2_rep")]
        F=[(S,"d00_ref",RR,None,0),("/mnt/features/carla_doses_ho2","d21_fog15",HO2,P2,15),(S,"d05_fog25",RR,None,25),("/mnt/features/carla_doses_ho","d10_fog35",HO,P1,35),("/mnt/features/carla_doses_ho2","d22_fog45",HO2,P2,45),
           (S,"d06_fog50",RR,None,50),("/mnt/features/carla_doses_ho","d11_fog65",HO,P1,65),(S,"d07_fog75",RR,None,75),("/mnt/features/carla_doses_ho","d12_fog90",HO,P1,90)]
        cells=[]
        for root,d,rr,P,dens in F:
            sub="" if d=="d00_ref" else f"measured −{drop_seedmean(t,d,rr):.1f} pp"+(f" | predicted −{P[f'{t}_{d}'].get('pred_isotonic',P[f'{t}_{d}'].get('obj_fid_ref',{}).get('pred',float('nan'))):.1f}" if P else "")
            cells.append((root,f"{t}_{d}",f"fog density {dens}"+(" (calibration)" if P is None and dens else (" (held-out)" if P else "")),sub))
        run(cells,t,frames,f"figures/cvpr/videos/V2_fog_sweep_{t}.mp4",f"{t}, held-out drive (frame {{frame}}): the same frames at nine fog densities; 'predicted' = label-free metric, calibrated on other doses")
