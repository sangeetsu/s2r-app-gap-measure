#!/usr/bin/env python3
"""V3: simulator-trained detectors on REAL images (side by side): clear-trained vs mild-fog-trained (Town10HD, seed 42). Slides are SELECTED to illustrate (stated on
screen): clear streets where both do well (KITTI, Cityscapes), real-fog images with the largest difference (Foggy Cityscapes beta=.02, RTTS). The end card reports
averages over ALL scored images (found objects per image) and the all-image, 8-town mAP — unselected. 'Found' = same class, IoU >= .5, score >= .5."""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
REPO=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(REPO/"src")); sys.path.insert(0,str(REPO/"tools"))
from s2rgap.detector import load_detector_from_ckpt
from s2rgap.data.kitti import KITTIDetection
from s2rgap.data.cityscapes_coco import CityscapesCOCO
from eval_detector_on_rtts import load as rtts_load
from per_object_mechanism import detected
F=lambda s,b=False: ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if b else ''}.ttf",s)
COL={1:(255,215,0),2:(0,229,255),3:(255,64,255)}; CW,CH=1920,1080; PW=944
dev=torch.device("cuda"); D=[load_detector_from_ckpt(f"checkpoints/anchind_Town10HD_{d}/last.pt",dev) for d in ("d00_ref","d05_fog25")]
def run(path,gt,gl):
    x=torch.from_numpy(np.asarray(Image.open(path).convert("RGB"),dtype=np.float32)/255.).permute(2,0,1).to(dev); res=[]
    for det in D:
        with torch.no_grad(): p=det([x])[0]
        p={k:v.cpu().numpy() for k,v in p.items()}; k=(gl==1)|(gl==2); res.append((p,int(detected(gt[k],gl[k],p["boxes"],p["labels"],p["scores"]).sum())))
    return res,int(((gl==1)|(gl==2)).sum())
rng=np.random.default_rng(0); pool={}
cs=CityscapesCOCO(Path.home()/"datasets/cityscapes/extracted/leftImg8bit","nvme/datasets/aldi_labels/cityscapes_val_instances.json",split="val",min_box_area=25)
for name,root in (("Cityscapes",Path.home()/"datasets/cityscapes/extracted/leftImg8bit/val"),("Foggy Cityscapes",Path("/mnt/features/cityscapes_foggy/b0.02/leftImg8bit/val"))):
    pool[name]=[]
    for i in range(len(cs)):
        tg=cs[i][1]; gl=tg["labels"].numpy(); r,n=run(root/tg["frame_rel"],tg["boxes"].numpy(),gl); pool[name].append((root/tg["frame_rel"],r,n))
    print(f"[v3] scored {name}: {len(pool[name])}",flush=True)
kv=KITTIDetection(kitti_root=str(Path.home()/"datasets/kitti/extracted/training"),split_json=str(Path.home()/"datasets/kitti/splits.json"),split="val"); pool["KITTI"]=[]
for i in rng.choice(len(kv),400,replace=False):
    tg=kv[int(i)][1]; r,n=run(Path.home()/"datasets/kitti/extracted/training/image_2"/tg["frame_rel"],tg["boxes"].numpy(),tg["labels"].numpy()); pool["KITTI"].append((Path.home()/"datasets/kitti/extracted/training/image_2"/tg["frame_rel"],r,n))
rt=rtts_load("/mnt/features/downloads/rtts"); pool["RTTS"]=[]
for i in rng.choice(len(rt),800,replace=False):
    f,b,l=rt[int(i)]; r,n=run(f,b,l); pool["RTTS"].append((f,r,n))
print("[v3] scoring done",flush=True)
stats={k:(np.mean([r[0][1] for _,r,n in v]),np.mean([r[1][1] for _,r,n in v]),np.mean([n for _,_,n in v]),len(v)) for k,v in pool.items()}
json.dump({k:{"clear_found":a,"fog_found":b,"labelled":c,"n_images":n} for k,(a,b,c,n) in stats.items()},open("figures/cvpr/videos/V3_all_image_stats.json","w"),indent=1)
def pick(name,k,mode):
    v=[z for z in pool[name] if z[2]>=4]
    key=(lambda z:(min(z[1][0][1],z[1][1][1])-abs(z[1][0][1]-z[1][1][1]))) if mode=="parity" else (lambda z:z[1][1][1]-z[1][0][1])
    return sorted(v,key=key,reverse=True)[:k]
slides=[("Clear real streets (KITTI)",z) for z in pick("KITTI",3,"parity")]+[("Clear real streets (Cityscapes)",z) for z in pick("Cityscapes",3,"parity")]+\
       [("Real streets with fog (Foggy Cityscapes)",z) for z in pick("Foggy Cityscapes",7,"gap")]+[("Real haze photographs (RTTS)",z) for z in pick("RTTS",7,"gap")]
def panel(path,res,n,lab,col):
    im=Image.open(path).convert("RGB"); s=PW/im.width; im=im.resize((PW,int(im.height*s))); dr=ImageDraw.Draw(im); p,h=res; k=p["scores"]>=.5
    for (x1,y1,x2,y2),l in zip(p["boxes"][k]*s,p["labels"][k]): dr.rectangle([x1,y1,x2,y2],outline=COL.get(int(l),(255,255,255)),width=3)
    bar=Image.new("RGB",(PW,46),(25,25,25)); d2=ImageDraw.Draw(bar); d2.text((10,4),lab,font=F(20,True),fill=col); txt=f"finds {h} of {n} cars/people"; d2.text((PW-10-d2.textlength(txt,font=F(22,True)),10),txt,font=F(22,True),fill=(255,220,120))
    out=Image.new("RGB",(PW,im.height+46)); out.paste(bar,(0,0)); out.paste(im,(0,46)); return out
tmp=Path(tempfile.mkdtemp(dir="/mnt/features")); fi=0
def emit(img,sec):
    global fi
    for _ in range(int(sec*10)): img.save(tmp/f"{fi:05d}.png"); fi+=1
title=Image.new("RGB",(CW,CH),(0,0,0)); d=ImageDraw.Draw(title)
for y,txt,sz,b in ((330,"A detector trained only in a driving simulator, tested on REAL photos",44,True),(420,"Left: trained on clear simulated weather.   Right: trained on light simulated fog.",32,False),(480,"Same simulated drives, same cars and people; only the weather differs.",32,False),(600,"Examples below are selected to show the effect; averages over all images are at the end.",26,False)):
    d.text(((CW-d.textlength(txt,font=F(sz,b)))/2,y),txt,font=F(sz,b),fill=(255,255,255))
emit(title,5)
for sec,(head,(path,res,n)) in [(3,s) for s in slides]:
    L=panel(path,res[0],n,"trained on CLEAR simulator images",(230,230,230)); R=panel(path,res[1],n,"trained on LIGHT SIMULATED FOG",(120,180,255))
    G=Image.new("RGB",(CW,CH),(0,0,0)); dg=ImageDraw.Draw(G); dg.text(((CW-dg.textlength(head,font=F(34,True)))/2,40),head,font=F(34,True),fill=(255,255,255))
    y0=max(110,(CH-max(L.height,R.height))//2); G.paste(L,(10,y0)); G.paste(R,(CW-PW-10,y0)); dg.text((20,CH-40),"selected example  |  car yellow, pedestrian cyan, cyclist magenta",font=F(20),fill=(170,170,170)); emit(G,sec)
T=["Town01","Town02","Town03","Town04","Town05","Town06","Town07","Town10HD"]; mm=lambda f: json.load(open(f))["metrics"]["mAP@IoU=0.5"]*100
SRC=[("KITTI (clear)",lambda c:f"results/calib_dose/src_{c}_kittival.json","KITTI"),("Cityscapes (clear)",lambda c:f"results/calib_dose/src_{c}_cityscapesval.json","Cityscapes"),("Foggy Cityscapes",lambda c:f"results/calib_dose_foggy/src_{c}_foggy0.02.json","Foggy Cityscapes"),("RTTS (real haze)",lambda c:f"results/calib_dose_rtts/src_{c}_rtts.json","RTTS")]
end=Image.new("RGB",(CW,CH),(0,0,0)); de=ImageDraw.Draw(end); de.text((80,60),"Over ALL images (not selected)",font=F(40,True),fill=(255,255,255))
de.text((80,130),"objects found per image (this town's two detectors)          accuracy (mAP), mean of 8 simulator towns",font=F(24),fill=(200,200,200))
for j,(lab,fn,key) in enumerate(SRC):
    a,b,n,ni=stats[key]; ca=np.mean([mm(fn(f"{t}_d00_ref")) for t in T]); cb=np.mean([mm(fn(f"{t}_d05_fog25")) for t in T]); y=220+j*190
    de.text((80,y),lab,font=F(32,True),fill=(255,255,255)); de.text((80,y+50),f"clear-trained {a:.1f}   fog-trained {b:.1f}   of {n:.1f} labelled  ({ni} images)",font=F(26),fill=(255,220,120))
    de.text((1060,y+50),f"clear-trained {ca:.1f}   fog-trained {cb:.1f}   ({cb-ca:+.1f})",font=F(26),fill=(120,180,255))
de.text((80,CH-80),"Takeaway: light simulated fog costs nothing on clear real streets and helps on real fog.",font=F(30,True),fill=(255,255,255)); emit(end,8)
out="figures/cvpr/videos/V3_real_images_clear_vs_fog_trained.mp4"
subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate","10","-i",str(tmp/"%05d.png"),"-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2","-c:v","libx264","-pix_fmt","yuv420p","-crf","22",out],check=True)
Image.open(tmp/f"{60:05d}.png").save(out.replace(".mp4","_slide1.png")); Image.open(tmp/f"{fi-1:05d}.png").save(out.replace(".mp4","_endcard.png")); Image.open(tmp/f"{50+30*8:05d}.png").save(out.replace(".mp4","_fogslide.png"))
for f in tmp.iterdir(): f.unlink()
tmp.rmdir(); print("[video]",out,{k:tuple(round(x,2) for x in v[:3]) for k,v in stats.items()})
