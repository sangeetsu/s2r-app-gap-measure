#!/usr/bin/env python3
"""Ledger §15.67 (E2): Virtual KITTI 2 -> CARLA-testbed cell layout (frame_XXXXXX.jpg symlink + KITTI-format .txt), Car only.
Box kept if label == Car, height >= 15 px, occupancy_ratio >= 0.1. Lines sorted by trackID so object order is identical across variants
(paired metrics in dose_metrics.py align objects by order). Also converts KITTI tracking (E2b) into a separate root."""
import argparse, os, zipfile
from pathlib import Path
VAR={"clone":"d00_ref","morning":"d20_morning","sunset":"d21_sunset","overcast":"d22_overcast","fog":"d23_fog","rain":"d24_rain",
     "15-deg-left":"d30_rot15L","15-deg-right":"d31_rot15R","30-deg-left":"d32_rot30L","30-deg-right":"d33_rot30R"}
SCENES={"Scene01":"0001","Scene02":"0002","Scene06":"0006","Scene18":"0018","Scene20":"0020"}
def line(x1,y1,x2,y2): return f"Car 0.00 0 0.00 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} -1 -1 -1 -1000 -1000 -1000 -10\n"
def vkitti(raw,out):
    for s in SCENES:
        for v,suf in VAR.items():
            src=raw/s/v; cell=out/f"{s}_{suf}"; cell.mkdir(parents=True,exist_ok=True)
            lab={int(l.split()[0]):l.split()[1] for l in open(src/"info.txt").readlines()[1:]}
            rows={}
            for l in open(src/"bbox.txt").readlines()[1:]:
                p=l.split(); f,cam,tid=int(p[0]),int(p[1]),int(p[2])
                if cam!=0 or lab.get(tid)!="Car": continue
                x1,x2,y1,y2=float(p[3]),float(p[4]),float(p[5]),float(p[6]); occ=float(p[9])
                if y2-y1<15 or occ<0.1: continue
                rows.setdefault(f,[]).append((tid,x1,y1,x2,y2))
            imgs=sorted((src/"frames/rgb/Camera_0").glob("rgb_*.jpg"))
            for img in imgs:
                k=int(img.stem.split("_")[1]); dst=cell/f"frame_{k:06d}.jpg"
                if not dst.exists(): os.symlink(img,dst)
                with open(cell/f"frame_{k:06d}.txt","w") as fh:
                    for tid,x1,y1,x2,y2 in sorted(rows.get(k,[])): fh.write(line(x1,y1,x2,y2))
            print(f"{cell.name:<22} {len(imgs)} frames, {sum(len(r) for r in rows.values())} Car boxes",flush=True)
def tracking(zimg,zlab,out):
    zi=zipfile.ZipFile(zimg); zl=zipfile.ZipFile(zlab)
    for s,seq in SCENES.items():
        cell=out/f"{s}_d00_ref"; cell.mkdir(parents=True,exist_ok=True); rows={}
        for l in zl.read(f"training/label_02/{seq}.txt").decode().splitlines():
            p=l.split(); f,cls,occ=int(p[0]),p[2],int(p[4]); x1,y1,x2,y2=map(float,p[6:10])
            if cls!="Car" or occ>2 or y2-y1<15: continue
            rows.setdefault(f,[]).append((int(p[1]),x1,y1,x2,y2))
        names=sorted(n for n in zi.namelist() if n.startswith(f"training/image_02/{seq}/") and n.endswith(".png"))
        for n in names:
            k=int(Path(n).stem); dst=cell/f"frame_{k:06d}.jpg"
            if not dst.exists():
                from PIL import Image; import io
                Image.open(io.BytesIO(zi.read(n))).convert("RGB").save(dst,quality=95)
            with open(cell/f"frame_{k:06d}.txt","w") as fh:
                for tid,x1,y1,x2,y2 in sorted(rows.get(k,[])): fh.write(line(x1,y1,x2,y2))
        print(f"real {cell.name:<18} {len(names)} frames, {sum(len(r) for r in rows.values())} Car boxes",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["vkitti","tracking"],required=True)
    p.add_argument("--raw",default="/mnt/features/vkitti2_raw"); p.add_argument("--out",default="/mnt/features/vkitti2")
    p.add_argument("--zimg",default="/mnt/features/downloads/kitti_tracking/data_tracking_image_2.zip"); p.add_argument("--zlab",default="/mnt/features/downloads/kitti_tracking/data_tracking_label_2.zip")
    a=p.parse_args()
    vkitti(Path(a.raw),Path(a.out)) if a.mode=="vkitti" else tracking(a.zimg,a.zlab,Path(a.out))
