#!/usr/bin/env python3
"""Ledger §15.93: graded VKITTI2 doses = (1 − α)·clone + α·variant (pixel-wise), labels copied from the clone. Output /mnt/features/vkitti2_graded/<Scene>/<cell>/."""
import os, shutil
from pathlib import Path
import numpy as np
from PIL import Image
SRC=Path("/mnt/features/vkitti2"); OUT=Path("/mnt/features/vkitti2_graded"); S=["Scene01","Scene02","Scene18","Scene20"]
V=[("d20_morning","morning"),("d21_sunset","sunset"),("d22_overcast","overcast"),("d23_fog","fog"),("d24_rain","rain")]; AL=[0.25,0.5,0.75]
for s in S:
    root=OUT/s; root.mkdir(parents=True,exist_ok=True)
    for c in [f"{s}_d00_ref"]+[f"{s}_{v}" for v,_ in V]:
        if not (root/c).exists(): os.symlink(SRC/c,root/c)
    clone=sorted((SRC/f"{s}_d00_ref").glob("frame_*.jpg"))
    for i,(v,name) in enumerate(V):
        for a in AL:
            cell=root/f"{s}_d4{i}_{name}_a{int(a*100)}"; cell.mkdir(exist_ok=True)
            for f in clone:
                o=cell/f.name
                if o.exists(): continue
                A=np.asarray(Image.open(f).convert("RGB"),dtype=np.float32); B=np.asarray(Image.open(SRC/f"{s}_{v}"/f.name).convert("RGB"),dtype=np.float32)
                Image.fromarray(np.clip((1-a)*A+a*B+.5,0,255).astype(np.uint8)).save(o,quality=95); shutil.copy(f.with_suffix(".txt"),o.with_suffix(".txt"))
            print(f"[blend] {cell.name}: {len(list(cell.glob('*.jpg')))} frames",flush=True)
