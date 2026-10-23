"""Assemble Cosmos-Transfer outputs back to 1242x375 CARLA frames.
Full-frame outputs: crop the encoder pad. Tiled outputs (two 667x375 tiles at x=0 and x=575):
linear cross-fade over the 92-px overlap.  usage: cosmos_stitch.py SRC_DIR DST_ROOT {full|tiles} W H
"""
import os, sys
import numpy as np
from PIL import Image

src, dst, mode = sys.argv[1:4]; W, H = int(sys.argv[4]), int(sys.argv[5])
TW, X0 = 667, 575
names = sorted({f[:-4].rsplit("_", 1)[0] if mode == "tiles" else f[:-4]
                for f in os.listdir(src) if f.endswith(".jpg") and "_control" not in f})
for n in names:
    town, fr = n.split("_", 1)
    os.makedirs(os.path.join(dst, town), exist_ok=True)
    if mode == "full":
        im = np.asarray(Image.open(os.path.join(src, n + ".jpg")).convert("RGB"))[:H, :W]
    else:
        L = np.asarray(Image.open(os.path.join(src, n + "_L.jpg")).convert("RGB"))[:H, :TW].astype(np.float32)
        R = np.asarray(Image.open(os.path.join(src, n + "_R.jpg")).convert("RGB"))[:H, :TW].astype(np.float32)
        assert L.shape == (H, TW, 3) and R.shape == (H, TW, 3), (n, L.shape, R.shape)
        im = np.zeros((H, W, 3), np.float32)
        im[:, :X0] = L[:, :X0]; im[:, TW:] = R[:, TW - X0:]
        ov = TW - X0; w = np.linspace(1, 0, ov, dtype=np.float32)[None, :, None]
        im[:, X0:TW] = w * L[:, X0:TW] + (1 - w) * R[:, :ov]
        im = im.round().clip(0, 255).astype(np.uint8)
    Image.fromarray(im).save(os.path.join(dst, town, fr + ".jpg"), quality=95)
print(f"{len(names)} frames -> {dst}")
