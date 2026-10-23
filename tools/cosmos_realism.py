"""How far does a generative restyle move CARLA frames toward real photographs? (exploratory, DINOv3 image level)
Same features as the testbed (dinov3-vitl16, bicubic 352x1216, mean of patch tokens). Real reference: KITTI val, first 1000 frames
(identical selection to tools/fid_to_real.py). Reports per set-up:
  p_real   mean probability that a linear real-vs-CARLA probe (KITTI val vs 400 CARLA clear-noon TRAINING records, 50 per town) calls the frame real
  nn_kitti median nearest-KITTI distance / KITTI's own leave-one-out median NN distance (1.0 = as close as a real KITTI frame is to its neighbour)
  disp     mean paired displacement from the original frame / median pairwise distance among the CARLA probe frames
usage: cosmos_realism.py --frames Town07:frame_000820 ... --set NAME=DIR_ROOT ... --out JSON   (DIR_ROOT/<Town>/<frame>.jpg)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "tools"))
from s2rgap.adaptation.spatial_lookup import SpatialFeatureLookup
from s2rgap.data.kitti import KITTIDetection
from fid_to_real import img_feats
from transformers import AutoModel

ap = argparse.ArgumentParser(); ap.add_argument("--frames", nargs="+", required=True); ap.add_argument("--set", nargs="+", required=True)
ap.add_argument("--root", default="/mnt/features/carla_doses"); ap.add_argument("--out", required=True); a = ap.parse_args()
dev = "cuda"
pre = transforms.Compose([transforms.Resize((352, 1216), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(),
                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
m = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", dtype=torch.bfloat16).to(dev).eval()
off = 1 + getattr(m.config, "num_register_tokens", 0)
def feats(paths, bs=16):
    out = []
    for i in range(0, len(paths), bs):
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(dev, torch.bfloat16)
        with torch.no_grad(): h = m(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, off:, :]
        out.append(h.float().mean(1).cpu().numpy())
    return np.concatenate(out)
lk = SpatialFeatureLookup.from_npz(Path.home() / "datasets/kitti/spatial_features_dinov3.npz")
kv = KITTIDetection(kitti_root=str(Path.home() / "datasets/kitti/extracted/training"), split_json=str(Path.home() / "datasets/kitti/splits.json"), split="val")
K = img_feats(lk, [kv[i][1]["frame_rel"] for i in range(min(1000, len(kv)))]).astype(np.float32)
towns = sorted(p.name[:-8] for p in Path(a.root).glob("Town*_d00_ref"))
C = feats([f"{a.root}/{t}_d00_ref/frame_{k:06d}.jpg" for t in towns for k in range(0, 800, 16)])
X = np.concatenate([K, C]); y = np.r_[np.ones(len(K)), np.zeros(len(C))]
probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
cv = float(cross_val_score(probe, X, y, cv=5).mean()); probe.fit(X, y)
Kt = torch.from_numpy(K).to(dev); dKK = torch.cdist(Kt, Kt); dKK.fill_diagonal_(float("inf")); kscale = float(dKK.min(1).values.median())
Ct = torch.from_numpy(C).to(dev); dCC = torch.cdist(Ct, Ct); cscale = float(dCC[torch.triu(torch.ones_like(dCC, dtype=torch.bool), 1)].median())
fr = [s.split(":") for s in a.frames]
F0 = feats([f"{a.root}/{t}_d00_ref/{f}.jpg" for t, f in fr])
def summ(F):
    d = torch.cdist(torch.from_numpy(F).to(dev), Kt).min(1).values.cpu().numpy()
    return dict(p_real=float(probe.predict_proba(F)[:, 1].mean()), p_real_per_frame=probe.predict_proba(F)[:, 1].round(3).tolist(),
                nn_kitti=float(np.median(d) / kscale), disp=float(np.linalg.norm(F - F0, axis=1).mean() / cscale))
res = dict(probe_cv_acc=cv, n_kitti=len(K), n_carla_probe=len(C), sets={"original": summ(F0)})
for s in a.set:
    name, root = s.split("=", 1); res["sets"][name] = summ(feats([f"{root}/{t}/{f}.jpg" for t, f in fr]))
json.dump(res, open(a.out, "w"), indent=1)
print(f"probe CV acc {cv:.3f} (KITTI {len(K)} vs CARLA {len(C)})")
for k, v in res["sets"].items(): print(f"{k:<10} p_real {v['p_real']:.3f}  nn_kitti {v['nn_kitti']:.2f}  disp {v['disp']:.2f}")
