"""Label-survival check for generative restyles (Cosmos-Transfer) of CARLA frames.

A neutral judge (torchvision COCO Faster R-CNN, never trained on CARLA) is run on the original
frame, a resampling-only control and the restyled frame. For every labeled Car/Pedestrian box
that the judge FINDS in the original (IoU >= .5, score >= .5, class group match), we ask whether
it is still found in the control / restyle. Also counts judge detections that match no label
(inserted objects = label noise). Exploratory instrument; see ledger.

usage: cosmos_survival.py --orig DIR --restyled DIR [--names f1 f2 ...] [--control-size 768x432] --out JSON
"""
import argparse, json, os, sys
import numpy as np, torch, torchvision
from PIL import Image
from torchvision.ops import box_iou

VEH = {3, 4, 6, 8}  # COCO car, motorcycle, bus, truck
PER = {1}           # person

def load_labels(p):
    out = []
    for ln in open(p):
        f = ln.split()
        if not f or f[0] not in ("Car", "Pedestrian"): continue
        x1, y1, x2, y2 = map(float, f[4:8]); trunc = float(f[1])
        out.append((f[0], [x1, y1, x2, y2], trunc))
    return out

def judge(model, img, dev):
    t = torchvision.transforms.functional.to_tensor(img).to(dev)
    with torch.no_grad(): o = model([t])[0]
    keep = o["scores"] >= 0.5
    return o["boxes"][keep].cpu(), o["labels"][keep].cpu()

def found(gt, boxes, labels):
    """bool per GT: matched by a same-group detection with IoU >= .5"""
    if len(gt) == 0: return np.zeros(0, bool)
    g = torch.tensor([b for _, b, _ in gt])
    res = np.zeros(len(gt), bool)
    if len(boxes) == 0: return res
    iou = box_iou(g, boxes)
    for i, (c, _, _) in enumerate(gt):
        grp = VEH if c == "Car" else PER
        ok = torch.tensor([int(l) in grp for l in labels])
        res[i] = bool(((iou[i] >= 0.5) & ok).any())
    return res

def inserted(gt, boxes, labels):
    if len(boxes) == 0: return 0
    sel = torch.tensor([int(l) in VEH | PER for l in labels])
    b = boxes[sel]
    if len(b) == 0: return 0
    if len(gt) == 0: return len(b)
    iou = box_iou(b, torch.tensor([x for _, x, _ in gt]))
    return int((iou.max(1).values < 0.3).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True); ap.add_argument("--restyled", required=True)
    ap.add_argument("--labels", default=None, help="label dir (default: --orig)")
    ap.add_argument("--names", nargs="*"); ap.add_argument("--control-size", default="768x432")
    ap.add_argument("--min-h", type=float, default=25.0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lab = a.labels or a.orig
    names = a.names or sorted(os.path.splitext(f)[0] for f in os.listdir(a.restyled)
                              if f.endswith(".jpg") and "_control" not in f)
    cw, ch = map(int, a.control_size.split("x"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="COCO_V1").eval().to(dev)
    rows = []
    for n in names:
        gt = [g for g in load_labels(os.path.join(lab, n + ".txt")) if g[1][3] - g[1][1] >= a.min_h and g[2] <= 0.5]
        o = Image.open(os.path.join(a.orig, n + ".jpg")).convert("RGB")
        r = Image.open(os.path.join(a.restyled, n + ".jpg")).convert("RGB")
        if r.size != o.size:  # mp4 encoding pads odd sizes at the right/bottom; crop the pad back off
            assert r.size[0] - o.size[0] in (0, 1) and r.size[1] - o.size[1] in (0, 1), (n, r.size, o.size)
            r = r.crop((0, 0) + o.size)
        c = o.resize((cw, ch), Image.LANCZOS).resize(o.size, Image.LANCZOS)
        d = {k: judge(m, im, dev) for k, im in (("orig", o), ("ctrl", c), ("rest", r))}
        f = {k: found(gt, *d[k]) for k in d}
        rows.append(dict(name=n, n_gt=len(gt), classes=[g[0] for g in gt],
                         found={k: f[k].tolist() for k in f},
                         inserted={k: inserted(gt, *d[k]) for k in d}))
    def surv(k, cls=None):
        num = den = 0
        for r_ in rows:
            for i, c in enumerate(r_["classes"]):
                if cls and c != cls: continue
                if r_["found"]["orig"][i]:
                    den += 1; num += r_["found"][k][i]
        return num, den
    summ = {}
    for cls in (None, "Car", "Pedestrian"):
        key = cls or "all"
        summ[key] = {k: surv(k, cls) for k in ("ctrl", "rest")}
        summ[key]["gt_evaluable"] = sum(sum(1 for c in r_["classes"] if not cls or c == cls) for r_ in rows)
        summ[key]["found_in_orig"] = surv("orig", cls)[1]
    summ["inserted_per_frame"] = {k: float(np.mean([r_["inserted"][k] for r_ in rows])) for k in ("orig", "ctrl", "rest")}
    json.dump(dict(args=vars(a), summary=summ, rows=rows), open(a.out, "w"), indent=1)
    for k, v in summ.items(): print(k, v)

if __name__ == "__main__":
    main()
