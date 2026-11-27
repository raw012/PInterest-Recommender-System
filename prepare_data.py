"""One-off preprocessing: download Pinterest images and extract image features
with a pretrained image encoder. workbook.ipynb only loads the cached outputs.

We compare two CLIP-trained encoders (a light CNN and a ViT), so the main
difference is the architecture:
  Model A (CNN):         python prepare_data.py --tag cnn --arch RN101 --weights openai
  Model B (Transformer): python prepare_data.py --tag vit --arch ViT-B-32       --weights laion2b_s34b_b79k

Outputs (in data/):
  images/{signature}.jpg           raw images, 236px wide (not tracked in git)
  feat_{tag}_product.npy           one embedding per product                 [n_products, D] float16
  feat_{tag}_scene_masked.npy      one embedding per (scene, product) pair, with that
                                   product's bounding box greyed out          [n_pairs, D]    float16
  feat_index.json                  row order of the product array (pairs follow fashion.json order)
  encoder_{tag}.json               parameter count + measured speed (images / second)
"""
import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--arch", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--benchmark-only", action="store_true")
args = ap.parse_args()

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMG = os.path.join(DATA, "images")
os.makedirs(IMG, exist_ok=True)

rows = [json.loads(l) for l in open(os.path.join(DATA, "fashion.json"))]
scenes = sorted({r["scene"] for r in rows})
products = sorted({r["product"] for r in rows})


# ---------- 1. download images (skipped if already present) ----------
def download(sig):
    path = os.path.join(IMG, sig + ".jpg")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    url = f"https://i.pinimg.com/236x/{sig[:2]}/{sig[2:4]}/{sig[4:6]}/{sig}.jpg"
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=20).read()
            open(path, "wb").write(data)
            return True
        except Exception:
            pass
    return False


ids = scenes + products
with cf.ThreadPoolExecutor(32) as ex:
    ok = list(tqdm(ex.map(download, ids), total=len(ids), desc="download"))
missing = {s for s, o in zip(ids, ok) if not o}
json.dump(sorted(missing), open(os.path.join(DATA, "missing_images.json"), "w"))
print(f"images available: {len(ids) - len(missing)}/{len(ids)}")

# ---------- 2. image encoder ----------
# Laptop-friendly: only the image tower is kept, GPU memory is capped, batches are small.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")
torch.set_num_threads(2)
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
clip_model, _, preprocess = open_clip.create_model_and_transforms(args.arch, pretrained=args.weights)
visual = clip_model.visual.to(device).eval()
del clip_model  # drop the text tower
GREY = (122, 116, 104)  # roughly the CLIP normalisation mean colour


def load(sig, bbox=None):
    if sig in missing:
        return Image.new("RGB", (224, 224), GREY)
    im = Image.open(os.path.join(IMG, sig + ".jpg")).convert("RGB")
    if bbox is not None:  # grey out the product so the query cannot "see the answer"
        w, h = im.size
        l, t, r, b = bbox
        patch = Image.new("RGB", (max(1, int((r - l) * w)), max(1, int((b - t) * h))), GREY)
        im.paste(patch, (int(l * w), int(t * h)))
    return im


@torch.no_grad()
def encode(batch):
    f = visual(batch.to(device))
    out = torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float16)
    if device == "mps":
        torch.mps.empty_cache()
    return out


def embed(items, name, bs=32, chunk=2048):
    """Encode in chunks saved to data/tmp/, so an interrupted run resumes where it stopped."""
    tmp = os.path.join(DATA, "tmp")
    os.makedirs(tmp, exist_ok=True)
    prep = lambda it: preprocess(load(*it))
    parts = []
    with cf.ThreadPoolExecutor(2) as pool:
        for c in tqdm(range(0, len(items), chunk), desc=name):
            path = os.path.join(tmp, f"{args.tag}_{name}_{c // chunk:04d}.npy")
            if not os.path.exists(path):
                sub = items[c:c + chunk]
                np.save(path, np.concatenate([encode(torch.stack(list(pool.map(prep, sub[i:i + bs]))))
                                              for i in range(0, len(sub), bs)]))
            parts.append(path)
    out = np.concatenate([np.load(f) for f in parts])
    for f in parts:
        os.remove(f)
    return out


# ---------- 3. speed benchmark (encoder only; skipped if already measured) ----------
stats_path = os.path.join(DATA, f"encoder_{args.tag}.json")
if not os.path.exists(stats_path):
    x = torch.stack([preprocess(load(p)) for p in products[:256]])
    encode(x[:32])  # warm-up
    t = time.time()
    for i in range(0, 256, 32):
        encode(x[i:i + 32])
    stats = {"arch": args.arch, "weights": args.weights,
             "params_M": round(sum(p.numel() for p in visual.parameters()) / 1e6, 1),
             "embed_dim": int(getattr(visual, "output_dim", 0)) or None,
             "input_px": int(x.shape[-1]),
             "images_per_sec": round(256 / (time.time() - t), 1), "device": device}
    json.dump(stats, open(stats_path, "w"), indent=1)
    del x
print(json.load(open(stats_path)))
if args.benchmark_only:
    raise SystemExit

# ---------- 4. features ----------
np.save(os.path.join(DATA, f"feat_{args.tag}_product.npy"),
        embed([(p, None) for p in products], "product"))
np.save(os.path.join(DATA, f"feat_{args.tag}_scene_masked.npy"),
        embed([(r["scene"], r["bbox"]) for r in rows], "scene_masked"))
json.dump({"products": products, "pairs": "same order as data/fashion.json"},
          open(os.path.join(DATA, "feat_index.json"), "w"))
print("done")
