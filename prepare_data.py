"""One-off preprocessing: download Pinterest images and extract CLIP features.

Run once (about 20-30 min on a laptop); workbook.ipynb only loads the cached outputs.

Outputs (in data/):
  images/{signature}.jpg          raw images, 236px wide (not tracked in git)
  feat_product.npy                 CLIP embedding per product      [n_products, 512], float16
  feat_scene.npy                   CLIP embedding per full scene   [n_scenes, 512],   float16
  feat_scene_masked.npy            CLIP embedding per (scene, product) pair, with that
                                   product's bounding box greyed out [n_pairs, 512], float16
  feat_index.json                  row order of the arrays above
"""
import concurrent.futures as cf
import json
import os
import urllib.request

import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMG = os.path.join(DATA, "images")
os.makedirs(IMG, exist_ok=True)

rows = [json.loads(l) for l in open(os.path.join(DATA, "fashion.json"))]
scenes = sorted({r["scene"] for r in rows})
products = sorted({r["product"] for r in rows})


# ---------- 1. download ----------
def url(sig, size="236x"):
    return f"https://i.pinimg.com/{size}/{sig[:2]}/{sig[2:4]}/{sig[4:6]}/{sig}.jpg"


def download(sig):
    path = os.path.join(IMG, sig + ".jpg")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    for _ in range(3):
        try:
            req = urllib.request.Request(url(sig), headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=20).read()
            with open(path, "wb") as f:
                f.write(data)
            return True
        except Exception:
            pass
    return False


ids = scenes + products
with cf.ThreadPoolExecutor(32) as ex:
    ok = list(tqdm(ex.map(download, ids), total=len(ids), desc="download"))
missing = [s for s, o in zip(ids, ok) if not o]
print(f"downloaded {sum(ok)}/{len(ids)}; missing {len(missing)}")
json.dump(missing, open(os.path.join(DATA, "missing_images.json"), "w"))
missing = set(missing)

# ---------- 2. CLIP features ----------
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
model = model.to(device).eval()
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
def embed(items, desc, bs=256):
    out = []
    for i in tqdm(range(0, len(items), bs), desc=desc):
        batch = torch.stack([preprocess(load(*it)) for it in items[i:i + bs]]).to(device)
        f = model.encode_image(batch)
        out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


np.save(os.path.join(DATA, "feat_product.npy"), embed([(p, None) for p in products], "products"))
np.save(os.path.join(DATA, "feat_scene.npy"), embed([(s, None) for s in scenes], "scenes"))
np.save(os.path.join(DATA, "feat_scene_masked.npy"),
        embed([(r["scene"], r["bbox"]) for r in rows], "masked scenes"))
json.dump({"products": products, "scenes": scenes, "pairs": "same order as fashion.json"},
          open(os.path.join(DATA, "feat_index.json"), "w"))
print("done")
