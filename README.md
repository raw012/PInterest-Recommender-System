# Scene-based Complementary Product Recommendation on Pinterest

Given a real-world scene (e.g., a street-style or outfit photo), we want to recommend fashion products that are **visually compatible** with it. This project studies the "Complete the Look" task on the Pinterest Fashion Compatibility dataset. We start from classic implicit-feedback recommenders and move to visually aware models.

## Dataset

**Pinterest Fashion Compatibility (STL-Dataset)**, from Kang et al., CVPR 2019.

| Statistic | Value |
|---|---|
| Scenes | 47,739 |
| Products | 38,111 |
| Scene–product pairs | 93,274 |

Each record links a scene image to a product that appears in it, together with the product's bounding box in the scene:

```json
{"product": "0027e30879ce3d87f82f699f148bff7e",
 "scene":   "cdab9160072dd1800038227960ff6467",
 "bbox":    [0.434097, 0.859363, 0.560254, 1.0]}
```

Bounding boxes are normalized `(left, top, right, bottom)`, with `(0, 0)` at the top-left corner.

### Getting the data

- Dataset page: https://cseweb.ucsd.edu/~jmcauley/datasets.html#pinterest
- Official repository: https://github.com/kang205/STL-Dataset
- Direct downloads (we use the fashion subset):
  - [`fashion.json`](https://raw.githubusercontent.com/kang205/STL-Dataset/master/fashion.json): scene–product pairs (~10 MB)
  - [`fashion-cat.json`](https://raw.githubusercontent.com/kang205/STL-Dataset/master/fashion-cat.json): product categories (~3 MB)

```bash
mkdir -p data
curl -L -o data/fashion.json     https://raw.githubusercontent.com/kang205/STL-Dataset/master/fashion.json
curl -L -o data/fashion-cat.json https://raw.githubusercontent.com/kang205/STL-Dataset/master/fashion-cat.json
```

Images are not included in the dataset. Each `scene` and `product` ID is a Pinterest image signature, and the image URL is built from it:

```python
def convert_to_url(signature):
    prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
    return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)
```

### Notes

- Interactions are very sparse (about 2 products per scene), and most evaluation scenes are unseen during training. ID-only collaborative filtering therefore faces a cold-start problem, which motivates using visual content.

## Problem Formulation

- **Task:** for a query scene *s*, rank candidate products *p* so that products compatible with *s* come first.
- **Input:** scene and product (IDs, image features, product category).
- **Output:** a compatibility score *f(s, p)*.
- **Learning objective:** pairwise ranking with Bayesian Personalized Ranking (BPR), using observed (scene, product) pairs as positives and sampled products as negatives.

## Evaluation Protocol

- Split train / validation / test **by scene**.
- For each held-out positive pair, sample 99 negative products, from the same category where possible.
- Metrics: **AUC**, **Hit@10**, **NDCG@10**.

## Models

| Model | Signal | Notes |
|---|---|---|
| Random | — | Lower bound |
| Popularity | Interaction counts | Non-personalized baseline |
| Jaccard / co-occurrence | Interactions | Item–item similarity |
| BPR-MF | Interactions | Latent factors; fails on unseen scenes (cold start) |
| Pretrained CNN + cosine | Images | Zero-shot visual similarity (ResNet / CLIP features) |
| Visual BPR | Images + interactions | Learned projection of visual features, trained with BPR (VBPR-style) |
| + Region / category variants | Images | Scene crops from bounding boxes; category-aware scoring |

## Results (v1, test set, 99 same-category negatives)

| Model | AUC | Hit@10 | NDCG@10 |
|---|---|---|---|
| Random | 0.500 | 0.108 | 0.048 |
| Popularity | 0.504 | 0.321 | 0.210 |
| BPR-MF (item bias, cold scenes) | 0.715 | 0.318 | 0.208 |
| CLIP zero-shot (masked scene) | 0.625 | 0.227 | 0.121 |
| **Visual BPR (CLIP + category, ours)** | **0.772** | **0.430** | **0.245** |

## Repository Structure

```
workbook.ipynb     # main notebook: EDA, split, models, evaluation (export -> workbook.html)
prepare_data.py    # one-off: download images + extract CLIP ViT-B/32 features
requirements.txt
data/
  fashion.json, fashion-cat.json   # STL-Dataset (fashion subset)
  feat_*.npy, feat_index.json      # cached CLIP features (float16)
  images/                          # downloaded images (not tracked, ~1 GB)
```

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python prepare_data.py        # optional: features are already in data/
.venv/bin/jupyter nbconvert --to notebook --execute --inplace workbook.ipynb
.venv/bin/jupyter nbconvert --to html workbook.ipynb
```

## References

- W.-C. Kang, E. Kim, J. Leskovec, C. Rosenberg, J. McAuley. *Complete the Look: Scene-based Complementary Product Recommendation.* CVPR 2019.
- R. He, J. McAuley. *VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback.* AAAI 2016.
- S. Rendle et al. *BPR: Bayesian Personalized Ranking from Implicit Feedback.* UAI 2009.
