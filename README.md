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

## Repository Structure

```
data/          # raw STL-Dataset files (cached image features are not tracked)
src/
  data.py      # loading, splitting, negative sampling
  evaluate.py  # shared evaluation (AUC, Hit@K, NDCG@K)
  models/      # baselines and visual models
notebooks/     # EDA and experiments
```

## References

- W.-C. Kang, E. Kim, J. Leskovec, C. Rosenberg, J. McAuley. *Complete the Look: Scene-based Complementary Product Recommendation.* CVPR 2019.
- R. He, J. McAuley. *VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback.* AAAI 2016.
- S. Rendle et al. *BPR: Bayesian Personalized Ranking from Implicit Feedback.* UAI 2009.
