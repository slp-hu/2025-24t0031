#!/usr/bin/env python
r"""
cluster_BCD_majority.py  –  fixed version
=========================================

• 输入 : C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_filtered_audioexist.csv
• 输出 : *_clustered_BCD.csv   (含 cluster_id_B/C/D 及 amb_B70 / amb_C70 / amb_D70)

功能概览
--------
1. **方案 B**  二值 0/1  → Cosine‑Agglomerative (average)
2. **方案 C**  标准化计数 → GaussianMixture(full)
3. **方案 D**  TF‑IDF     → Ward 层次
4. 自动选 k (Silhouette / BIC / CH)
5. 计算每簇情感均值 → "emotion → 唯一簇" 映射
6. **主簇占比 ≥70% 规则** 重新判定 ambiguous
7. 曲线弹窗 (`plt.show()`)，不保存 PNG

依赖
----
```
pip install pandas numpy scipy scikit-learn matplotlib scikit-learn-extra
```
"""
from __future__ import annotations
import ast, numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster

warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["font.size"] = 9
np.random.seed(0)   # 保证可复现

# ------------------ 常量 ------------------ #
CSV   = Path(r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_filtered_audioexist.csv")
EMO   = ['Healing','Nostalgia','Excitement','Sadness','Romantic','Quiet',
         'Happiness','Loneliness','Touching','Missing','Fresh','Relaxation']
THRESH = 0.7     # 主簇占比阈值

# ------------------ 读数据 ------------------ #
df = pd.read_csv(CSV)
X_count = np.array([[ast.literal_eval(t)[e] for e in EMO] for t in df['tag']], dtype=float)
X_bin   = (X_count > 0).astype(int)
X_std   = StandardScaler(with_mean=False).fit_transform(X_count)
X_tfidf = normalize(TfidfTransformer(norm=None).fit_transform(X_count), norm='l2')

# ------------------ 公用函数 ------------------ #

def build_unique_map(cluster_means: pd.DataFrame) -> dict[str,int]:
    """将每个情感词唯一地指派给均值最高的簇"""
    return {emo: int(cluster_means[emo].idxmax()) for emo in cluster_means.columns}


def ambiguous_majority(active_emo: list[str], emo2cid: dict[str,int], thresh: float = THRESH) -> int:
    if not active_emo:
        return 0
    cids = [emo2cid[e] for e in active_emo]
    if len(set(cids)) == 1:
        return 0
    counts = np.bincount(cids, minlength=max(emo2cid.values())+1)
    return int(counts.max() / len(cids) < thresh)


def count_amb_majority(tag_series: pd.Series, emo2cid: dict[str,int], thresh: float = THRESH) -> int:
    amb = 0
    for s in tag_series:
        tag = ast.literal_eval(s)
        active = [e for e,v in tag.items() if v>0]
        amb  += ambiguous_majority(active, emo2cid, thresh)
    return amb

# ------------------ 方案 B ------------------ #
print("\n=== 方案 B (Cosine‑Agglo) ===")
D_cos = pairwise_distances(X_bin, metric='cosine')
ks, sil_scores = [], []
for k in range(2,11):
    labels_tmp = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average', compute_full_tree=True).fit(D_cos).labels_
    sil = silhouette_score(D_cos, labels_tmp, metric='precomputed')
    ks.append(k); sil_scores.append(sil)
Best_k_B = ks[int(np.argmax(sil_scores))]
print(f"k_B={Best_k_B}, silhouette={sil_scores[Best_k_B-2]:.3f}")
plt.figure(); plt.plot(ks, sil_scores, marker='o'); plt.axvline(Best_k_B, c='red', ls='--');
plt.title('Silhouette vs k (Cosine)'); plt.xlabel('k'); plt.ylabel('Silhouette'); plt.tight_layout(); plt.show()
labels_B = AgglomerativeClustering(n_clusters=Best_k_B, metric='precomputed', linkage='average', compute_full_tree=True).fit(D_cos).labels_
cluster_means_B = pd.DataFrame(X_bin).groupby(labels_B).mean(); cluster_means_B.columns = EMO
emo2cid_B = build_unique_map(cluster_means_B)
amb_B = count_amb_majority(df['tag'], emo2cid_B)
print(f"ambiguous_B (≥2簇且主簇<70%) = {amb_B}  ({amb_B/len(df):.1%})")
for cid,size in sorted(zip(*np.unique(labels_B, return_counts=True))):
    print(f"  B簇{cid}: {size} 首, kw={[e for e,c in emo2cid_B.items() if c==cid]}")

df['cluster_id_B'] = labels_B

# ------------------ 方案 C ------------------ #
print("\n=== 方案 C (GMM+BIC) ===")
BICs, models = [], []
for k in range(2,11):
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=0).fit(X_std)
    BICs.append(gmm.bic(X_std)); models.append(gmm)
Best_k_C = int(np.argmin(BICs)) + 2
print(f"k_C={Best_k_C}, BIC={min(BICs):.0f}")
plt.figure(); plt.plot(range(2,11), BICs, marker='o'); plt.axvline(Best_k_C,c='red',ls='--');
plt.title('GMM BIC vs k'); plt.xlabel('k'); plt.ylabel('BIC'); plt.tight_layout(); plt.show()
labels_C = models[Best_k_C-2].predict(X_std)
cluster_means_C = pd.DataFrame(X_bin).groupby(labels_C).mean(); cluster_means_C.columns = EMO
emo2cid_C = build_unique_map(cluster_means_C)
amb_C = count_amb_majority(df['tag'], emo2cid_C)
print(f"ambiguous_C (≥2簇且主簇<70%) = {amb_C}  ({amb_C/len(df):.1%})")
for cid,size in sorted(zip(*np.unique(labels_C, return_counts=True))):
    print(f"  C簇{cid}: {size} 首, kw={[e for e,c in emo2cid_C.items() if c==cid]}")

df['cluster_id_C'] = labels_C

# ------------------ 方案 D ------------------ #
print("\n=== 方案 D (TF‑IDF + Ward) ===")
Z_D = linkage(X_tfidf.toarray(), method='ward')
CH_scores = [calinski_harabasz_score(X_tfidf.toarray(), fcluster(Z_D,k,criterion='maxclust')) for k in range(2,11)]
Best_k_D = int(np.argmax(CH_scores)) + 2
print(f"k_D={Best_k_D}, CH={max(CH_scores):.1f}")
plt.figure(); plt.plot(range(2,11), CH_scores, marker='s'); plt.axvline(Best_k_D,c='red',ls='--');
plt.title('Calinski–Harabasz vs k (TF‑IDF+Ward)'); plt.xlabel('k'); plt.ylabel('CH'); plt.tight_layout(); plt.show()
labels_D = fcluster(Z_D, Best_k_D, criterion='maxclust')
cluster_means_D = pd.DataFrame(X_bin).groupby(labels_D).mean(); cluster_means_D.columns = EMO
emo2cid_D = build_unique_map(cluster_means_D)
amb_D = count_amb_majority(df['tag'], emo2cid_D)
print(f"ambiguous_D (≥2簇且主簇<70%) = {amb_D}  ({amb_D/len(df):.1%})")
for cid,size in sorted(zip(*np.unique(labels_D, return_counts=True))):
    print(f"  D簇{cid}: {size} 首, kw={[e for e,c in emo2cid_D.items() if c==cid]}")

df['cluster_id_D'] = labels_D

# ------------------ 保存 ------------------ #
OUT = CSV.with_name(f"{CSV.stem}_clustered_BCD.csv")
df.to_csv(OUT, index=False)
print("\nSaved →", OUT)
