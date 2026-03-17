#!/usr/bin/env python
r"""
make_clusters.py
================

Reads the Joyful‑Melon CSV, parses the *tag* column, assigns each song to one of
three **pre‑defined emotion clusters** (see CLUSTERS below), and marks songs
whose tags span ≥2 clusters as **ambiguous**.

Output file keeps所有原列 + 三个 ClusterX 列 + ambiguous 列。

• 数据源 : C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_intersection_v2_with8feats.csv
• 输出    : same dir, 新文件 *_with_clusters.csv

Run simply:
    python make_clusters.py

Dependence: pandas only (built‑in ast).
"""
from __future__ import annotations
import ast
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. 路径配置
# ---------------------------------------------------------------------------
IN_PATH  = Path(r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_intersection_v2_with8feats.csv")
OUT_PATH = IN_PATH.with_name(IN_PATH.stem + "_with_clusters.csv")
TAG_COL  = "tag"  # 标签列名字

# ---------------------------------------------------------------------------
# 2. 预定义簇映射
# ---------------------------------------------------------------------------
CLUSTERS: dict[int, set[str]] = {
    0: {"Nostalgia", "Touching"},
    1: {"Excitement", "Fresh", "Happiness", "Romantic", "Relaxation"},
    2: {"Healing", "Quiet", "Loneliness", "Missing", "Sadness"},
}
ALL_LABELS = {lab for s in CLUSTERS.values() for lab in s}

# ---------------------------------------------------------------------------
# 3. 读取并解析
# ---------------------------------------------------------------------------
print(f"Loading {IN_PATH} …")
df = pd.read_csv(IN_PATH)
print(f"Rows: {len(df):,}")
if TAG_COL not in df.columns:
    raise KeyError(f"Column '{TAG_COL}' not found in CSV")

print("Parsing tag column …")
tag_dicts = df[TAG_COL].apply(ast.literal_eval)

# ---------------------------------------------------------------------------
# 4. 生成 Cluster 列
# ---------------------------------------------------------------------------
for cid, labels in CLUSTERS.items():
    col = f"Cluster{cid}"
    df[col] = tag_dicts.apply(lambda d: int(any(d.get(l, 0) > 0 for l in labels)))

cluster_cols = [f"Cluster{cid}" for cid in CLUSTERS]

df["ambiguous"] = (df[cluster_cols].sum(axis=1) > 1).astype(int)

# ---------------------------------------------------------------------------
# 5. 保存
# ---------------------------------------------------------------------------
print("Saving …")
df.to_csv(OUT_PATH, index=False)
print(f"Done. Output → {OUT_PATH}")
