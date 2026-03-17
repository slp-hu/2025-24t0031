#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_8feats_and_append.py  (dtype-fixed)
-------------------------------------------------
• 递归扫描 mel_root/**/*.npy
• 提取 8 维情感特征 → 归一化
• melon_id 列强制转换为 str，避免 merge 类型冲突
• 合并到原 CSV，输出新 CSV
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# ──────────────── CLI 参数 ────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mel_root",
                    default=r"E:\melon\extracted",
                    help="存放所有 .npy 的根目录；递归搜索")
parser.add_argument("--csv_in",
                    default=r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_intersection_v2_cleaned.csv",
                    help="待追加特征的原始 CSV 路径")
parser.add_argument("--csv_out",
                    default="joyful_melon_intersection_v2_with8feats.csv",
                    help="保存结果的路径")
parser.add_argument("--sr", type=int, default=16000,
                    help="采样率——应与提取 Mel 时一致")
parser.add_argument("--norm", choices=["minmax", "zscore"],
                    default="minmax", help="归一化方法")
args = parser.parse_args()

mel_root = Path(args.mel_root)
sr       = args.sr
n_mels   = 48
mel_bins = np.linspace(0, sr/2, n_mels, dtype=np.float64)

# ──────────────── 工具函数 ────────────────
def log_mel_to_lin(mel_log: np.ndarray) -> np.ndarray:
    mel_log = np.asarray(mel_log, dtype=np.float64)
    return (10.0 ** mel_log - 1.0) / 1e4

def extract_feat_from_file(npy_path: Path) -> dict:
    melon_id = npy_path.stem
    mel_log  = np.load(npy_path).astype(np.float64)
    mel      = log_mel_to_lin(mel_log)

    E   = mel.sum(axis=0, dtype=np.float64)
    dE  = np.diff(E)

    mel_norm = mel / (E + 1e-8)
    rough    = (mel_norm[:-1] * mel_norm[1:]).sum(axis=0)

    centroid = (mel.T @ mel_bins) / (E + 1e-8)
    flux     = np.abs(np.diff(mel, axis=1)).sum(axis=0)

    high_mask = mel_bins > 2000
    low_mask  = mel_bins < 300

    return dict(
        melon_id          = melon_id,
        median_dE_dt      = float(np.median(dE)),
        mean_roughness_prx= float(rough.mean()),
        p90_centroid      = float(np.percentile(centroid, 90)),
        high_band_ratio   = float(mel[high_mask].sum() / mel.sum()),
        low_band_ratio    = float(mel[low_mask].sum() / mel.sum()),
        energy_std        = float(np.std(E, dtype=np.float64)),
        p10_centroid      = float(np.percentile(centroid, 10)),
        median_flux       = float(np.median(flux)),
    )

# ──────────────── 1) 收集 .npy ────────────────
npy_files = list(mel_root.rglob("*.npy"))
if not npy_files:
    raise FileNotFoundError(f"在 {mel_root} 下没找到任何 .npy 文件！")

# ──────────────── 2) 提取特征 ────────────────
feature_rows = []
for npy in tqdm(npy_files, desc="Extracting", unit="file"):
    try:
        feature_rows.append(extract_feat_from_file(npy))
    except Exception as e:
        print(f"⚠️  跳过 {npy}: {e}")

df_feat = pd.DataFrame(feature_rows)

# ──────────────── 3) 归一化 ────────────────
norm_cols = [c for c in df_feat.columns if c != "melon_id"]
if args.norm == "minmax":
    df_feat[norm_cols] = (df_feat[norm_cols] - df_feat[norm_cols].min()) / \
                         (df_feat[norm_cols].max() - df_feat[norm_cols].min() + 1e-12)
else:
    df_feat[norm_cols] = (df_feat[norm_cols] - df_feat[norm_cols].mean()) / \
                         (df_feat[norm_cols].std(ddof=0) + 1e-12)

# ──────────────── 4) 合并 ────────────────
df_base = pd.read_csv(args.csv_in, dtype={"melon_id": str})   # ← ① 强制为 str
df_feat["melon_id"] = df_feat["melon_id"].astype(str)         # ← ② 也转为 str

df_join = df_base.merge(df_feat, on="melon_id", how="left")

df_join.to_csv(args.csv_out, index=False)
print(f"✅  已保存含 8 维特征的文件 → {args.csv_out}")
