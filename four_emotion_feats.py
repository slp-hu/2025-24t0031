#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_feats_by_emotion.py
---------------------------------
从 Mel .npy (48×T, log10 缩放) 直接派生 8 维统计特征。
写出：
    angry_feats.csv      (song_id, median_dE_dt, mean_roughness_proxy)
    happy_feats.csv      (song_id, p90_centroid, high_band_ratio)
    sad_feats.csv        (song_id, low_band_ratio, energy_std)
    relax_feats.csv      (song_id, p10_centroid, median_flux)
"""
import numpy as np, pandas as pd, os, argparse, tqdm, json
from pathlib import Path

# ---------- 参数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--mel_dir", default="mel_features_aug", help="*.npy 路径")
parser.add_argument("--sr", type=int, default=16000)
args = parser.parse_args()

mel_dir = Path(args.mel_dir)
sr      = args.sr
n_fft   = 512      # 和提取 Mel 时一致
n_mels  = 48
mel_bins = np.linspace(0, sr/2, n_mels)

def log_mel_to_lin(mel_log):
    return (10**mel_log - 1.0) / 10000.0

# ---------- 每首歌提特征 ----------
records_angry, rec_happy, rec_sad, rec_relax = [], [], [], []
for npy in tqdm.tqdm(sorted(mel_dir.glob("*.npy")), desc="Extract"):
    song_id = npy.stem
    mel_log = np.load(npy)
    mel     = log_mel_to_lin(mel_log)
    E       = mel.sum(axis=0)
    dE      = np.diff(E)
    # 动态粗糙度 proxy
    mel_norm = mel / (E + 1e-8)
    rough    = (mel_norm[:-1] * mel_norm[1:]).sum(axis=0)

    # centroid & flux
    centroid = (mel.T @ mel_bins) / (E + 1e-8)
    flux     = np.abs(np.diff(mel, axis=1)).sum(axis=0)

    # angry
    records_angry.append(dict(song_id=song_id,
                              median_dE_dt=float(np.median(dE)),
                              mean_roughness_proxy=float(rough.mean())))
    # happy
    high_mask = mel_bins > 2000
    rec_happy.append(dict(song_id=song_id,
                          p90_centroid=float(np.percentile(centroid, 90)),
                          high_band_ratio=float(mel[high_mask].sum()/mel.sum())))
    # sad
    low_mask  = mel_bins < 300
    rec_sad.append(dict(song_id=song_id,
                        low_band_ratio=float(mel[low_mask].sum()/mel.sum()),
                        energy_std=float(E.std())))
    # relax
    rec_relax.append(dict(song_id=song_id,
                          p10_centroid=float(np.percentile(centroid, 10)),
                          median_flux=float(np.median(flux))))

# ---------- 写 CSV ----------
pd.DataFrame(records_angry).to_csv("angry_feats_aug.csv",   index=False)
pd.DataFrame(rec_happy).to_csv("happy_feats_aug.csv",   index=False)
pd.DataFrame(rec_sad).to_csv("sad_feats_aug.csv",       index=False)
pd.DataFrame(rec_relax).to_csv("relax_feats_aug.csv",   index=False)
print("✔  All feature CSVs saved.")
