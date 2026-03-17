#!/usr/bin/env python3
# coding: utf-8
"""
train_cm_drop_missing_norm.py
-----------------------------
• 验证集 = 旧 78 首原曲
• 训练集 = 其余原曲 + 特征齐全的增广样本
• 缺特征增广样本直接剔除
• 使用离线归一化好的特征文件，不再动态标准化
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------- 超参 ----------
VAL_SIZE, SEED = 78, 42
BATCH, LR, EPOCH, PATIENCE = 32, 1e-4, 30, 1
USE_ES = False

# ---------- 路径 & 离线归一化特征 ----------
CSV_BASE = "processed_annotations.csv"
CSV_AUG  = "processed_annotations_aug.csv"
MEL_DIR  = "mel_features_aug"
BEST_PTH = "best_drop_norm.pth"

FEAT = {
    "angry": ("angry_feats_aug_norm.csv", ["median_dE_dt", "mean_roughness_proxy"]),
    "happy": ("happy_feats_aug_norm.csv", ["p90_centroid", "high_band_ratio"]),
    "sad":   ("sad_feats_aug_norm.csv",   ["low_band_ratio", "energy_std"]),
    "relax": ("relax_feats_aug_norm.csv", ["p10_centroid", "median_flux"]),
}
LOAD_ANGRY  = True
LOAD_HAPPY  = True
LOAD_SAD    = True
LOAD_RELAX  = True

# ---------- 0 读取 & 验证集 ----------
df_base = pd.read_csv(CSV_BASE)
df_aug  = pd.read_csv(CSV_AUG)
df_base["song_id"] = df_base["song_id"].astype(str)
df_aug ["song_id"] = df_aug ["song_id"].astype(str)
df_aug["base_id"]  = df_aug["song_id"].str.replace(r"_.*$", "", regex=True)

sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
_, va_idx_390 = next(sss.split(df_base, df_base["label"]))
val_base_ids = set(df_base.loc[va_idx_390, "song_id"])

# ---------- 1 合并离线归一化特征 ----------
feat_cols = []
for tag, (csv_f, cols) in FEAT.items():
    if locals()[f"LOAD_{tag.upper()}"]:
        df_aug = df_aug.merge(pd.read_csv(csv_f), on="song_id", how="left")
        feat_cols += cols

# ---------- 2 剔除缺特征增广样本 ----------
before_train = df_aug[~df_aug["base_id"].isin(val_base_ids)].shape[0]
mask_all = df_aug[feat_cols].notna().all(axis=1)
df_aug = df_aug[mask_all].reset_index(drop=True)
after_train = df_aug[~df_aug["base_id"].isin(val_base_ids)].shape[0]
print(f"▶ 训练候选: {before_train}  剔除缺特征: {before_train-after_train}  最终训练: {after_train}")

is_val = (df_aug["base_id"].isin(val_base_ids)) & (df_aug["song_id"] == df_aug["base_id"])
va_idx = df_aug[is_val].index
tr_idx = df_aug[~df_aug["base_id"].isin(val_base_ids)].index
print(f"train={len(tr_idx)}  val={len(va_idx)}")  # val 应为 78
df = df_aug

# ---------- 3 标签 & genre ----------
le = LabelEncoder()
df["label_idx"] = le.fit_transform(df["label"])
CLS = list(le.classes_)

genres = sorted({g for row in df.genres for g in row.split(";")})
g2i = {g: i for i, g in enumerate(genres)}
N_GEN = len(genres)

# ---------- 4 Dataset ----------
class MusicDS(Dataset):
    def __init__(self, idxs):
        self.idxs = pd.Index(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        row = df.iloc[self.idxs[i]]
        # Mel-spectrogram
        mel = torch.from_numpy(
            np.load(os.path.join(MEL_DIR, f"{row.song_id}.npy"))
        ).unsqueeze(0).float()
        # Genre one-hot (with positional weights)
        gs = row.genres.split(";")
        w  = 1.0 / np.arange(1, len(gs)+1, dtype=np.float32)
        w /= w.sum()
        vec = np.zeros(N_GEN, np.float32)
        for k, g in enumerate(gs):
            vec[g2i[g]] = w[k]
        # 离线归一化好的额外特征
        extra_vals = row[feat_cols].to_numpy().astype(np.float32)
        extra = torch.from_numpy(extra_vals)
        # 标签
        label = torch.tensor(row.label_idx, dtype=torch.long)
        return mel, torch.from_numpy(vec), extra, label

# ---------- 5 网络架构 ----------
class Block(nn.Module):
    def __init__(self, c1, c2, down=True):
        super().__init__()
        self.c1, self.b1 = nn.Conv2d(c1,c2,3,1,1), nn.BatchNorm2d(c2)
        self.c2, self.b2 = nn.Conv2d(c2,c2,3,1,1), nn.BatchNorm2d(c2)
        self.short = nn.Identity() if c1==c2 else nn.Conv2d(c1,c2,1)
        self.relu, self.pool = nn.ReLU(), (nn.MaxPool2d(2) if down else nn.Identity())

    def forward(self, x):
        h = self.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return self.pool(self.relu(h + self.short(x)))

class EmoNet(nn.Module):
    def __init__(self, n_gen, extra_dim, n_cls):
        super().__init__()
        self.back = nn.Sequential(
            Block(1,16), Block(16,32), Block(32,64), Block(64,128,down=False)
        )
        self.gap   = nn.AdaptiveAvgPool2d((1,1))
        self.bn_mel, self.d_mel = nn.BatchNorm1d(128), nn.Dropout(0.6)
        self.fc_gen = nn.Linear(n_gen,32); self.bn_gen = nn.BatchNorm1d(32); self.d_gen = nn.Dropout(0.6)
        self.bn_ex  = nn.BatchNorm1d(extra_dim); self.d_ex = nn.Dropout(0.5)
        hid = 64
        self.head = nn.Sequential(
            nn.Linear(128+32+extra_dim, hid),
            nn.BatchNorm1d(hid), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(hid, n_cls)
        )

    def forward(self, m, g, e):
        B = m.size(0)
        x = self.gap(self.back(m)).view(B, -1)
        x = self.d_mel(F.relu(self.bn_mel(x)))
        g = self.d_gen(F.relu(self.bn_gen(self.fc_gen(g))))
        e = self.d_ex(F.relu(self.bn_ex(e)))
        return self.head(torch.cat([x, g, e], dim=1))

# ---------- 6 Loader & 训练 ----------
tr_ds = MusicDS(tr_idx)
va_ds = MusicDS(va_idx)
tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
va_ld = DataLoader(va_ds, batch_size=BATCH)

dev  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net  = EmoNet(N_GEN, len(feat_cols), len(CLS)).to(dev)
opt  = Adam(net.parameters(), lr=LR)
sched = ReduceLROnPlateau(opt, mode="min", factor=0.7, patience=PATIENCE)
crit = nn.CrossEntropyLoss()

hist = {"tr_loss":[], "va_loss":[], "tr_acc":[], "va_acc":[]}
best, wait = float("inf"), 0

for ep in range(1, EPOCH+1):
    # 训练
    net.train()
    tl = tc = 0
    for m, g, e, y in tr_ld:
        m, g, e, y = [t.to(dev) for t in (m, g, e, y)]
        opt.zero_grad()
        out = net(m, g, e)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        tl += loss.item() * y.size(0)
        tc += (out.argmax(1) == y).sum().item()
    tr_loss, tr_acc = tl/len(tr_ds), tc/len(tr_ds)

    # 验证
    net.eval()
    vl = vc = 0
    with torch.no_grad():
        for m, g, e, y in va_ld:
            m, g, e, y = [t.to(dev) for t in (m, g, e, y)]
            out = net(m, g, e)
            vl += crit(out, y).item() * y.size(0)
            vc += (out.argmax(1) == y).sum().item()
    va_loss, va_acc = vl/len(va_ds), vc/len(va_ds)

    hist["tr_loss"].append(tr_loss)
    hist["va_loss"].append(va_loss)
    hist["tr_acc"].append(tr_acc)
    hist["va_acc"].append(va_acc)

    print(f"Ep{ep:02d}  tr_loss={tr_loss:.3f}  va_loss={va_loss:.3f}  "
          f"tr_acc={tr_acc:.3f}  va_acc={va_acc:.3f}")
        # 打印 LR
    lr = opt.param_groups[0]['lr']
    print(f"        lr={lr:.2e}")

    sched.step(va_loss)
    if va_loss < best:
        best, wait = va_loss, 0
        torch.save(net.state_dict(), BEST_PTH)
    else:
        wait += 1
        if USE_ES and wait >= 5:
            print("Early stop!")
            break

# ---------- 7 评估 & 可视化 ----------
net.load_state_dict(torch.load(BEST_PTH))
net.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for m, g, e, y in va_ld:
        m, g, e = [t.to(dev) for t in (m, g, e)]
        y_pred.extend(net(m, g, e).argmax(1).cpu().numpy())
        y_true.extend(y.numpy())

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred, labels=range(len(CLS)))
ConfusionMatrixDisplay(cm, display_labels=CLS).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.show()

# Loss 曲线
plt.figure()
epochs = range(1, len(hist["tr_loss"]) + 1)
plt.plot(epochs, hist["tr_loss"], marker='o', label="Train Loss")
plt.plot(epochs, hist["va_loss"], marker='o', label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
