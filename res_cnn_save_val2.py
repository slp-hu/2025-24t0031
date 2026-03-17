#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_with_extra_feats.py
--------------------------------------------------
在原 CNN+Genre 结构上，加入两维归一化特征：
    median_dE_dt , mean_roughness_proxy
并在每个 epoch 结束时打印验证集各情感正确率。
"""

import os, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict

# ======================== 超参数 ========================
BATCH_SIZE   = 32
LR           = 1e-3
EPOCHS       = 100
VAL_SIZE     = 78          # 固定验证集 78 首
PATIENCE     = 5           # Early-Stopping
USE_ES       = False
MEL_DIR      = 'mel_features'           # 48 × T Mel .npy
CSV_PATH     = 'processed_annotations.csv'
EXTRA_CSV    = 'angry_feats.csv'          # song_id, median_dE_dt, mean_roughness_proxy
SCALER_PKL   = 'extra_feat_scaler.pkl'
BEST_MODEL   = 'best_model.pth'
# ========================================================

# ---------- 1. 读主 CSV + 额外特征 ----------
df = pd.read_csv(CSV_PATH)
extra_df = pd.read_csv(EXTRA_CSV)
df = df.merge(extra_df, on='song_id', how='left')
assert df[['median_dE_dt','mean_roughness_proxy']].notna().all().all(), "缺少额外特征"

# ---------- 2. LabelEncoder ----------
le = LabelEncoder()
df['label_idx'] = le.fit_transform(df['label'])
classes = list(le.classes_)           # ['Ambiguous','Angry',...]

# ---------- 3. Genre → One-hot 权重向量 ----------
all_genres = sorted({g for row in df['genres'] for g in row.split(';')})
genre2idx  = {g:i for i,g in enumerate(all_genres)}
n_genres   = len(all_genres)

# ---------- 4. 先划分索引，再拟合 scaler ----------
total_idx = np.arange(len(df))
train_idx, val_idx = random_split(total_idx, [len(df)-VAL_SIZE, VAL_SIZE],
                                  generator=torch.Generator().manual_seed(42))
train_idx, val_idx = train_idx.indices, val_idx.indices    # list[int]

scaler = StandardScaler()
scaler.fit(df.loc[train_idx, ['median_dE_dt','mean_roughness_proxy']])
df[['feat1_norm','feat2_norm']] = scaler.transform(
        df[['median_dE_dt','mean_roughness_proxy']])
joblib.dump(scaler, SCALER_PKL)

# ---------- 5. Dataset ----------
class MelGenreDataset(Dataset):
    def __init__(self, df, indices, mel_dir):
        self.df  = df.iloc[indices].reset_index(drop=True)
        self.dir = mel_dir
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        # Mel
        mel = torch.tensor(np.load(os.path.join(self.dir,f"{row.song_id}.npy")),
                           dtype=torch.float32).unsqueeze(0)   # (1,48,T)
        # Genre-weight vec
        w = np.zeros(n_genres, dtype=np.float32)
        gs = row.genres.split(';')
        inv = np.array([1/(i+1) for i in range(len(gs))], np.float32)
        inv /= inv.sum()
        for i,g in enumerate(gs): w[genre2idx[g]] = inv[i]
        w = torch.tensor(w)
        # Extra (2,)
        extra = torch.tensor([row.feat1_norm, row.feat2_norm], dtype=torch.float32)
        label = torch.tensor(row.label_idx, dtype=torch.long)
        return mel, w, extra, label

train_ds = MelGenreDataset(df, train_idx, MEL_DIR)
val_ds   = MelGenreDataset(df, val_idx,   MEL_DIR)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE)

# ---------- 6. 网络 ----------
class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, downsample=True, drop=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in,ch_out,3,1,1); self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,3,1,1); self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu, self.drop = nn.ReLU(), nn.Dropout2d(drop)
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()
        self.shortcut = (nn.Sequential(nn.Conv2d(ch_in,ch_out,1), nn.BatchNorm2d(ch_out))
                         if ch_in!=ch_out else nn.Identity())
    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(self.drop(out)))
        out = self.relu(out + self.shortcut(x))
        return self.pool(out)

class EmotionCNNRes(nn.Module):
    def __init__(self, n_genres, genre_emb=32, extra_dim=2, n_classes=5):
        super().__init__()
        self.backbone = nn.Sequential(
            ResBlock(1,16), ResBlock(16,32),
            ResBlock(32,64), ResBlock(64,128,downsample=False))
        self.gap  = nn.AdaptiveAvgPool2d((1,1))
        self.genre_fc = nn.Linear(n_genres, genre_emb)
        self.cls = nn.Sequential(
            nn.Linear(128+genre_emb+extra_dim,64),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,n_classes))
    def forward(self, mel, g_vec, extra):
        x = self.gap(self.backbone(mel)).view(mel.size(0), -1)     # (B,128)
        g = F.relu(self.genre_fc(g_vec))
        h = torch.cat([x, g, extra], dim=1)
        return self.cls(h)

# ---------- 7. 训练 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = EmotionCNNRes(n_genres, 32, 2, len(classes)).to(device)
opt    = Adam(model.parameters(), LR)
crit   = nn.CrossEntropyLoss()

best_val = 1e9; wait=0
for ep in range(1, EPOCHS+1):
    # --- train ---
    model.train(); t_loss=0; t_correct=0
    for mel,g,extra,y in train_loader:
        mel,g,extra,y = mel.to(device),g.to(device),extra.to(device),y.to(device)
        opt.zero_grad(); out=model(mel,g,extra); loss=crit(out,y); loss.backward(); opt.step()
        t_loss += loss.item()*y.size(0)
        t_correct += (out.argmax(1)==y).sum().item()
    train_loss = t_loss/len(train_ds); train_acc = t_correct/len(train_ds)

    # --- val ---
    model.eval(); v_loss=0; v_correct=0
    per_cls_hit = defaultdict(int); per_cls_cnt = defaultdict(int)
    with torch.no_grad():
        for mel,g,extra,y in val_loader:
            mel,g,extra,y = mel.to(device),g.to(device),extra.to(device),y.to(device)
            out=model(mel,g,extra); loss=crit(out,y)
            v_loss += loss.item()*y.size(0)
            preds = out.argmax(1)
            v_correct += (preds==y).sum().item()
            for yi,pi in zip(y.cpu().numpy(), preds.cpu().numpy()):
                per_cls_cnt[yi]+=1
                if yi==pi: per_cls_hit[yi]+=1
    val_loss = v_loss/len(val_ds); val_acc = v_correct/len(val_ds)

    # per-class accuracy
    per_cls_str = " | ".join(
        f"{classes[c]}:{per_cls_hit[c]/per_cls_cnt[c]:.2f}"
        for c in range(len(classes)))
    print(f"Ep{ep:03d}  "
          f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
          f"val_loss={val_loss:.4f} acc={val_acc:.3f} || {per_cls_str}")

    # save best
    if val_loss < best_val:
        best_val = val_loss; wait=0
        torch.save(model.state_dict(), BEST_MODEL)
        print(f"  ↳ saved best to {BEST_MODEL}")
    else:
        wait += 1
        if USE_ES and wait >= PATIENCE:
            print("Early-Stopping"); break
