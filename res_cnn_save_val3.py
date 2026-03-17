#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cm.py
--------------------------------------------------
• 固定验证集 (VAL_SIZE 首，随机种子保证可复现)
• 训练过程中遇到更低 val_loss 就保存权重 best.pth
• 训练结束后加载 best.pth → 重新跑验证集 → 画混淆矩阵
• 可视化 Loss & Accuracy 曲线
"""

import os, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.optim import Adam

# ========== 开关 ==========
LOAD_ANGRY  = True
LOAD_HAPPY  = True
LOAD_SAD    = True
LOAD_RELAX  = True
USE_ES      = False
VAL_SIZE    = 78
SEED        = 42

# ========== 路径 ==========
MEL_DIR  = "mel_features"
CSV_PATH = "processed_annotations.csv"
FEAT = {
    "angry": ("angry_feats.csv", ["median_dE_dt","mean_roughness_proxy"]),
    "happy": ("happy_feats.csv", ["p90_centroid","high_band_ratio"]),
    "sad":   ("sad_feats.csv",   ["low_band_ratio","energy_std"]),
    "relax": ("relax_feats.csv", ["p10_centroid","median_flux"]),
}
BEST_PTH = "best.pth"

# ========== 训练超参 ==========
BATCH = 32; LR = 1e-4; EPOCH = 50; PATIENCE = 5
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("▶ device:", dev)

# ========== 0 读数据 ==========
df = pd.read_csv(CSV_PATH)
feat_cols=[]
for tag,(csv_f,cols) in FEAT.items():
    if locals()[f"LOAD_{tag.upper()}"]:
        df = df.merge(pd.read_csv(csv_f), on="song_id", how="left")
        feat_cols += cols
print("▶ feat cols:", feat_cols)
assert df[feat_cols].notna().all().all(), "缺失特征！"

# 标签
le = LabelEncoder(); df["label_idx"]=le.fit_transform(df["label"])
CLS = list(le.classes_)

# genre one-hot
genres = sorted({g for row in df.genres for g in row.split(";")})
g2i = {g:i for i,g in enumerate(genres)}; N_GEN=len(genres)

# 固定验证集
split = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
tr_idx, va_idx = next(split.split(df, df.label_idx))
print(f"train={len(tr_idx)}  val={len(va_idx)}")

# ========= Dataset =========
class MusicDS(Dataset):
    def __init__(self, idxs, scaler):
        self.idxs, self.scaler = idxs, scaler
    def __len__(self): return len(self.idxs)
    def __getitem__(self,i):
        row=df.iloc[self.idxs[i]]
        mel=torch.from_numpy(np.load(os.path.join(MEL_DIR,f"{row.song_id}.npy")).astype(np.float32)).unsqueeze(0)
        vec=np.zeros(N_GEN,np.float32); gs=row.genres.split(';')
        w=(1/np.arange(1,len(gs)+1,dtype=np.float32)); w/=w.sum()
        for k,g in enumerate(gs): vec[g2i[g]]=w[k]
        genre=torch.from_numpy(vec)
        extra=torch.from_numpy(
            self.scaler.transform(pd.DataFrame([row[feat_cols].values],columns=feat_cols)).astype(np.float32).squeeze())
        label=torch.tensor(row.label_idx,dtype=torch.long)
        return mel,genre,extra,label

# ========= 网络 =========
class Block(nn.Module):
    def __init__(s,c1,c2,down=True):
        super().__init__()
        s.c1=nn.Conv2d(c1,c2,3,1,1);s.b1=nn.BatchNorm2d(c2)
        s.c2=nn.Conv2d(c2,c2,3,1,1);s.b2=nn.BatchNorm2d(c2)
        s.short=nn.Identity() if c1==c2 else nn.Conv2d(c1,c2,1)
        s.relu=nn.ReLU(); s.pool=nn.MaxPool2d(2) if down else nn.Identity()
    def forward(s,x):
        h=s.relu(s.b1(s.c1(x))); h=s.b2(s.c2(h))
        return s.pool(s.relu(h+s.short(x)))

class EmoNet(nn.Module):
    def __init__(s,n_gen,extra,n_cls):
        super().__init__()
        s.back=nn.Sequential(Block(1,16),Block(16,32),Block(32,64),Block(64,128,down=False))
        s.gap=nn.AdaptiveAvgPool2d((1,1)); s.gfc=nn.Linear(n_gen,32)
        s.cls=nn.Sequential(nn.Linear(128+32+extra,64),nn.ReLU(),nn.Dropout(0.3),nn.Linear(64,n_cls))
    def forward(s,m,g,e):
        x=s.gap(s.back(m)).view(m.size(0),-1); g=F.relu(s.gfc(g))
        return s.cls(torch.cat([x,g,e],1))

# ========= Loader & Scaler =========
scaler = StandardScaler().fit(df.loc[tr_idx, feat_cols])
tr_ds, va_ds = MusicDS(tr_idx,scaler), MusicDS(va_idx,scaler)
tr_ld = DataLoader(tr_ds,BATCH,shuffle=True); va_ld = DataLoader(va_ds,BATCH)

net=EmoNet(N_GEN,len(feat_cols),len(CLS)).to(dev)
opt=Adam(net.parameters(),LR); crit=nn.CrossEntropyLoss()

hist={"tr_loss":[],"va_loss":[],"tr_acc":[],"va_acc":[]}
best,wait=1e9,0

# ========= 训练循环 =========
for ep in range(1,EPOCH+1):
    net.train(); tl,tc=0,0
    for m,g,e,y in tr_ld:
        m,g,e,y=[t.to(dev) for t in (m,g,e,y)]
        opt.zero_grad(); out=net(m,g,e); loss=crit(out,y); loss.backward(); opt.step()
        tl+=loss.item()*y.size(0); tc+=(out.argmax(1)==y).sum().item()
    tr_loss, tr_acc = tl/len(tr_ds), tc/len(tr_ds)

    net.eval(); vl,vc=0,0; hit=[0]*len(CLS); cnt=[0]*len(CLS)
    with torch.no_grad():
        for m,g,e,y in va_ld:
            m,g,e,y=[t.to(dev) for t in (m,g,e,y)]
            out=net(m,g,e); loss=crit(out,y)
            vl+=loss.item()*y.size(0); p=out.argmax(1); vc+=(p==y).sum().item()
            for yi,pi in zip(y.cpu(),p.cpu()): cnt[yi]+=1; hit[yi]+=int(yi==pi)
    va_loss, va_acc = vl/len(va_ds), vc/len(va_ds)

    hist["tr_loss"].append(tr_loss); hist["va_loss"].append(va_loss)
    hist["tr_acc"].append(tr_acc);   hist["va_acc"].append(va_acc)

    per=" | ".join(f"{c}:{hit[i]}/{cnt[i]}={hit[i]/cnt[i]:.2f}"
                   for i,c in enumerate(CLS))
    print(f"Ep{ep:02d} tl={tr_loss:.3f} va={va_loss:.3f} acc={va_acc:.2f} || {per}")

    if va_loss < best:
        best = va_loss; wait = 0
        torch.save(net.state_dict(), BEST_PTH)
    else:
        wait += 1
        if USE_ES and wait>=PATIENCE:
            print("Early-Stop"); break

# ========= 评估最佳模型 → 混淆矩阵 =========
net.load_state_dict(torch.load(BEST_PTH))
net.eval(); y_true, y_pred = [], []
with torch.no_grad():
    for m,g,e,y in va_ld:
        m,g,e=[t.to(dev) for t in (m,g,e)]
        p=net(m,g,e).argmax(1).cpu().numpy()
        y_pred.extend(p); y_true.extend(y.numpy())

cm = confusion_matrix(y_true, y_pred, labels=range(len(CLS)))
disp = ConfusionMatrixDisplay(cm, display_labels=CLS)
disp.plot(cmap="Blues",values_format='d')
plt.title("Confusion Matrix (best model)")
plt.show()

# ========= Loss & Acc 曲线 =========
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
ax1.plot(hist["tr_loss"],label="train");ax1.plot(hist["va_loss"],label="val")
ax1.set_title("Loss");ax1.set_xlabel("epoch");ax1.legend();ax1.grid(True)
ax2.plot(hist["tr_acc"],label="train");ax2.plot(hist["va_acc"],label="val")
ax2.set_title("Accuracy");ax2.set_xlabel("epoch");ax2.legend();ax2.grid(True)
plt.tight_layout();plt.show()
