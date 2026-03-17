#!/usr/bin/env python3
# coding: utf-8
"""
train_light_aug_scaled.py  (auto-compatible AMP)
------------------------------------------------
• 8 个情感特征已预标准化
• 删除激进增广 (_ps±2, _ts1.05)
• 打印每个 epoch 的 tr/val Loss/Acc 和当前 LR
• 双窗口：混淆矩阵 & Loss/Acc 曲线
"""

import os, re, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ====== 自动选择 AMP API (兼容新旧 PyTorch) ======
try:
    from torch.amp import autocast as _new_autocast, GradScaler as _new_scaler
    def autocast():
        return _new_autocast(device_type='cuda')        # 新 API (≥2.1)
    GradScaler = _new_scaler
except ImportError:                                     # 旧版本 fallback
    from torch.cuda.amp import autocast, GradScaler

# ────────── 常量 ──────────
SEED, VAL_SIZE = 42, 78
BATCH, LR, EPOCH, PATIENCE = 32, 1e-4, 30, 3
USE_ES = False

CSV_BASE = "processed_annotations.csv"
CSV_AUG  = "processed_annotations_aug.csv"
MEL_DIR  = "mel_features_aug"
BEST_PTH = "best_scaled.pth"

FEAT = {
    "angry": ("angry_feats_aug_norm.csv", ["median_dE_dt", "mean_roughness_proxy"]),
    "happy": ("happy_feats_aug_norm.csv", ["p90_centroid", "high_band_ratio"]),
    "sad":   ("sad_feats_aug_norm.csv",   ["low_band_ratio", "energy_std"]),
    "relax": ("relax_feats_aug_norm.csv", ["p10_centroid", "median_flux"]),
}
LOAD_ANGRY = LOAD_HAPPY = LOAD_SAD = LOAD_RELAX = True
# ──────────────────────────


# ========= 网络 =========
class Block(nn.Module):
    def __init__(self,c1,c2,down=True):
        super().__init__()
        self.c1,self.b1 = nn.Conv2d(c1,c2,3,1,1), nn.BatchNorm2d(c2)
        self.c2,self.b2 = nn.Conv2d(c2,c2,3,1,1), nn.BatchNorm2d(c2)
        self.short = nn.Identity() if c1==c2 else nn.Conv2d(c1,c2,1)
        self.relu,self.pool = nn.ReLU(), (nn.MaxPool2d(2) if down else nn.Identity())
    def forward(self,x):
        h=self.relu(self.b1(self.c1(x))); h=self.b2(self.c2(h))
        return self.pool(self.relu(h+self.short(x)))

class EmoNet(nn.Module):
    def __init__(self,n_gen,extra_dim,n_cls):
        super().__init__()
        self.back = nn.Sequential(Block(1,16),Block(16,32),Block(32,64),Block(64,128,False))
        self.gap  = nn.AdaptiveAvgPool2d((1,1))
        self.bn_mel,self.d_mel = nn.BatchNorm1d(128), nn.Dropout(0.3)

        self.fc_gen  = nn.Linear(n_gen,32)
        self.bn_gen,self.d_gen = nn.BatchNorm1d(32), nn.Dropout(0.2)

        self.bn_ex,self.d_ex = nn.BatchNorm1d(extra_dim), nn.Dropout(0.2)

        hid=64
        self.head = nn.Sequential(
            nn.Linear(128+32+extra_dim,hid), nn.BatchNorm1d(hid),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid,n_cls)
        )
    def forward(self,m,g,e):
        B=m.size(0)
        x=self.d_mel(F.relu(self.bn_mel(self.gap(self.back(m)).view(B,-1))))
        g=self.d_gen(F.relu(self.bn_gen(F.relu(self.fc_gen(g)))))
        e=self.d_ex(F.relu(self.bn_ex(e)))
        return self.head(torch.cat([x,g,e],1))

# ========= Dataset =========
class MusicDS(Dataset):
    def __init__(self, df, idxs, feat_cols, g2i, n_gen):
        self.df,self.idxs = df, pd.Index(idxs)
        self.feat_cols,self.g2i,self.n_gen = feat_cols,g2i,n_gen
    def __len__(self): return len(self.idxs)
    def __getitem__(self,i):
        row = self.df.iloc[self.idxs[i]]
        mel = torch.from_numpy(
            np.load(os.path.join(MEL_DIR,f"{row.song_id}.npy"))
        ).unsqueeze(0).float()
        gs=row.genres.split(";")
        w=1.0/np.arange(1,len(gs)+1,dtype=np.float32); w/=w.sum()
        vec=np.zeros(self.n_gen,np.float32)
        for k,g in enumerate(gs): vec[self.g2i[g]] = w[k]
        extra=torch.from_numpy(row[self.feat_cols].values.astype(np.float32))
        return mel, torch.from_numpy(vec), extra, torch.tensor(row.label_idx)

# ========= 主流程 =========
def main():
    np.random.seed(SEED); torch.manual_seed(SEED)

    # 0 读取 & 过滤
    df_base = pd.read_csv(CSV_BASE)
    df_aug  = pd.read_csv(CSV_AUG)
    df_base["song_id"] = df_base["song_id"].astype(str)
    df_aug ["song_id"] = df_aug ["song_id"].astype(str)
    df_aug["base_id"]  = df_aug["song_id"].str.replace(r"_.*$", "", regex=True)
    df_aug = df_aug[~df_aug["song_id"].str.contains(r"_ps[+-]2|_ts1\.05")].reset_index(drop=True)

    # 固定验证
    sss = StratifiedShuffleSplit(1,test_size=VAL_SIZE,random_state=SEED)
    _, va_idx_390 = next(sss.split(df_base,df_base["label"]))
    val_base_ids = set(df_base.loc[va_idx_390,"song_id"])

    # 合并已归一化特征
    feat_cols=[]
    for tag,(csv_f,cols) in FEAT.items():
        if globals()[f"LOAD_{tag.upper()}"]:
            df_aug = df_aug.merge(pd.read_csv(csv_f), on="song_id", how="left")
            feat_cols += cols
    df_aug = df_aug[df_aug[feat_cols].notna().all(axis=1)].reset_index(drop=True)

    is_val = (df_aug["base_id"].isin(val_base_ids)) & (df_aug["song_id"]==df_aug["base_id"])
    va_idx = df_aug[is_val].index
    tr_idx = df_aug[~df_aug["base_id"].isin(val_base_ids)].index
    print(f"train = {len(tr_idx)},  val = {len(va_idx)}")

    df = df_aug
    le=LabelEncoder(); df["label_idx"]=le.fit_transform(df["label"]); CLS=list(le.classes_)
    genres=sorted({g for row in df.genres for g in row.split(";")}); g2i={g:i for i,g in enumerate(genres)}
    N_GEN=len(genres)

    tr_ds=MusicDS(df,tr_idx,feat_cols,g2i,N_GEN)
    va_ds=MusicDS(df,va_idx,feat_cols,g2i,N_GEN)
    tr_ld=DataLoader(tr_ds,batch_size=BATCH,shuffle=True,num_workers=10,
                     pin_memory=True,persistent_workers=True)
    va_ld=DataLoader(va_ds,batch_size=BATCH,num_workers=4,pin_memory=True)

    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=EmoNet(N_GEN,len(feat_cols),len(CLS)).to(dev)
    opt=Adam(net.parameters(),lr=LR,weight_decay=1e-4)
    sched=ReduceLROnPlateau(opt,'min',factor=0.5,patience=PATIENCE)
    crit=nn.CrossEntropyLoss()
    scaler=GradScaler()

    hist={"trL":[],"vaL":[],"trA":[],"vaA":[]}
    best=float('inf'); wait=0
    for ep in range(1,EPOCH+1):
        # Train
        net.train(); tl=tc=0
        for m,g,e,y in tr_ld:
            m,g,e,y=[t.to(dev,non_blocking=True) for t in (m,g,e,y)]
            opt.zero_grad()
            with autocast():
                out=net(m,g,e); loss=crit(out,y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl+=loss.item()*y.size(0); tc+=(out.argmax(1)==y).sum().item()
        trL,trA = tl/len(tr_ds), tc/len(tr_ds)

        # Val
        net.eval(); vl=vc=0
        with torch.no_grad(), autocast():
            for m,g,e,y in va_ld:
                m,g,e,y=[t.to(dev) for t in (m,g,e,y)]
                out=net(m,g,e); vl+=crit(out,y).item()*y.size(0); vc+=(out.argmax(1)==y).sum().item()
        vaL,vaA = vl/len(va_ds), vc/len(va_ds)

        hist["trL"].append(trL); hist["vaL"].append(vaL)
        hist["trA"].append(trA); hist["vaA"].append(vaA)
        current_lr = opt.param_groups[0]['lr']
        print(f"Ep{ep:02d}  trL={trL:.3f}  vaL={vaL:.3f}  trA={trA:.3f}  vaA={vaA:.3f}  lr={current_lr:.2e}")

        sched.step(vaL)
        if vaL<best:
            best,wait=vaL,0; torch.save(net.state_dict(),BEST_PTH)
        else:
            wait+=1
            if USE_ES and wait>=PATIENCE:
                print("Early stop"); break

    # 评估 & 绘图
    net.load_state_dict(torch.load(BEST_PTH)); net.eval()
    y_true,y_pred=[],[]
    with torch.no_grad(), autocast():
        for m,g,e,y in va_ld:
            m,g,e=[t.to(dev) for t in (m,g,e)]
            y_pred.extend(net(m,g,e).argmax(1).cpu().numpy()); y_true.extend(y.numpy())
    cm=confusion_matrix(y_true,y_pred,labels=range(len(CLS)))
    ConfusionMatrixDisplay(cm,display_labels=CLS).plot(cmap="Blues",values_format='d')
    plt.title("Confusion Matrix"); plt.show()

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.plot(hist["trL"],label="train"); plt.plot(hist["vaL"],label="val")
    plt.title("Loss"); plt.grid(); plt.legend()
    plt.subplot(1,2,2); plt.plot(hist["trA"],label="train"); plt.plot(hist["vaA"],label="val")
    plt.title("Accuracy"); plt.grid(); plt.legend()
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
