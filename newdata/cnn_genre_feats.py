#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_onecycle_cm_focal.py  (Windows-safe, picklable classes)
--------------------------------------------------
• 80/20 Stratified split
• OneCycleLR 调度
• WeightedRandomSampler + FocalLoss
• 四处可调 Dropout (CNN / genre / extra / fusion，双层融合头)
• 支持可变长度 mel (padding)
• Windows 下多进程 DataLoader => 需要 Dataset / collate 在全局作用域
"""

import os, random, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support, accuracy_score)
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

# ------------------------ 可调超参 ------------------------
SEED       = 42
VAL_RATIO  = 0.2
BATCH      = 64
EPOCH      = 50
MAX_LR     = 1e-3
PCT_START  = 0.15
BEST_PTH   = "best.pth"

# —— Dropout rates —— #
CNN_DP   = 0.1   # 卷积骨干
GENRE_DP = 0.1   # genre 头
EXTRA_DP = 0.1   # extra 8-feat 头
FUSE_DP  = 0.30   # 融合层 (两处)

# ------------------------ 路径 & 列名 ------------------------
MEL_DIR  = r"E:\melon\extracted"
CSV_PATH = r"C:\Users\YAO\Desktop\genre ml\joyful_melon_intersection_v2_with8feats.csv"

FEAT_COLS = ["median_dE_dt","mean_roughness_prx","p90_centroid","high_band_ratio",
             "low_band_ratio","energy_std","p10_centroid","median_flux"]
GENRE_COL = "major_genre_name_en"
ID_COL    = "melon_id"

# ----------------------- Focal Loss -----------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, logits, target):
        logpt = F.log_softmax(logits, dim=1)
        pt    = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt    = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(0, target)
            logpt = logpt * at
        loss = - (1-pt) ** self.gamma * logpt
        return loss.mean() if self.reduction=='mean' else loss.sum()

# ----------------------- Dataset (全局可 pickle) ---------- #
class MusicDS(Dataset):
    def __init__(self, df, idxs, scaler, g2i, n_gen):
        self.df, self.idxs, self.scaler, self.g2i, self.n_gen = df, idxs, scaler, g2i, n_gen
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        row = self.df.iloc[self.idxs[i]]
        mel = torch.from_numpy(np.load(os.path.join(MEL_DIR,f"{row[ID_COL]}.npy"))
                               .astype(np.float32)).unsqueeze(0)
        gvec = np.zeros(self.n_gen, np.float32); gvec[self.g2i[row[GENRE_COL]]] = 1
        extra = self.scaler.transform(row[FEAT_COLS].values.reshape(1,-1)).astype(np.float32).squeeze()
        return mel, torch.from_numpy(gvec), torch.from_numpy(extra), torch.tensor(row.label_idx)

# ----------------------- collate_fn (全局) -----------------
def pad_collate(batch):
    mels,gs,es,ys = zip(*batch)
    T = max(m.shape[2] for m in mels)
    mels = torch.stack([ F.pad(m,(0,T-m.shape[2])) for m in mels ])
    return mels, torch.stack(gs), torch.stack(es), torch.tensor(ys)

# ----------------------- 网络模块 ------------------------
class Block(nn.Module):
    def __init__(self, c1, c2, down=True, dp=CNN_DP):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(c1,c2,3,1,1), nn.BatchNorm2d(c2)
        self.conv2, self.bn2 = nn.Conv2d(c2,c2,3,1,1), nn.BatchNorm2d(c2)
        self.short = nn.Identity() if c1==c2 else nn.Conv2d(c1,c2,1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2) if down else nn.Identity()
        self.dp2d  = nn.Dropout2d(dp) if dp>0 else nn.Identity()
    def forward(self,x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.relu(h + self.short(x))
        return self.dp2d(self.pool(h))

class EmoNet(nn.Module):
    def __init__(self, n_gen, extra_dim, n_cls):
        super().__init__()
        self.back = nn.Sequential(
            Block(1,16), Block(16,32), Block(32,64), Block(64,128,down=False)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.gproj    = nn.Linear(n_gen, 32)
        self.dp_genre = nn.Dropout(GENRE_DP) if GENRE_DP>0 else nn.Identity()
        self.dp_extra = nn.Dropout(EXTRA_DP) if EXTRA_DP>0 else nn.Identity()

        self.cls = nn.Sequential(                      # 双层融合头
            nn.Linear(128+32+extra_dim, 64),
            nn.ReLU(),
            nn.Dropout(FUSE_DP) if FUSE_DP>0 else nn.Identity(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(FUSE_DP) if FUSE_DP>0 else nn.Identity(),
            nn.Linear(32, n_cls)
        )
    def forward(self, m, g, e):
        x = self.gap(self.back(m)).view(m.size(0), -1)
        g = self.dp_genre(F.relu(self.gproj(g)))
        e = self.dp_extra(e)
        return self.cls(torch.cat([x,g,e],dim=1))

# ----------------------- 主逻辑 -------------------------
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(CSV_PATH)
    df["label"] = df["emo_tag"]
    le = LabelEncoder(); df["label_idx"] = le.fit_transform(df["label"]); CLS = list(le.classes_)
    df[GENRE_COL] = df[GENRE_COL].fillna("Unknown").astype(str)
    genres = sorted(df[GENRE_COL].unique()); g2i = {g:i for i,g in enumerate(genres)}; N_GEN=len(genres)

    split = StratifiedShuffleSplit(n_splits=1,test_size=VAL_RATIO,random_state=SEED)
    tr_idx, va_idx = next(split.split(df, df.label_idx))
    print(f"train={len(tr_idx)}  val={len(va_idx)}")

    scaler = StandardScaler().fit(df.loc[tr_idx,FEAT_COLS].values)
    tr_ds = MusicDS(df,tr_idx,scaler,g2i,N_GEN)
    va_ds = MusicDS(df,va_idx,scaler,g2i,N_GEN)

    counts = df.loc[tr_idx,"label_idx"].value_counts().sort_index().values
    weights = 1./torch.tensor(counts,dtype=torch.float)
    sampler = WeightedRandomSampler(weights[df.loc[tr_idx,"label_idx"].values],
                                    len(tr_idx),replacement=True)

    tr_ld = DataLoader(tr_ds,BATCH,sampler=sampler,collate_fn=pad_collate,
                       num_workers=4,pin_memory=True)
    va_ld = DataLoader(va_ds,BATCH,shuffle=False,collate_fn=pad_collate,
                       num_workers=4,pin_memory=True)

    alpha = torch.tensor(1./counts); alpha = alpha/alpha.sum()*len(counts)
    net = EmoNet(N_GEN,len(FEAT_COLS),len(CLS)).to(dev)
    crit=FocalLoss(alpha=None,gamma=2.0)
    opt = Adam(net.parameters(),lr=MAX_LR/25)
    sch = OneCycleLR(opt,max_lr=MAX_LR,steps_per_epoch=len(tr_ld),epochs=EPOCH,
                     pct_start=PCT_START,div_factor=50,final_div_factor=1e4,
                     anneal_strategy='cos')

    hist,best={"tr_loss":[],"va_loss":[],"tr_acc":[],"va_acc":[]},float('inf')

    for ep in range(1,EPOCH+1):
        net.train(); tl=tc=0
        for m,g,e,y in tr_ld:
            m,g,e,y=[t.to(dev) for t in (m,g,e,y)]
            opt.zero_grad(); loss=crit(net(m,g,e),y)
            loss.backward(); opt.step(); sch.step()
            tl+=loss.item()*y.size(0); tc+=(net(m,g,e).argmax(1)==y).sum().item()
        tr_loss, tr_acc = tl/len(tr_ds), tc/len(tr_ds)

        net.eval(); vl=vc=0
        with torch.no_grad():
            for m,g,e,y in va_ld:
                m,g,e,y=[t.to(dev) for t in (m,g,e,y)]
                out=net(m,g,e); vl+=crit(out,y).item()*y.size(0); vc+=(out.argmax(1)==y).sum().item()
        va_loss, va_acc = vl/len(va_ds), vc/len(va_ds)

        hist["tr_loss"].append(tr_loss); hist["va_loss"].append(va_loss)
        hist["tr_acc"].append(tr_acc);   hist["va_acc"].append(va_acc)
        print(f"Ep{ep:02d} tr_loss={tr_loss:.3f} va_loss={va_loss:.3f} "
              f"tr_acc={tr_acc:.3f} va_acc={va_acc:.3f}")

        if va_loss<best: best=va_loss; torch.save(net.state_dict(),BEST_PTH)

    # ------- Eval / Plot -------
    net.load_state_dict(torch.load(BEST_PTH)); net.eval()
    y_true=y_pred=[]
    with torch.no_grad():
        for m,g,e,y in va_ld:
            m,g,e=[t.to(dev) for t in (m,g,e)]
            y_pred.extend(net(m,g,e).argmax(1).cpu()); y_true.extend(y)
    def macro(a,b):
        p,r,f,_=precision_recall_fscore_support(a,b,average='macro',zero_division=0)
        return p,r,f,accuracy_score(a,b)
    p_all,r_all,f1_all,acc_all = macro(y_true,y_pred)
    amb_idx=[CLS.index("Ambiguous")] if "Ambiguous" in CLS else []
    keep=[lbl not in amb_idx for lbl in y_true]
    p_n,r_n,f1_n,acc_n = macro(np.array(y_true)[keep],np.array(y_pred)[keep])

    cm=confusion_matrix(y_true,y_pred,labels=list(range(len(CLS))))
    ConfusionMatrixDisplay(cm,display_labels=CLS).plot(cmap="Blues",values_format='d')
    plt.title("Confusion Matrix")
    plt.figure(); plt.plot(hist["tr_loss"],label="tr_loss"); plt.plot(hist["va_loss"],label="va_loss")
    plt.plot(hist["tr_acc"],label="tr_acc"); plt.plot(hist["va_acc"],label="va_acc"); plt.legend(); plt.grid()
    plt.title("Training Curves")
    plt.figure(figsize=(6,3)); txt=(f"[all] MacroF1={f1_all:.3f} Acc={acc_all:.3f}\n"
                                    f"[no-amb] MacroF1={f1_n:.3f} Acc={acc_n:.3f}")
    plt.text(0.01,0.5,txt,fontsize=11,family="monospace"); plt.axis("off"); plt.title("Metric Summary")
    plt.show()

# ----------------------- Windows 入口 ---------------------
if __name__ == "__main__":
    main()
