# ─────────────────── import ───────────────────
import os, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support, accuracy_score)
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

# ═══════════＝  超参配置 ＝═══════════
CSV_PATH = "processed_annotations.csv"
EMB_DIR  = "emb_full"
EMB_DIM  = 1024

FEATURES = {
    "angry": ("angry_feats_norm.csv",  ["median_dE_dt", "mean_roughness_proxy"]),
    "happy": ("happy_feats_norm.csv",  ["p90_centroid",  "high_band_ratio"]),
    "sad":   ("sad_feats_norm.csv",    ["low_band_ratio","energy_std"]),
    "relax": ("relax_feats_norm.csv",  ["p10_centroid",  "median_flux"]),
}

VAL_SIZE     = 78
SEED         = 42
BATCH        = 32
LR           = 1e-3           # OneCycle 的 max_lr 会覆盖这里
EPOCH        = 80
ES_PATIENCE  = 5
USE_ES       = False
BEST_PTH     = "best_muq_norm2.pth"
AMB_LABELS   = ["Ambiguous"]

# ═══════════＝  设备 ＝═══════════
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("▶ device:", dev)

# ═══════════＝  读取主表 ＝═══════════
df = pd.read_csv(CSV_PATH)

# 读取归一化特征
feat_cols = []
for csv_f, cols in FEATURES.values():
    fdf = pd.read_csv(csv_f)
    fdf["song_id"] = fdf["song_id"].astype(df.song_id.dtype)
    df = df.merge(fdf, on="song_id", how="left")
    feat_cols += cols

if df[feat_cols].isna().any(axis=None):
    miss = df[df[feat_cols].isna().any(axis=1)]
    print("⚠️ 缺特征示例:", miss.song_id.head().tolist())
    raise ValueError("请补全 *_norm.csv 中缺漏行。")

# 标签编码 & genre one-hot
le = LabelEncoder()
df["label_idx"] = le.fit_transform(df["label"])
CLS = list(le.classes_)
genres = sorted({g for row in df.genres for g in row.split(';')})
g2i = {g: i for i, g in enumerate(genres)}
N_GEN = len(genres)

# 固定验证集
split = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
tr_idx, va_idx = next(split.split(df, df.label_idx))
print(f"train={len(tr_idx)}  val={len(va_idx)}")

# ═══════════＝  Dataset ＝═══════════
class MuQDS(Dataset):
    def __init__(self, idxs): self.idxs = idxs
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        row = df.iloc[self.idxs[i]]
        emb   = torch.from_numpy(np.load(os.path.join(EMB_DIR, f"{row.song_id}.npy")).astype(np.float32))
        vec   = np.zeros(N_GEN, np.float32)
        gs    = row.genres.split(';')
        w     = 1 / np.arange(1, len(gs)+1); w /= w.sum()
        for k, g in enumerate(gs): vec[g2i[g]] = w[k]
        extra = torch.from_numpy(row[feat_cols].values.astype(np.float32))
        label = torch.tensor(row.label_idx, dtype=torch.long)
        return emb, torch.from_numpy(vec), extra, label

tr_ds, va_ds = MuQDS(tr_idx), MuQDS(va_idx)
tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True)
va_ld = DataLoader(va_ds, batch_size=BATCH)

# ═══════════＝  模型 ＝═══════════
class Net(nn.Module):
    def __init__(self, d_emb, d_gen, d_ext, n_cls):
        super().__init__()

        # ── Projection layers for each input ──
        self.proj_emb = nn.Sequential(
            nn.Linear(d_emb, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.7)
        )

        self.proj_gen = nn.Sequential(
            nn.Linear(d_gen, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.7)
        )

        self.bn_ext = nn.BatchNorm1d(d_ext)
        self.dp_ext = nn.Dropout(0.7)

        # ── Head layers ──
        self.head = nn.Sequential(
            nn.Linear(128 + 32 + d_ext, 64),  # 第一层 Head（融合层）
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.7),

            nn.Linear(64, 32),               # 第二层 Head（新增）
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(32, n_cls)             # 输出层
        )
    def forward(self, emb, g, e):
        z = self.proj_emb(emb)
        g = self.proj_gen(g)
        e = self.dp_ext(F.relu(self.bn_ext(e)))
        return self.head(torch.cat([z, g, e], 1))

net = Net(EMB_DIM, N_GEN, len(feat_cols), len(CLS)).to(dev)
opt = Adam(net.parameters(), lr=LR, weight_decay=1e-4)

# ─── OneCycleLR ───
sched = OneCycleLR(
    opt, max_lr=1e-3,
    steps_per_epoch=len(tr_ld), epochs=EPOCH,
    pct_start=0.15, div_factor=50, final_div_factor=1e4,
    anneal_strategy='cos')

# >>> ① 计算类别权重（可选，不想用就注释掉下面三行）
cls_cnt = torch.bincount(torch.tensor(df.label_idx.values))
weights = (1. / cls_cnt.float()).to(dev)

# >>> ② 换成带 Label Smoothing 的 Loss
crit = nn.CrossEntropyLoss(
    weight=None,          # 若不想用权重可写成 None
    label_smoothing=0.10)    # ← 平滑系数 0.10

# ═══════════＝  训练 ＝═══════════
hist = {k: [] for k in ["tr_loss", "va_loss", "tr_acc", "va_acc", "lr"]}
best = float('inf'); wait = 0

for ep in range(1, EPOCH + 1):
    net.train(); tl = tc = 0
    for emb, g, e, y in tr_ld:
        emb, g, e, y = [t.to(dev) for t in (emb, g, e, y)]
        opt.zero_grad()
        out  = net(emb, g, e)
        loss = crit(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step(); sched.step()

        tl += loss.item() * y.size(0)
        tc += (out.argmax(1) == y).sum().item()
    tr_loss, tr_acc = tl / len(tr_ds), tc / len(tr_ds)

    net.eval(); vl = vc = 0
    with torch.no_grad():
        for emb, g, e, y in va_ld:
            emb, g, e, y = [t.to(dev) for t in (emb, g, e, y)]
            out = net(emb, g, e)
            vl += crit(out, y).item() * y.size(0)
            vc += (out.argmax(1) == y).sum().item()
    va_loss, va_acc = vl / len(va_ds), vc / len(va_ds)

    hist["tr_loss"].append(tr_loss); hist["va_loss"].append(va_loss)
    hist["tr_acc"].append(tr_acc);   hist["va_acc"].append(va_acc)
    hist["lr"].append(opt.param_groups[0]['lr'])

    print(f"Ep{ep:02d} │ loss {tr_loss:.3f}/{va_loss:.3f} │ "
          f"acc {tr_acc:.3f}/{va_acc:.3f} │ lr {opt.param_groups[0]['lr']:.2e}")

    if va_loss < best:
        best = va_loss; wait = 0
        torch.save(net.state_dict(), BEST_PTH)
    else:
        wait += 1
        if USE_ES and wait >= ES_PATIENCE:
            print("Early stop 🌙"); break

# ═══════════＝  评估 & 可视化 ＝═══════════
net.load_state_dict(torch.load(BEST_PTH)); net.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for emb, g, e, y in va_ld:
        emb, g, e = [t.to(dev) for t in (emb, g, e)]
        y_pred.extend(net(emb, g, e).argmax(1).cpu())
        y_true.extend(y)

def macro(yt, yp):
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
    return p, r, f1, accuracy_score(yt, yp)

p_all, r_all, f1_all, acc_all = macro(y_true, y_pred)
if AMB_LABELS:
    amb_idx = [CLS.index(l) for l in AMB_LABELS]
    keep = [lbl not in amb_idx for lbl in y_true]
    p_n, r_n, f1_n, acc_n = macro(np.array(y_true)[keep], np.array(y_pred)[keep])
else:
    p_n = r_n = f1_n = acc_n = np.nan

# ① Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLS))))
fig_cm = plt.figure(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=CLS).plot(
    cmap="Blues", values_format='d', ax=fig_cm.gca())
fig_cm.suptitle("Confusion Matrix (all)")

# ② Loss & Accuracy 曲线
fig_curve, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(hist["tr_loss"], label="train"); ax1.plot(hist["va_loss"], label="val")
ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend(); ax1.grid(True)
ax2.plot(hist["tr_acc"], label="train"); ax2.plot(hist["va_acc"], label="val")
ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend(); ax2.grid(True)
fig_curve.suptitle("Training Curves")

# ③ 指标摘要
fig_txt = plt.figure(figsize=(6, 3))
txt = (f"[all]          Macro-P={p_all:.3f}  R={r_all:.3f}  F1={f1_all:.3f}  Acc={acc_all:.3f}\n"
       f"[no-ambiguous] Macro-P={p_n:.3f}  R={r_n:.3f}  F1={f1_n:.3f}  Acc={acc_n:.3f}")
fig_txt.text(0.01, 0.5, txt, font="monospace", fontsize=11, va="center")
fig_txt.gca().axis("off"); fig_txt.suptitle("Metric Summary")

plt.show()
