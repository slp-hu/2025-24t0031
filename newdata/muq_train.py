# ─────────────────── 1. 导入与配置 ───────────────────
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm 
import warnings
warnings.filterwarnings("ignore")

class Config:
    EMB_DIR  = r"G:\13kmid30s_muq"
    CSV_PATH = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_final_labeled_dataset.csv"
    BEST_MODEL_PATH = "best_model_muq_final.pth"
    ID_COL = "id"
    LABEL_COL = "final_label"
    GENRE_COL = "major_genre_name_en" 
    AUDIO_FEATURES = [
        "median_dE_dt", "mean_roughness_prx", "p90_centroid", "high_band_ratio",
        "low_band_ratio", "energy_std", "p10_centroid", "median_flux"
    ]
    EMB_DIM  = 1024
    VAL_SPLIT_SIZE = 0.2
    SEED         = 42
    BATCH_SIZE   = 64
    EPOCHS       = 50
    
    # =========================================================================
    # 【实验 D 配置】
    # =========================================================================
    MAX_LR       = 1e-3         
    WEIGHT_DECAY = 1e-4          
    DROPOUT_RATE = 0.2 # 保持温和的Dropout
    USE_CLASS_WEIGHTS = True   
    LABEL_SMOOTHING = 0.0       
    # =========================================================================
    
    NUM_WORKERS = 4
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 7

# ... (MelonDataset 定义保持不变) ...
class MelonDataset(Dataset):
    def __init__(self, dataframe, config, g2i):
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.g2i = g2i
        self.n_gen = len(g2i)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        embedding_id_str = str(row['embedding_id'])
        emb_path = os.path.join(self.config.EMB_DIR, f"{embedding_id_str}.npy")
        emb = np.load(emb_path).astype(np.float32)
        genre_vec = np.zeros(self.n_gen, np.float32)
        genre = row[self.config.GENRE_COL]
        if genre in self.g2i:
            genre_vec[self.g2i[genre]] = 1.0
        audio_feats = row[self.config.AUDIO_FEATURES].values.astype(np.float32)
        label = row.label_idx
        return torch.from_numpy(emb), torch.from_numpy(genre_vec), torch.from_numpy(audio_feats), torch.tensor(label, dtype=torch.long)

class Net(nn.Module):
    def __init__(self, d_emb, d_gen, d_ext, n_cls, dropout_rate):
        super().__init__()
        # 分支的Dropout率也需要传递
        self.proj_emb = nn.Sequential(nn.Linear(d_emb, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate))
        self.proj_gen = nn.Sequential(nn.Linear(d_gen, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout_rate))
        self.proj_ext = nn.Sequential(nn.BatchNorm1d(d_ext), nn.ReLU(), nn.Dropout(dropout_rate))
        
        # =========================================================================
        # 【核心修改】恢复为更深的两层Head，以增加模型容量
        # =========================================================================
        self.head = nn.Sequential(
            nn.Linear(128 + 32 + d_ext, 64), 
            nn.BatchNorm1d(64), 
            nn.ReLU(), 
            nn.Dropout(0.3), # 第一层的Dropout可以稍高

            nn.Linear(64, 32), 
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3), # 第二层的Dropout可以稍低

            nn.Linear(32, n_cls)
        )
        # =========================================================================

    def forward(self, emb, g, e):
        z = self.proj_emb(emb)
        g = self.proj_gen(g)
        e = self.proj_ext(e)
        return self.head(torch.cat([z, g, e], 1))

# =========================================================================
# 【主程序入口】
# =========================================================================
if __name__ == '__main__':
    # ... (数据加载、训练循环、评估可视化代码完全保持不变) ...
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    print(f"▶ 使用设备: {dev}")
    print("\n--- 正在加载和预处理数据 ---")
    df = pd.read_csv(Config.CSV_PATH)
    required_cols = [Config.ID_COL, Config.LABEL_COL, Config.GENRE_COL] + Config.AUDIO_FEATURES
    df.dropna(subset=required_cols, inplace=True)
    def get_first_id(id_string):
        try:
            cleaned_string = id_string.strip().strip('[]').strip()
            return cleaned_string.split(',')[0].strip() if cleaned_string else None
        except: return None
    df['embedding_id'] = df[Config.ID_COL].apply(get_first_id)
    df.dropna(subset=['embedding_id'], inplace=True)
    print("\n--- 正在校验Embedding文件是否存在... ---")
    df['emb_path'] = df['embedding_id'].apply(lambda x: os.path.join(Config.EMB_DIR, f"{x}.npy"))
    tqdm.pandas(desc="校验文件")
    df['file_exists'] = df['emb_path'].progress_apply(os.path.exists)
    df = df[df['file_exists']].copy()
    if df.empty: raise SystemExit("错误：校验后数据集为空。")
    print(f"最终用于训练的数据集大小: {len(df)} 条。")
    df.drop(columns=['emb_path', 'file_exists'], inplace=True)
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df[Config.LABEL_COL])
    CLASSES = list(le.classes_)
    print(f"发现 {len(CLASSES)} 个类别: {CLASSES}")
    genres = sorted(df[Config.GENRE_COL].unique())
    g2i = {g: i for i, g in enumerate(genres)}
    N_GEN = len(genres)
    print(f"发现 {N_GEN} 个流派。")
    split = StratifiedShuffleSplit(n_splits=1, test_size=Config.VAL_SPLIT_SIZE, random_state=Config.SEED)
    train_indices, val_indices = next(split.split(df, df.label_idx))
    df_train = df.iloc[train_indices]
    df_val = df.iloc[val_indices]
    print(f"数据集划分 -> 训练集: {len(df_train)}条 | 验证集: {len(df_val)}条")
    train_ds = MelonDataset(df_train, Config, g2i)
    val_ds = MelonDataset(df_val, Config, g2i)
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, pin_memory=True)
    net = Net(Config.EMB_DIM, N_GEN, len(Config.AUDIO_FEATURES), len(CLASSES), Config.DROPOUT_RATE).to(dev)
    opt = Adam(net.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY) 
    sched = OneCycleLR(opt, max_lr=Config.MAX_LR, steps_per_epoch=len(train_loader), epochs=Config.EPOCHS, pct_start=0.2)
    class_weights = None
    if Config.USE_CLASS_WEIGHTS:
        print("\n--- 正在计算类别权重以处理数据不平衡 ---")
        cls_counts = torch.bincount(torch.tensor(df_train['label_idx'].values))
        class_weights = 1. / cls_counts.float()
        class_weights = class_weights / class_weights.sum() * len(CLASSES)
        print("计算完成。")
    crit = nn.CrossEntropyLoss(weight=class_weights.to(dev) if class_weights is not None else None, label_smoothing=Config.LABEL_SMOOTHING)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float('inf')
    early_stop_counter = 0
    print("\n--- 开始训练 (实验D: 恢复两层Head, 保持正则化) ---")
    for ep in range(1, Config.EPOCHS + 1):
        net.train()
        total_loss = total_correct = 0
        for emb, g, e, y in train_loader:
            emb, g, e, y = emb.to(dev), g.to(dev), e.to(dev), y.to(dev)
            opt.zero_grad()
            out = net(emb, g, e)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * y.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
        train_loss = total_loss / len(train_ds)
        train_acc = total_correct / len(train_ds)
        net.eval()
        val_loss = val_correct = 0
        with torch.no_grad():
            for emb, g, e, y in val_loader:
                emb, g, e, y = emb.to(dev), g.to(dev), e.to(dev), y.to(dev)
                out = net(emb, g, e)
                val_loss += crit(out, y).item() * y.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc); history["val_acc"].append(val_acc)
        print(f"Epoch {ep:02d}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(net.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  -> 模型已保存，验证集损失降至: {best_val_loss:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"  -> 验证集损失未改善，早停计数: {early_stop_counter}/{Config.EARLY_STOPPING_PATIENCE}")
            if Config.USE_EARLY_STOPPING and early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"验证集损失连续 {Config.EARLY_STOPPING_PATIENCE} 个epoch未改善，触发早停。")
                break
    print("\n--- 开始评估最佳模型 ---")
    net.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for emb, g, e, y in val_loader:
            emb, g, e = emb.to(dev), g.to(dev), e.to(dev)
            y_pred.extend(net(emb, g, e).argmax(1).cpu().numpy())
            y_true.extend(y.numpy())
    print("\n" + "="*50)
    print("           Classification Report")
    print("="*50)
    report = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)
    print(report)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle("Training & Validation Curves", fontsize=16)
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epochs"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True)
    ax2.plot(history["train_acc"], label="Train Accuracy")
    ax2.plot(history["val_acc"], label="Validation Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epochs"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(True)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig2, (ax_cm, ax_report) = plt.subplots(1, 2, figsize=(20, 8), width_ratios=[1.2, 1])
    fig2.suptitle("Performance Analysis on Validation Set", fontsize=16)
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax_cm,
                                                   display_labels=CLASSES,
                                                   cmap='Blues', xticks_rotation=45)
    ax_cm.set_title("Confusion Matrix")
    ax_report.axis('off')
    ax_report.text(0.02, 0.98, report, family='monospace', size=11, va='top')
    ax_report.set_title("Metrics Details")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("\n实验完成。")