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
    CSV_PATH = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_final_labeled_dataset.csv"
    MUQ_EMB_DIR = r"G:\13kmid30s_muq"
    MEL_SPEC_DIR = r"E:\melon\extracted" 
    BEST_MODEL_PATH = "best_model_muq_mel_2layer_head.pth" # 新模型用新名字
    MUQ_ID_COL = "id" 
    MEL_ID_COL = "melon_id"
    LABEL_COL = "final_label"
    GENRE_COL = "major_genre_name_en"
    AUDIO_FEATURES = [
        "median_dE_dt", "mean_roughness_prx", "p90_centroid", "high_band_ratio",
        "low_band_ratio", "energy_std", "p10_centroid", "median_flux"
    ]
    MUQ_EMB_DIM = 1024
    VAL_SPLIT_SIZE = 0.2
    SEED         = 42
    BATCH_SIZE   = 64
    EPOCHS       = 50
    MAX_LR       = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.3
    USE_CLASS_WEIGHTS = True
    LABEL_SMOOTHING = 0.0
    NUM_WORKERS = 4
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 7

# ═══════════＝ 2. 自定义 Collate Function ＝═══════════
def pad_collate_fn(batch):
    muqs, mels, genres, feats, labels = zip(*batch)
    max_len = max(mel.shape[2] for mel in mels)
    padded_mels = []
    for mel in mels:
        pad_width = max_len - mel.shape[2]
        padded_mels.append(F.pad(mel, (0, pad_width), "constant", 0))
    return torch.stack(muqs), torch.stack(padded_mels), torch.stack(genres), torch.stack(feats), torch.stack(labels)

# ═══════════＝ 3. 模型与Dataset定义 ＝═══════════
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
        muq_id_str = str(row['muq_embedding_id'])
        muq_path = os.path.join(self.config.MUQ_EMB_DIR, f"{muq_id_str}.npy")
        muq_emb = np.load(muq_path).astype(np.float32)
        mel_id_str = str(row['mel_embedding_id'])
        mel_path = os.path.join(self.config.MEL_SPEC_DIR, f"{mel_id_str}.npy")
        mel_spec = np.load(mel_path).astype(np.float32)
        mel_spec = np.expand_dims(mel_spec, axis=0)
        genre_vec = np.zeros(self.n_gen, np.float32)
        genre = row[self.config.GENRE_COL]
        if genre in self.g2i: genre_vec[self.g2i[genre]] = 1.0
        audio_feats = row[self.config.AUDIO_FEATURES].values.astype(np.float32)
        label = row.label_idx
        return torch.from_numpy(muq_emb), torch.from_numpy(mel_spec), torch.from_numpy(genre_vec), torch.from_numpy(audio_feats), torch.tensor(label, dtype=torch.long)

class MelCNN(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding='same'), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 32, 3, padding='same'), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 4)))
        self.conv_block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding='same'), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 64, 3, padding='same'), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 4)))
        self.conv_block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding='same'), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 128, 3, padding='same'), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((3, 5)))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        x = self.conv_block1(x); x = self.conv_block2(x); x = self.conv_block3(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        return torch.flatten(x, 1)

class Net(nn.Module):
    def __init__(self, d_muq_emb, d_gen, d_audio_feats, n_cls, dropout_rate):
        super().__init__()
        self.muq_branch = nn.Sequential(nn.Linear(d_muq_emb, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate))
        self.mel_cnn_branch = MelCNN(dropout_rate)
        self.genre_branch = nn.Sequential(nn.Linear(d_gen, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout_rate))
        self.audio_feat_branch = nn.Sequential(nn.Linear(d_audio_feats, 16), nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout_rate))

        # =========================================================================
        # 【核心修改】将融合模型的Head也加深为两层
        # =========================================================================
        self.head = nn.Sequential(
            nn.Linear(128 + 128 + 32 + 16, 128), # 第一层融合
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(0.3),

            nn.Linear(128, 64), # 第二层
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, n_cls) # 输出层
        )
        # =========================================================================

    def forward(self, muq, mel, genre, feats):
        muq_z = self.muq_branch(muq)
        mel_z = self.mel_cnn_branch(mel)
        genre_z = self.genre_branch(genre)
        feat_z = self.audio_feat_branch(feats)
        combined_z = torch.cat([muq_z, mel_z, genre_z, feat_z], 1)
        return self.head(combined_z)

# =========================================================================
# 【主程序入口】
# =========================================================================
if __name__ == '__main__':
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(Config.SEED)
    print(f"▶ 使用设备: {dev}")

    df = pd.read_csv(Config.CSV_PATH)
    required_cols = [Config.MUQ_ID_COL, Config.MEL_ID_COL, Config.LABEL_COL, Config.GENRE_COL] + Config.AUDIO_FEATURES
    df.dropna(subset=required_cols, inplace=True)
    
    def get_first_id(id_string):
        try:
            cleaned_string = id_string.strip().strip('[]').strip()
            return cleaned_string.split(',')[0].strip() if cleaned_string else None
        except: return None
    df['muq_embedding_id'] = df[Config.MUQ_ID_COL].apply(get_first_id)
    df.dropna(subset=['muq_embedding_id'], inplace=True)
    df['mel_embedding_id'] = df[Config.MEL_ID_COL].astype(int).astype(str)

    print("\n--- 正在校验 MuQ 和 Mel 文件是否存在... ---")
    df['muq_path'] = df['muq_embedding_id'].apply(lambda x: os.path.join(Config.MUQ_EMB_DIR, f"{x}.npy"))
    df['mel_path'] = df['mel_embedding_id'].apply(lambda x: os.path.join(Config.MEL_SPEC_DIR, f"{x}.npy"))
    tqdm.pandas(desc="校验文件")
    df['muq_exists'] = df['muq_path'].progress_apply(os.path.exists)
    df['mel_exists'] = df['mel_path'].progress_apply(os.path.exists)
    df = df[df['muq_exists'] & df['mel_exists']].copy()
    if df.empty: raise SystemExit("错误：校验后数据集为空。")
    print(f"最终用于训练的数据集大小: {len(df)} 条。")
    df.drop(columns=['muq_path', 'mel_path', 'muq_exists', 'mel_exists'], inplace=True)

    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df[Config.LABEL_COL])
    CLASSES = list(le.classes_)
    genres = sorted(df[Config.GENRE_COL].unique())
    g2i = {g: i for i, g in enumerate(genres)}
    N_GEN = len(genres)
    split = StratifiedShuffleSplit(n_splits=1, test_size=Config.VAL_SPLIT_SIZE, random_state=Config.SEED)
    train_indices, val_indices = next(split.split(df, df.label_idx))
    df_train = df.iloc[train_indices]
    df_val = df.iloc[val_indices]
    
    train_ds = MelonDataset(df_train, Config, g2i)
    val_ds = MelonDataset(df_val, Config, g2i)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True, collate_fn=pad_collate_fn)
    
    net = Net(Config.MUQ_EMB_DIM, N_GEN, len(Config.AUDIO_FEATURES), len(CLASSES), Config.DROPOUT_RATE).to(dev)
    opt = Adam(net.parameters(), lr=Config.MAX_LR, weight_decay=Config.WEIGHT_DECAY) 
    sched = OneCycleLR(opt, max_lr=Config.MAX_LR, steps_per_epoch=len(train_loader), epochs=Config.EPOCHS, pct_start=0.2)
    
    class_weights = None
    if Config.USE_CLASS_WEIGHTS:
        cls_counts = torch.bincount(torch.tensor(df_train['label_idx'].values))
        class_weights = 1. / cls_counts.float()
        class_weights = class_weights / class_weights.sum() * len(CLASSES)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(dev) if class_weights is not None else None, 
                              label_smoothing=Config.LABEL_SMOOTHING)
                              
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float('inf')
    early_stop_counter = 0
    print("\n--- 开始训练 (最终融合模型, 2-Layer Head) ---")
    for ep in range(1, Config.EPOCHS + 1):
        net.train()
        total_loss = total_correct = 0
        for muq, mel, genre, feats, y in train_loader:
            muq, mel, genre, feats, y = muq.to(dev), mel.to(dev), genre.to(dev), feats.to(dev), y.to(dev)
            opt.zero_grad()
            out = net(muq, mel, genre, feats)
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
            for muq, mel, genre, feats, y in val_loader:
                muq, mel, genre, feats, y = muq.to(dev), mel.to(dev), genre.to(dev), feats.to(dev), y.to(dev)
                out = net(muq, mel, genre, feats)
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
            if Config.USE_EARLY_STOPPING and early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"验证集损失连续 {Config.EARLY_STOPPING_PATIENCE} 个epoch未改善，触发早停。")
                break
                
    print("\n--- 开始评估最佳模型 ---")
    net.load_state_dict(torch.load(Config.BEST_MODEL_PATH))
    net.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for muq, mel, genre, feats, y in val_loader:
            muq, mel, genre, feats = muq.to(dev), mel.to(dev), genre.to(dev), feats.to(dev)
            y_pred.extend(net(muq, mel, genre, feats).argmax(1).cpu().numpy())
            y_true.extend(y.numpy())
            
    report = classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)
    print("\n" + "="*50)
    print("           Classification Report")
    print("="*50)
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