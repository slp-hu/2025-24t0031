#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train_save_best.py

该脚本基于您的原始训练和验证代码，增加了以下两点功能：
1. 在验证集上表现最好的模型保存逻辑（原有逻辑）。
2. 生成并保存“验证集 CSV”，方便后续拿去推理使用。

运行后会在每次出现更低验证损失时，将模型权重保存为 `best_model.pth`。  
同时会在训练结束后绘制训练/验证损失和准确率曲线，并把那 78 首验证集对应的行保存为 `validation_set.csv`。

使用方法：
    python train_save_best.py

请确保以下几点：
1. `processed_annotations.csv`、`mel_features/` 文件夹位于与该脚本同级目录下。
2. `processed_annotations.csv` 中包含以下列：
       - song_id: 唯一标识符
       - genres: 多标签流派（例如 "rock;pop;metal"）
       - label: 情感标签（例如 "happy","sad","calm","fuzzy","tense"）
3. mel_features/ 下存储了与 song_id 对应的 .npy 谱图文件，如 mel_features/12345.npy。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder

# =============================== 超参数（可根据需要修改） ===============================
BATCH_SIZE       = 32          # 每次训练/验证的批量大小
LR               = 1e-3        # 学习率
EPOCHS           = 100          # 最大训练轮数
PATIENCE         = 5           # EarlyStopping 的容忍轮数（如果启用 EarlyStopping）
USE_ES           = False       # 是否启用 EarlyStopping
VAL_SIZE         = 78          # 验证集固定样本数（78 首）
MEL_DIR          = 'mel_features'               # .npy 谱图文件夹路径
CSV_PATH         = 'processed_annotations.csv'  # 原始标注 CSV 路径
SAVE_MODEL_PATH  = 'best_model.pth'             # 保存最优模型权重的文件名
VALIDATION_CSV   = 'validation_set.csv'          # 本脚本新增：导出验证集 CSV 的文件名
# ==========================================================================================

# 1. 读取并准备数据
df = pd.read_csv(CSV_PATH)

# 如果 processed_annotations.csv 中存在 label 列（我们需要用它做 LabelEncoder）
if 'label' not in df.columns:
    raise ValueError("CSV 中缺少 'label' 列，请检查 processed_annotations.csv")

# 构造 LabelEncoder，将情感标签转换为索引
le = LabelEncoder()
df['label_idx'] = le.fit_transform(df['label'])
classes = list(le.classes_)      # 类别名称列表，例如 ['calm','fuzzy','happy','sad','tense']
n_classes = len(classes)

# 构造 genre 到索引的映射（与训练时保持一致）
all_genres = set(g for row in df['genres'] for g in row.split(';'))
genre2idx = {g: i for i, g in enumerate(sorted(all_genres))}
n_genres = len(genre2idx)

# 2. 自定义 Dataset
class MelGenreDataset(Dataset):
    def __init__(self, df, mel_dir):
        self.df = df.reset_index(drop=True)
        self.mel_dir = mel_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        song_id = row['song_id']
        # 1) 读取 mel 谱图 .npy 并转换为 tensor，形状变为 (1, n_mel_bins, time_steps)
        mel_path = os.path.join(self.mel_dir, f"{song_id}.npy")
        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"找不到 {mel_path}")
        mel = np.load(mel_path)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mel_bins, time_steps)

        # 2) 构造 genre 权重向量 w_vec，长度为 n_genres
        #    流派字符串示例："rock;pop;metal"
        genres = row['genres'].split(';')
        # 按顺序给出倒数权重：1/(i+1)，然后归一化
        inv = np.array([1.0 / (i + 1) for i in range(len(genres))], dtype=np.float32)
        weights = inv / inv.sum()
        w_vec = np.zeros(n_genres, dtype=np.float32)
        for i, g in enumerate(genres):
            w_vec[genre2idx[g]] = weights[i]
        w_vec = torch.tensor(w_vec)

        # 3) 构造标签
        label = torch.tensor(row['label_idx'], dtype=torch.long)

        return mel, w_vec, label

# 实例化数据集
dataset = MelGenreDataset(df, mel_dir=MEL_DIR)

# 3. 划分训练集和验证集，并“导出验证集 CSV”
#    random_split 会返回两个 Subset 对象：train_ds 和 val_ds
#    val_ds.indices 可以拿到验证集对应的原始索引列表
train_size = len(dataset) - VAL_SIZE
val_size   = VAL_SIZE

# 如果希望划分固定的随机结果，可以在这里设置随机种子，例如：
# generator = torch.Generator().manual_seed(42)
# train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

train_ds, val_ds = random_split(dataset, [train_size, val_size])

# 新增：将验证集对应的 DataFrame 行导出到 CSV，以便后续推理时使用
# val_ds.indices 是一个包含 78 个 int 索引的列表，对应 df 的行位置
val_indices = val_ds.indices
val_df = df.iloc[val_indices].reset_index(drop=True)
val_df.to_csv(VALIDATION_CSV, index=False, encoding='utf-8')
print(f"已将验证集的 {VAL_SIZE} 条记录保存为：{VALIDATION_CSV}")

# 构造 DataLoader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# 4. 定义含残差连接的模型结构
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, downsample=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.pool  = nn.MaxPool2d(2) if downsample else nn.Identity()

        # 如果 in_channels != out_channels，需要一个 1x1 卷积来匹配维度
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        out = self.pool(out)
        return out

class EmotionCNNRes(nn.Module):
    def __init__(self, n_genres, genre_emb_dim=32, n_classes=5):
        super().__init__()
        self.res_blocks = nn.Sequential(
            ResBlock(1,  16),
            ResBlock(16, 32),
            ResBlock(32, 64),
            ResBlock(64, 128, downsample=False)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.genre_fc = nn.Linear(n_genres, genre_emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128 + genre_emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, mel, genre_w):
        """
        输入：
            - mel: (batch_size, 1, n_mel_bins, time_steps)
            - genre_w: (batch_size, n_genres)
        输出：
            - logits: (batch_size, n_classes)
        """
        x = self.res_blocks(mel)                   # (batch_size, 128, h, w)
        x = self.pool(x).view(x.size(0), -1)       # (batch_size, 128)

        g = F.relu(self.genre_fc(genre_w))         # (batch_size, genre_emb_dim)
        h = torch.cat([x, g], dim=1)               # (batch_size, 128 + genre_emb_dim)
        logits = self.classifier(h)                # (batch_size, n_classes)
        return logits

# 5. 训练函数，包含 EarlyStopping 和最佳模型保存逻辑
def train_model(model, train_loader, val_loader,
                lr, epochs, use_early_stopping, patience,
                save_path):
    model.to(device)
    optim     = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    wait = 0

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss'  : [], 'val_acc'  : []
    }

    for ep in range(1, epochs + 1):
        # === 训练阶段 ===
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        for mel, w_vec, label in train_loader:
            mel, w_vec, label = mel.to(device), w_vec.to(device), label.to(device)
            optim.zero_grad()
            logits = model(mel, w_vec)
            loss = criterion(logits, label)
            loss.backward()
            optim.step()

            total_train_loss += loss.item() * mel.size(0)
            train_correct += (logits.argmax(dim=1) == label).sum().item()

        train_loss = total_train_loss / len(train_loader.dataset)
        train_acc  = train_correct / len(train_loader.dataset)

        # === 验证阶段 ===
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for mel, w_vec, label in val_loader:
                mel, w_vec, label = mel.to(device), w_vec.to(device), label.to(device)
                logits = model(mel, w_vec)
                loss = criterion(logits, label)
                total_val_loss += loss.item() * mel.size(0)
                val_correct += (logits.argmax(dim=1) == label).sum().item()

        val_loss = total_val_loss / len(val_loader.dataset)
        val_acc  = val_correct / len(val_loader.dataset)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印信息
        print(f"Epoch {ep:02d} | "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

        # 如果当前 val_loss 是最低的，则保存模型权重
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ↓ New best model saved to {save_path} (val_loss={val_loss:.4f})")
        else:
            wait += 1
            if use_early_stopping and wait >= patience:
                print(f"Early stopping triggered at epoch {ep}")
                break

    return history

# 6. 主流程
if __name__ == '__main__':
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 实例化模型
    model = EmotionCNNRes(n_genres=n_genres, genre_emb_dim=32, n_classes=n_classes)

    # 训练并保存最佳模型
    history = train_model(
        model,
        train_loader,
        val_loader,
        lr=LR,
        epochs=EPOCHS,
        use_early_stopping=USE_ES,
        patience=PATIENCE,
        save_path=SAVE_MODEL_PATH
    )

    # （可选）保存最后一轮模型
    last_model_path = 'last_model.pth'
    torch.save(model.state_dict(), last_model_path)
    print(f"Finished training. Last model saved to {last_model_path}")

    # 7. 绘制训练/验证曲线
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss 曲线
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss')
    ax1.plot(epochs_range, history['val_loss'],   label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend(loc='best')

    # Accuracy 曲线
    ax2.plot(epochs_range, history['train_acc'], label='Train Acc')
    ax2.plot(epochs_range, history['val_acc'],   label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.show()
