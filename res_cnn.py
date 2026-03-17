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

# ===============================
#       超参数（请在此处修改）
# ===============================
BATCH_SIZE = 32    # 每次训练/验证的样本数
LR         = 1e-3  # 学习率
EPOCHS     = 50    # 最大训练轮数
PATIENCE   = 5     # EarlyStopping 的容忍轮数
USE_ES     = False  # 是否启用 EarlyStopping
VAL_SIZE   = 78    # 验证集大小（固定条数）
MEL_DIR    = 'mel_features'  # .npy 谱图文件夹路径
CSV_PATH   = 'processed_annotations.csv'  # 处理后 CSV 路径
# ===============================
#      下面无需修改，直接运行
# ===============================

# 1. 读取并准备数据
df = pd.read_csv(CSV_PATH)
all_genres = set(g for row in df['genres'] for g in row.split(';'))
genre2idx = {g: i for i, g in enumerate(sorted(all_genres))}
n_genres = len(genre2idx)
le = LabelEncoder()
df['label_idx'] = le.fit_transform(df['label'])
n_classes = len(le.classes_)

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
        mel = np.load(os.path.join(self.mel_dir, f"{song_id}.npy"))
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        genres = row['genres'].split(';')
        inv = np.array([1.0/(i+1) for i in range(len(genres))], dtype=np.float32)
        weights = inv / inv.sum()
        w_vec = np.zeros(n_genres, dtype=np.float32)
        for i, g in enumerate(genres):
            w_vec[genre2idx[g]] = weights[i]
        w_vec = torch.tensor(w_vec)

        label = torch.tensor(row['label_idx'], dtype=torch.long)
        return mel, w_vec, label

dataset = MelGenreDataset(df, mel_dir=MEL_DIR)

# 3. 划分训练/验证集
train_size = len(dataset) - VAL_SIZE
train_ds, val_ds = random_split(dataset, [train_size, VAL_SIZE])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

# 4. 定义含残差连接的模型
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2, downsample=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()

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
            ResBlock(1, 16),
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
        x = self.res_blocks(mel)
        x = self.pool(x).view(x.size(0), -1)
        g = F.relu(self.genre_fc(genre_w))
        h = torch.cat([x, g], dim=1)
        return self.classifier(h)

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = EmotionCNNRes(n_genres=n_genres, genre_emb_dim=32, n_classes=n_classes).to(device)

# 5. 训练函数（含 EarlyStopping）
def train_model(model, train_loader, val_loader,
                lr, epochs, use_early_stopping, patience):
    optim     = Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')
    wait = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for ep in range(1, epochs+1):
        # 训练
        model.train()
        total_loss, correct = 0.0, 0
        for mel, w_vec, label in train_loader:
            mel, w_vec, label = mel.to(device), w_vec.to(device), label.to(device)
            optim.zero_grad()
            logits = model(mel, w_vec)
            loss = criterion(logits, label)
            loss.backward()
            optim.step()
            total_loss += loss.item() * mel.size(0)
            correct += (logits.argmax(1) == label).sum().item()

        train_loss = total_loss / len(train_loader.dataset)
        train_acc  = correct / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for mel, w_vec, label in val_loader:
                mel, w_vec, label = mel.to(device), w_vec.to(device), label.to(device)
                logits = model(mel, w_vec)
                val_loss += criterion(logits, label).item() * mel.size(0)
                val_correct += (logits.argmax(1) == label).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc  = val_correct / len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {ep:02d} | "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")

        if use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {ep}")
                    break

    return history

# 6. 运行训练
history = train_model(
    model,
    train_loader,
    val_loader,
    lr=LR,
    epochs=EPOCHS,
    use_early_stopping=USE_ES,
    patience=PATIENCE
)

# 7. 绘制曲线
epochs = range(1, len(history['train_loss']) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, history['train_loss'], label='Train Loss')
ax1.plot(epochs, history['val_loss'],   label='Val Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.grid(True); ax1.legend(loc='best')

ax2.plot(epochs, history['train_acc'], label='Train Acc')
ax2.plot(epochs, history['val_acc'],   label='Val Acc')
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
ax2.grid(True); ax2.legend(loc='best')

plt.tight_layout()
plt.show()
