#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================== 参数区域 ==========================
DEVICE            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BEST_MODEL_PATH   = 'best_model0814.pth'          # 训练时保存的最佳模型权重
TRAIN_CSV_PATH    = 'processed_annotations.csv'   # 训练时用的完整标注 CSV，包含所有流派
VALID_CSV_PATH    = 'validation_set.csv'          # 仅含验证集 78 首的 CSV
MEL_DIR           = 'mel_features'                # mel 特征 .npy 所在文件夹
OUTPUT_CSV        = 'inference_results_78.csv'    # 最终要输出的推理结果
# =============================================================

# 1. 先从 TRAIN_CSV_PATH 里读取完整标注 CSV，构造与训练时一致的 genre2idx
train_df = pd.read_csv(TRAIN_CSV_PATH, dtype={'song_id': str})
if 'genres' not in train_df.columns:
    raise ValueError(f"{TRAIN_CSV_PATH} 中缺少 'genres' 列，请检查。")

# 把所有训练集中出现过的流派都取出来
all_train_genres = set()
for row in train_df['genres']:
    for g in row.split(';'):
        all_train_genres.add(g)

# 排序后构造映射，保证顺序和训练时一模一样
genre2idx = {g: i for i, g in enumerate(sorted(all_train_genres))}
n_genres  = len(genre2idx)

# 2. 固定训练时的类别顺序 (classes)，不要在验证集上重新 fit
#    假设训练时 LabelEncoder.fit_transform(df['label']) 得到这五个类别：
#    ['calm', 'fuzzy', 'happy', 'sad', 'tense']
classes   = ['calm', 'fuzzy', 'happy', 'sad', 'tense']
n_classes = len(classes)

# 3. 定义模型结构（与训练时完全一致）
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
        self.pool      = nn.AdaptiveAvgPool2d((1, 1))
        self.genre_fc  = nn.Linear(n_genres, genre_emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128 + genre_emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, mel, genre_w):
        x = self.res_blocks(mel)                   # (batch_size, 128, h, w)
        x = self.pool(x).view(x.size(0), -1)       # (batch_size, 128)
        g = F.relu(self.genre_fc(genre_w))         # (batch_size, genre_emb_dim)
        h = torch.cat([x, g], dim=1)               # (batch_size, 128 + genre_emb_dim)
        return self.classifier(h)                  # (batch_size, n_classes)

# 4. 实例化模型并加载权重
model = EmotionCNNRes(n_genres=n_genres, genre_emb_dim=32, n_classes=n_classes).to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()
softmax = nn.Softmax(dim=1)

# 5. 读入验证集 CSV，只做推理循环
val_df = pd.read_csv(VALID_CSV_PATH, dtype={'song_id': str})
required_cols = {'song_id', 'genres', 'label'}
if not required_cols.issubset(set(val_df.columns)):
    raise ValueError(f"{VALID_CSV_PATH} 文件缺少 {required_cols} 中的列，请检查。")

# 6. 类别名到输出列名的映射（训练标签 → 最终输出列名）
mapping = {
    'calm':   'p_Relaxed',
    'fuzzy':  'p_Ambiguous',
    'happy':  'p_Happy',
    'sad':    'p_Sad',
    'tense':  'p_Angry'
}

results = []
for idx, row in val_df.iterrows():
    song_id = row['song_id']
    genres  = row['genres']
    label_gt= row['label']  # 真实标签（可选，推理时也输出一并对照）

    # 6.1 读取 mel 特征 .npy
    mel_path = os.path.join(MEL_DIR, f"{song_id}.npy")
    if not os.path.isfile(mel_path):
        print(f"[Warning] 找不到 mel 文件：{mel_path}，跳过该条。")
        continue

    mel_np = np.load(mel_path)  # e.g. (n_mel_bins, time_steps)
    mel_tensor = torch.tensor(mel_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    #    变成 (1, 1, n_mel_bins, time_steps)

    # 6.2 构造 genres 权重向量（与训练时完全一致）
    genre_list = genres.split(';')
    inv    = np.array([1.0/(i+1) for i in range(len(genre_list))], dtype=np.float32)
    weights= inv / inv.sum()
    w_vec_np = np.zeros(n_genres, dtype=np.float32)
    for i, g in enumerate(genre_list):
        # “g” 必须存在于全量训练集里的 genre2idx，否则会 KeyError
        w_vec_np[genre2idx[g]] = weights[i]
    w_vec_tensor = torch.tensor(w_vec_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, n_genres)

    # 6.3 前向推理
    with torch.no_grad():
        logits = model(mel_tensor, w_vec_tensor)  # (1, n_classes)
        probs  = softmax(logits)                  # (1, n_classes)

    probs = probs.squeeze(0).cpu().numpy()  # 一维长度 = 5

    # 6.4 拼成一条结果字典
    row_dict = {
        'song_id': song_id,
        'genres' : genres,
        'label'  : label_gt,
    }
    for cls_idx, cls_name in enumerate(classes):
        out_col = mapping[cls_name]  # 例如 'calm' -> 'p_Relaxed'
        row_dict[out_col] = float(probs[cls_idx])

    results.append(row_dict)

# 7. 输出到 CSV，并严格保证列顺序
out_df = pd.DataFrame(results)
cols_order = ['song_id', 'genres', 'label',
              'p_Ambiguous', 'p_Angry', 'p_Happy', 'p_Relaxed', 'p_Sad']
out_df = out_df[cols_order]
out_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
print(f"推理完成，已将结果保存到：{OUTPUT_CSV}")
