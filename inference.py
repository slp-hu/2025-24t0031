import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

# ========================== 参数区域 ==========================
DEVICE           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BEST_MODEL_PATH  = 'best_model0814.pth'              # 第一步训练并保存的最佳模型
MEL_DIR          = 'mel_features'                # mel 特征 .npy 所在文件夹
CSV_PATH         = 'validation_set.csv'   # 原始标注 CSV（必须包含 song_id, genres, label 三列）
OUTPUT_CSV       = 'inference_results_78.csv'       # 最终输出的文件
# =============================================================

# 1. 载入原始标注 CSV，确保里面有 “song_id”, “genres”, “label”
df = pd.read_csv(CSV_PATH)

# 2. 重新构造 LabelEncoder，保证顺序和训练时一致
le = LabelEncoder()
le.fit(df['label'])       # 假设这里的 df['label'] 中有 ['happy','sad','calm','fuzzy','tense'] 等类别
classes = list(le.classes_)  # 类别列表，比如 ['calm','fuzzy','happy','sad','tense']

# 3. 构造 genre2idx（同训练时保持完全一致）
all_genres = set(g for row in df['genres'] for g in row.split(';'))
genre2idx  = {g: i for i, g in enumerate(sorted(all_genres))}
n_genres   = len(genre2idx)
n_classes  = len(classes)

# ================= 定义模型结构（必须与训练时一模一样） =================
import torch.nn as nn

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
            ResBlock(1, 16),
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
        x = self.res_blocks(mel)
        x = self.pool(x).view(x.size(0), -1)
        g = F.relu(self.genre_fc(genre_w))
        h = torch.cat([x, g], dim=1)
        return self.classifier(h)
# =======================================================================

# 4. 实例化模型并加载最佳权重
model = EmotionCNNRes(n_genres=n_genres, genre_emb_dim=32, n_classes=n_classes).to(DEVICE)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()

softmax = nn.Softmax(dim=1)

# 5. 对每首歌做推理，并把结果保存在列表里
results = []
for idx, row in df.iterrows():
    song_id = row['song_id']
    genres  = row['genres']     # 原始字符串，比如 "rock;pop;metal"
    label_gt= row['label']      # 原始标注的情感标签

    # 5.1 读取对应 .npy
    mel_path = os.path.join(MEL_DIR, f"{song_id}.npy")
    if not os.path.isfile(mel_path):
        print(f"[Warning] 找不到 mel 文件：{mel_path}，跳过这条记录。")
        continue

    mel_np = np.load(mel_path)                                  # e.g. (n_mel_bins, time_steps)
    # 转成 (1,1,n_mel_bins,time_steps)
    mel_tensor = torch.tensor(mel_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # 5.2 构造 genres 对应的权重向量 w_vec（与训练时代码完全一致）
    genre_list = genres.split(';')
    inv = np.array([1.0/(i+1) for i in range(len(genre_list))], dtype=np.float32)
    weights = inv / inv.sum()
    w_vec_np = np.zeros(n_genres, dtype=np.float32)
    for i, g in enumerate(genre_list):
        w_vec_np[genre2idx[g]] = weights[i]
    w_vec_tensor = torch.tensor(w_vec_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, n_genres)

    # 5.3 前向推理，得到 logits，再做 Softmax 得到概率
    with torch.no_grad():
        logits = model(mel_tensor, w_vec_tensor)   # (1, n_classes)
        probs  = softmax(logits)                   # (1, n_classes)

    probs = probs.squeeze(0).cpu().numpy()  # 变成一维 array，长度 = n_classes

    # 5.4 把这一条的结果拼成一个 dict，包括 song_id、genres、原始 label 和各类别概率
    row_dict = {
        'song_id': song_id,
        'genres' : genres,
        'label'  : label_gt,
    }
    # 把每个类别的概率加进来，列名就是 p_<类别名称>
    for cls_idx, cls_name in enumerate(classes):
        row_dict[f"p_{cls_name}"] = float(probs[cls_idx])

    results.append(row_dict)

# 6. 最后把所有结果写成 DataFrame，然后输出到 CSV
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"推理完成，已将结果保存到：{OUTPUT_CSV}")
