import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm

# ════════════════════ 1. 配置区域 ════════════════════
MUQ_EMB_DIR = "G:\\13kmid30s_muq"
MEL_FEAT_DIR = "G:\\13kmid30smel"
DATASET_CSV = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon.csv"
MODEL_PATH = "best_ranking_model_by_ndcg.pth"

ALL_EMOTIONS = [
    'Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 'Quiet',
    'Happiness', 'Loneliness', 'Touching', 'Missing', 'Fresh', 'Relaxation'
]
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(ALL_EMOTIONS)}
IDX_TO_EMOTION = {i: emo for i, emo in enumerate(ALL_EMOTIONS)}
N_EMOTIONS = len(ALL_EMOTIONS)

BATCH_SIZE = 64
EMB_DIM = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════ 2. 评估函数 (标准分类指标) ════════════════════
def evaluate_classification_metrics(all_pred_scores, all_true_lists, threshold=0.0):
    """
    计算标准的多标签分类指标：Precision, Recall, F1
    因为模型输出的是 raw logits (未经过 Sigmoid)，所以默认 threshold 设置为 0.0
    （这等价于 Sigmoid 后的 threshold = 0.5）
    """
    emo_precision_list = []
    emo_recall_list = []
    emo_f1_list = []
    
    print(f"\n> 正在计算 12 种情感各自的标准 Precision, Recall 以及 F1 (Logit 阈值={threshold})...")
    
    for emo_idx in range(N_EMOTIONS):
        emo_name = IDX_TO_EMOTION[emo_idx]
        
        # 1. 真实标签：包含该情感为 1，否则为 0
        y_true = np.array([1 if emo_name in tags else 0 for tags in all_true_lists])
        
        # 2. 预测标签：分数大于阈值则判定为 1，否则为 0
        y_pred = (all_pred_scores[:, emo_idx] > threshold).astype(int)
        
        # 3. 计算 TP, FP, FN
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
        
        # 4. 计算指标 (注意处理分母为0的情况)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        emo_precision_list.append(precision)
        emo_recall_list.append(recall)
        emo_f1_list.append(f1)
        
        # 打印每个情感的具体明细
        print(f"  - {emo_name:12}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}  (TP:{TP}, FP:{FP}, FN:{FN})")

    avg_p = np.mean(emo_precision_list)
    avg_r = np.mean(emo_recall_list)
    macro_f1 = np.mean(emo_f1_list) # 计算 Macro F1
    
    return avg_p, avg_r, macro_f1

# ════════════════════ 3. 模型与数据类 ════════════════════
class QuadFusionNet(nn.Module):
    def __init__(self, emb_dim, genre_dim, audio_feature_dim, n_classes):
        super().__init__()
        self.mel_cnn_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.emb_branch = nn.Sequential(nn.Linear(emb_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5))
        self.genre_branch = nn.Sequential(nn.Linear(genre_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5))
        self.audio_feature_branch = nn.Sequential(nn.BatchNorm1d(audio_feature_dim), nn.ReLU(), nn.Dropout(0.5))
        fusion_input_dim = 256 + 128 + 32 + audio_feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, n_classes)
        )
    def forward(self, mel, emb, genre, audio_feats):
        mel = mel.unsqueeze(1); z_mel = self.mel_cnn_branch(mel); z_mel = z_mel.view(z_mel.size(0), -1)
        z_emb = self.emb_branch(emb); z_genre = self.genre_branch(genre); z_audio = self.audio_feature_branch(audio_feats)
        combined = torch.cat([z_mel, z_emb, z_genre, z_audio], dim=1); output = self.fusion_head(combined)
        return output

class FullInferenceDataset(Dataset):
    def __init__(self, dataframe, mel_dir, muq_dir, feature_cols, n_genres, genre_to_idx):
        self.df = dataframe.reset_index(drop=True)
        self.mel_dir = mel_dir; self.muq_dir = muq_dir; self.feature_cols = feature_cols
        self.n_genres = n_genres; self.genre_to_idx = genre_to_idx

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]; file_id = str(row.id)
        mel_spec = torch.from_numpy(np.load(os.path.join(self.mel_dir, f"{file_id}.npy")).astype(np.float32))
        embedding = torch.from_numpy(np.load(os.path.join(self.muq_dir, f"{file_id}.npy")).astype(np.float32))
        genre_vector = torch.zeros(self.n_genres)
        if row.major_genre_name_en in self.genre_to_idx: genre_vector[self.genre_to_idx[row.major_genre_name_en]] = 1.0
        audio_features = torch.tensor(row[self.feature_cols].values.astype(np.float32))
        true_emotions = row.emotion_sequence.split(',') if pd.notna(row.emotion_sequence) else []
        return mel_spec, embedding, genre_vector, audio_features, true_emotions

def collate_fn(batch):
    mels, embs, genres, audios, labels = zip(*batch)
    max_len = max(m.shape[1] for m in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len)
    for i, m in enumerate(mels): padded_mels[i, :, :m.shape[1]] = m
    return padded_mels, default_collate(embs), default_collate(genres), default_collate(audios), labels

# ════════════════════ 4. 执行流程 ════════════════════
if __name__ == '__main__':
    # 1. 加载并清洗数据
    df = pd.read_csv(DATASET_CSV)
    df.dropna(subset=['id', 'major_genre_name_en', 'emotion_sequence'], inplace=True)
    df['id'] = df['id'].astype(str)
    
    # 【关键修复】：必须使用过滤前的全量 df 来获取 genre_list，保证维度为训练时的维度，与权重文件对齐！
    genre_list = sorted(df["major_genre_name_en"].unique())
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}
    
    # 2. 计算各个流派的歌曲数量，并过滤出歌曲数 >= 150 的流派
    genre_counts = df['major_genre_name_en'].value_counts()
    valid_genres = genre_counts[genre_counts >= 150].index
    
    # 保留数据集中仅属于有效流派的歌曲 (推理时只评估这些歌)
    df_filtered = df[df['major_genre_name_en'].isin(valid_genres)].copy()
    print(f"> 数据过滤完毕：原歌曲数 {len(df)} -> 过滤后歌曲数 {len(df_filtered)}")
    print(f"> 模型将使用原 {len(genre_list)} 个流派的维度，但仅对其中 {len(valid_genres)} 个流派的数据进行评估。")
    
    # 3. 初始化环境
    feature_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    
    # Dataset 传入 df_filtered (只评估过滤后的歌)，但使用全量的 genre_to_idx
    dataset = FullInferenceDataset(df_filtered, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, len(genre_list), genre_to_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    # Model 传入 len(genre_list) 保证维度一致，完美加载预训练权重
    model = QuadFusionNet(EMB_DIM, len(genre_list), len(feature_cols), N_EMOTIONS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_scores = []; all_labels = []

    # 4. 推理阶段
    print("> 正在对筛选后的数据进行推理...")
    with torch.no_grad():
        for mel, emb, genre, audio, label_batch in tqdm(loader):
            outputs = model(mel.to(device), emb.to(device), genre.to(device), audio.to(device))
            all_scores.append(outputs.cpu().numpy())
            all_labels.extend(label_batch)

    # 5. 指标计算阶段
    all_scores_matrix = np.concatenate(all_scores, axis=0)
    
    # 调用标准的分类评估函数
    avg_p, avg_r, macro_f1 = evaluate_classification_metrics(all_scores_matrix, all_labels, threshold=0.0)

    print(f"\n{'='*50}")
    print(f"📊 多标签分类视角评估结果 (Macro-average):")
    print(f"   平均 Precision : {avg_p:.4f}")
    print(f"   平均 Recall    : {avg_r:.4f}")
    print(f"   Macro F1       : {macro_f1:.4f}")
    print(f"{'='*50}\n")