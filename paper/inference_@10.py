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
K = 10  # 评估 Top-10 推荐列表
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════ 2. 推荐列表评估函数 ════════════════════
def evaluate_recommendation_list(all_pred_scores, all_true_lists, k=10):
    """
    以情感为查询中心，评估歌曲推荐列表的准确率
    all_pred_scores: [N_songs, 12] 的预测分数矩阵
    all_true_lists: 长度为 N_songs 的列表，每个元素是该歌曲的真值标签集合
    """
    emo_precision_list = []
    emo_recall_list = []
    
    print(f"\n> 正在计算 12 种情感各自的 Precision@{k} 和 Recall@{k}...")
    
    for emo_idx in range(N_EMOTIONS):
        emo_name = IDX_TO_EMOTION[emo_idx]
        
        # 1. 获取所有歌曲在这个情感维度的分数
        scores_for_this_emo = all_pred_scores[:, emo_idx]
        
        # 2. 对所有歌曲按分数从高到低排序，取前 K 个索引
        top_k_song_indices = np.argsort(scores_for_this_emo)[::-1][:k]
        
        # 3. 统计全量数据中，哪些歌曲真的属于这个情感
        # 建立一个布尔向量：真值标签里有该情感则为 1，否则为 0
        is_relevant_global = np.array([1 if emo_name in tags else 0 for tags in all_true_lists])
        total_relevant_in_db = np.sum(is_relevant_global)
        
        if total_relevant_in_db == 0:
            continue
            
        # 4. 计算 Top-K 中命中的数量
        hits = np.sum(is_relevant_global[top_k_song_indices])
        
        # Precision@K: 推荐的 10 首歌里对了几首
        p_at_k = hits / k
        
        # Recall@K: 全库这么多属于该情感的歌，前 10 名里抓到了几首
        r_at_k = hits / total_relevant_in_db
        
        emo_precision_list.append(p_at_k)
        emo_recall_list.append(r_at_k)
        
        # 打印每个情感的明细（可选，用于写论文分析）
        # print(f"  - {emo_name:12}: P@{k}={p_at_k:.4f}, R@{k}={r_at_k:.4f} (Hits: {hits}/{k})")

    avg_p = np.mean(emo_precision_list)
    avg_r = np.mean(emo_recall_list)
    
    return avg_p, avg_r

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
    
    # 2. 初始化环境
    feature_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    genre_list = sorted(df["major_genre_name_en"].unique()); genre_to_idx = {g: i for i, g in enumerate(genre_list)}
    
    dataset = FullInferenceDataset(df, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, len(genre_list), genre_to_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    model = QuadFusionNet(EMB_DIM, len(genre_list), len(feature_cols), N_EMOTIONS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_scores = []; all_labels = []

    # 3. 推理阶段
    print("> 正在对全量数据进行推理...")
    with torch.no_grad():
        for mel, emb, genre, audio, label_batch in tqdm(loader):
            outputs = model(mel.to(device), emb.to(device), genre.to(device), audio.to(device))
            all_scores.append(outputs.cpu().numpy())
            all_labels.extend(label_batch)

    # 4. 指标计算阶段
    all_scores_matrix = np.concatenate(all_scores, axis=0)
    
    # 调用针对推荐列表的评估函数
    p_at_10, r_at_10 = evaluate_recommendation_list(all_scores_matrix, all_labels, k=K)

    print(f"\n{'='*50}")
    print(f"📊 推荐系统视角评估结果 (K={K}):")
    print(f"   平均 Precision@{K} : {p_at_10:.4f}  (前{K}首歌中有百分之几是符合情感的)")
    print(f"   平均 Recall@{K}    : {r_at_10:.4f}  (全库符合情感的歌中有百分之几进入了前{K})")
    print(f"{'='*50}\n")