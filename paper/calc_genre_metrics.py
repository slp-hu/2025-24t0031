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
K = 10  # 评估 Precision@10
MIN_SONGS_PER_GENRE = 150  # 过滤曲目数大于等于 150 的流派

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════ 2. 模型定义 ════════════════════
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

class MultiOutputDataset(Dataset):
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
        genre_name = row.major_genre_name_en
        return mel_spec, embedding, genre_vector, audio_features, true_emotions, genre_name

def collate_fn(batch):
    mels, embs, genres, audios, labels, genre_names = zip(*batch)
    max_len = max(m.shape[1] for m in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len)
    for i, m in enumerate(mels): padded_mels[i, :, :m.shape[1]] = m
    return padded_mels, default_collate(embs), default_collate(genres), default_collate(audios), labels, genre_names

# ════════════════════ 3. 主程序 ════════════════════
def main():
    print("--- > Loading Data ---")
    df = pd.read_csv(DATASET_CSV)
    df.dropna(subset=['id', 'major_genre_name_en', 'emotion_sequence'], inplace=True)
    df['id'] = df['id'].astype(str)
    
    # 统计流派曲数
    genre_counts = df['major_genre_name_en'].value_counts()
    valid_genres = genre_counts[genre_counts >= MIN_SONGS_PER_GENRE].index.tolist()
    print(f"> Found {len(valid_genres)} genres with >= {MIN_SONGS_PER_GENRE} songs.")
    
    # 初始化
    feature_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    all_genres_list = sorted(df["major_genre_name_en"].unique()); genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    
    dataset = MultiOutputDataset(df, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, len(all_genres_list), genre_to_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    model = QuadFusionNet(EMB_DIM, len(all_genres_list), len(feature_cols), N_EMOTIONS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_scores = []; all_labels = []; all_genres = []

    print("> Running inference...")
    with torch.no_grad():
        for mel, emb, genre_vec, audio, label_batch, genre_batch in tqdm(loader):
            outputs = model(mel.to(device), emb.to(device), genre_vec.to(device), audio.to(device))
            all_scores.append(outputs.cpu().numpy())
            all_labels.extend(label_batch)
            all_genres.extend(genre_batch)

    scores_matrix = np.concatenate(all_scores, axis=0)
    
    # ════════════════════ 按流派统计 Precision@10 ════════════════════
    genre_results = []

    for genre in valid_genres:
        # 1. 提取该流派的数据索引
        genre_indices = [i for i, g in enumerate(all_genres) if g == genre]
        sub_scores = scores_matrix[genre_indices]
        sub_labels = [all_labels[i] for i in genre_indices]
        
        emo_precisions = []
        # 2. 对每个情感在该流派内计算 Precision@10
        for emo_idx in range(N_EMOTIONS):
            emo_name = IDX_TO_EMOTION[emo_idx]
            emo_scores = sub_scores[:, emo_idx]
            
            # 流派内排序，取前 10
            top_k_indices = np.argsort(emo_scores)[::-1][:K]
            
            hits = 0
            for idx in top_k_indices:
                if emo_name in sub_labels[idx]:
                    hits += 1
            
            p_at_k = hits / K
            emo_precisions.append(p_at_k)
        
        avg_p10 = np.mean(emo_precisions)
        genre_results.append({'Genre': genre, 'Count': len(genre_indices), 'Precision@10': avg_p10})

    # ════════════════════ 输出结果 ════════════════════
    res_df = pd.DataFrame(genre_results).sort_values(by='Precision@10', ascending=False)
    
    print(f"\n{'='*60}")
    print(f"{'Genre':<20} | {'Count':<8} | {'Precision@10':<12}")
    print("-" * 60)
    for _, row in res_df.iterrows():
        print(f"{row['Genre']:<20} | {int(row['Count']):<8} | {row['Precision@10']:.4f}")
    print(f"{'='*60}\n")
    
    res_df.to_csv("genre_precision_results.csv", index=False)
    print("> Results saved to genre_precision_results.csv")

if __name__ == '__main__':
    main()