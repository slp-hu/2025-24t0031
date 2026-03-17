import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm

# ════════════════════ 配置区域 ════════════════════
# 请根据你的实际路径修改
MUQ_EMB_DIR = r"G:\13kmid30s_muq"
MEL_FEAT_DIR = r"G:\13kmid30smel"
DATASET_CSV = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon.csv"
MODEL_PATH = "best_ranking_model_by_ndcg.pth"  # 你的模型权重文件

# 目标查询：EDM + Excitement
TARGET_GENRE = "EDM"
TARGET_EMOTION = "Excitement"

ALL_EMOTIONS = [
    'Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 'Quiet',
    'Happiness', 'Loneliness', 'Touching', 'Missing', 'Fresh', 'Relaxation'
]
TARGET_EMO_IDX = ALL_EMOTIONS.index(TARGET_EMOTION)

BATCH_SIZE = 64
EMB_DIM = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════ 模型定义 (保持不变) ════════════════════
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
        # 兼容处理：有些文件名可能没有.npy后缀，视你的数据情况而定
        mel_path = os.path.join(self.mel_dir, f"{file_id}.npy")
        muq_path = os.path.join(self.muq_dir, f"{file_id}.npy")
        
        mel_spec = torch.from_numpy(np.load(mel_path).astype(np.float32))
        embedding = torch.from_numpy(np.load(muq_path).astype(np.float32))
        
        genre_vector = torch.zeros(self.n_genres)
        if row.major_genre_name_en in self.genre_to_idx: genre_vector[self.genre_to_idx[row.major_genre_name_en]] = 1.0
        audio_features = torch.tensor(row[self.feature_cols].values.astype(np.float32))
        true_emotions = row.emotion_sequence # Just pass string
        return mel_spec, embedding, genre_vector, audio_features, file_id, true_emotions

def collate_fn(batch):
    mels, embs, genres, audios, ids, labels = zip(*batch)
    max_len = max(m.shape[1] for m in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len)
    for i, m in enumerate(mels): padded_mels[i, :, :m.shape[1]] = m
    return padded_mels, default_collate(embs), default_collate(genres), default_collate(audios), ids, labels

# ════════════════════ 主逻辑 ════════════════════
if __name__ == '__main__':
    print(f"> Loading Data for Query: {TARGET_GENRE} + {TARGET_EMOTION}")
    
    # 1. Load CSV
    df = pd.read_csv(DATASET_CSV)
    df.dropna(subset=['id', 'major_genre_name_en', 'emotion_sequence'], inplace=True)
    df['id'] = df['id'].astype(str)
    
    # 2. Filter ONLY EDM songs for inference
    edm_df = df[df['major_genre_name_en'] == TARGET_GENRE].copy()
    print(f"> Found {len(edm_df)} songs with genre {TARGET_GENRE}")

    # 3. Setup Dataset/Loader
    feature_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    # 注意：这里 genre_list 必须基于全集生成，以保持 one-hot 维度一致
    full_genre_list = sorted(df["major_genre_name_en"].unique())
    genre_to_idx = {g: i for i, g in enumerate(full_genre_list)}
    
    dataset = FullInferenceDataset(edm_df, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, len(full_genre_list), genre_to_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    
    # 4. Model
    model = QuadFusionNet(EMB_DIM, len(full_genre_list), len(feature_cols), len(ALL_EMOTIONS)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 5. Inference
    all_scores = []
    all_ids = []
    all_embeddings = [] # Save for UI
    all_true_labels = []

    print("> Running Inference...")
    with torch.no_grad():
        for mel, emb, genre, audio, ids, labels in tqdm(loader):
            outputs = model(mel.to(device), emb.to(device), genre.to(device), audio.to(device))
            # Extract score ONLY for Excitement
            excitement_scores = outputs[:, TARGET_EMO_IDX].cpu().numpy()
            
            all_scores.extend(excitement_scores)
            all_ids.extend(ids)
            all_embeddings.extend(emb.cpu().numpy())
            all_true_labels.extend(labels)

    # 6. Ranking
    # Create a list of tuples: (score, id, embedding, true_label)
    combined = list(zip(all_scores, all_ids, all_embeddings, all_true_labels))
    # Sort descending by score
    combined.sort(key=lambda x: x[0], reverse=True)
    
    # Take Top 40
    top_40 = combined[:40]
    
    # 7. Save CSV for Manual Evaluation
    print("> Saving manual_eval.csv...")
    eval_data = []
    for rank, (score, song_id, _, true_lbl) in enumerate(top_40, 1):
        eval_data.append({
            'Rank': rank,
            'Song_ID': song_id,
            'Predicted_Score': score,
            'True_Labels': true_lbl,
            'Is_Relevant (1/0)': '' # User fills this
        })
    pd.DataFrame(eval_data).to_csv("manual_eval.csv", index=False)
    print("  Done. Please open 'manual_eval.csv' and fill in the 'Is_Relevant' column.")
    
    # 8. Save Data for UI Visualization
    print("> Saving top40_for_ui.npz...")
    top40_ids = [x[1] for x in top_40]
    top40_embs = np.array([x[2] for x in top_40])
    top40_ranks = np.arange(1, 41)
    
    np.savez("top40_for_ui.npz", 
             ids=top40_ids, 
             embeddings=top40_embs, 
             ranks=top40_ranks,
             genre=TARGET_GENRE,
             emotion=TARGET_EMOTION)
    print("  Done. Now run the UI script.")