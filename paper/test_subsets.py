# ─────────────────── Import Libraries ───────────────────
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.model_selection import train_test_split

# ═══════════＝ 1. Configurations and Paths ＝═══════════
# 请确保这些路径与您训练时的设置一致
MUQ_EMB_DIR = "G:\\13kmid30s_muq"
MEL_FEAT_DIR = "G:\\13kmid30smel"
DATASET_CSV = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon.csv"
BEST_MODEL_PATH = "best_ranking_model_by_ndcg.pth"

ALL_EMOTIONS = [
    'Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 'Quiet',
    'Happiness', 'Loneliness', 'Touching', 'Missing', 'Fresh', 'Relaxation'
]
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(ALL_EMOTIONS)}
N_EMOTIONS = len(ALL_EMOTIONS)

TEST_SIZE = 0.2
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 64
EMB_DIM = 1024 # 训练代码中硬编码的值

# ═══════════＝ 2. Device Configuration ＝═══════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"> Using device: {device}")

# ═══════════＝ 3. Model Definition (Must match training) ＝═══════════
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
        z_emb = self.emb_branch(emb)
        z_genre = self.genre_branch(genre)
        z_audio = self.audio_feature_branch(audio_feats)
        combined = torch.cat([z_mel, z_emb, z_genre, z_audio], dim=1); output = self.fusion_head(combined)
        return output

# ═══════════＝ 4. Dataset and Metric ＝═══════════
class RankingDataset(Dataset):
    def __init__(self, dataframe, mel_dir, muq_dir, feature_cols, n_genres, genre_to_idx, emotion_to_idx):
        self.df = dataframe.reset_index(drop=True)
        self.mel_dir = mel_dir; self.muq_dir = muq_dir; self.feature_cols = feature_cols
        self.n_genres = n_genres; self.genre_to_idx = genre_to_idx
        self.emotion_to_idx = emotion_to_idx

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]; file_id = row.id
        mel_path = os.path.join(self.mel_dir, f"{file_id}.npy")
        mel_spec = torch.from_numpy(np.load(mel_path).astype(np.float32))
        
        emb_path = os.path.join(self.muq_dir, f"{file_id}.npy")
        embedding = torch.from_numpy(np.load(emb_path).astype(np.float32))
        
        genre_vector = torch.zeros(self.n_genres, dtype=torch.float32)
        genre = row.major_genre_name_en
        if genre in self.genre_to_idx: genre_vector[self.genre_to_idx[genre]] = 1.0
            
        audio_features = torch.tensor(row[self.feature_cols].values.astype(np.float32), dtype=torch.float32)
        
        emotion_str = row.emotion_sequence
        if pd.isna(emotion_str) or not emotion_str:
            emotion_indices = []
        else:
            emotions = emotion_str.split(',')
            emotion_indices = [self.emotion_to_idx[emo] for emo in emotions if emo in self.emotion_to_idx]
        
        label_seq = torch.tensor(emotion_indices, dtype=torch.long)
        original_emotion_list = emotion_str.split(',') if pd.notna(emotion_str) and emotion_str else []

        return mel_spec, embedding, genre_vector, audio_features, label_seq, original_emotion_list

def ranking_collate_fn(batch):
    mels, embs, genres, audios, label_seqs, original_lists = zip(*batch)
    
    max_len_mel = max(mel.shape[1] for mel in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len_mel)
    for i, mel in enumerate(mels): padded_mels[i, :, :mel.shape[1]] = mel
    
    embs_collated = default_collate(embs)
    genres_collated = default_collate(genres)
    audios_collated = default_collate(audios)
    
    max_len_seq = max(len(seq) for seq in label_seqs if len(seq) > 0) if any(len(seq) > 0 for seq in label_seqs) else 0
    padded_labels = torch.full((len(label_seqs), max_len_seq), -1, dtype=torch.long)
    for i, seq in enumerate(label_seqs):
        if len(seq) > 0:
            padded_labels[i, :len(seq)] = seq
            
    return padded_mels, embs_collated, genres_collated, audios_collated, padded_labels, original_lists

def ndcg_at_k(y_scores, y_true_lists, k=3):
    batch_ndcg = []
    for scores, true_list in zip(y_scores, y_true_lists):
        if not true_list:
            continue

        relevance = torch.zeros_like(scores)
        true_indices = [EMOTION_TO_IDX[emo] for emo in true_list if emo in EMOTION_TO_IDX]
        if not true_indices: continue
        relevance[true_indices] = 1.0

        _, top_k_indices = torch.topk(scores, k=k)

        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            dcg += relevance[idx] / torch.log2(torch.tensor(i + 2.0, device=scores.device))

        idcg = 0.0
        num_true = min(k, len(true_indices))
        for i in range(num_true):
             idcg += 1.0 / torch.log2(torch.tensor(i + 2.0, device=scores.device))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        batch_ndcg.append(ndcg)
        
    return torch.mean(torch.tensor(batch_ndcg)) if batch_ndcg else torch.tensor(0.0)

# ═══════════＝ 5. Evaluation Function ＝═══════════
def evaluate_subset(model, df_subset, feature_cols, n_genres, genre_to_idx, subset_name):
    if len(df_subset) == 0:
        print(f"[{subset_name}] No samples found.")
        return 0.0

    dataset = RankingDataset(df_subset, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx, EMOTION_TO_IDX)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=ranking_collate_fn, num_workers=0)
    
    model.eval()
    all_scores = []
    all_true_lists = []
    
    with torch.no_grad():
        for mel, emb, genre, audio, _, original_lists in loader:
            if mel is None: continue
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            outputs = model(mel, emb, genre, audio)
            all_scores.append(outputs.cpu())
            all_true_lists.extend(original_lists)
    
    if not all_scores:
        print(f"[{subset_name}] No valid batches.")
        return 0.0

    scores_tensor = torch.cat(all_scores, dim=0)
    ndcg_3 = ndcg_at_k(scores_tensor, all_true_lists, k=3).item()
    
    print(f"[{subset_name}] Samples: {len(df_subset)} | nDCG@3: {ndcg_3:.4f}")
    return ndcg_3

# ═══════════＝ 6. Main Logic ＝═══════════
if __name__ == '__main__':
    print("--- > Loading Data and Recreating Splits ---")
    try: 
        df = pd.read_csv(DATASET_CSV)
    except FileNotFoundError as e: 
        print(f"FATAL ERROR: CSV file not found -> {e.filename}"); sys.exit()
    
    df.dropna(subset=['id', 'emotion_sequence', 'major_genre_name_en'], inplace=True)
    df = df[df['id'] != 'id'].copy(); df['id'] = df['id'].astype(str)
    
    # 检查缺失文件 (保持与训练一致的数据过滤逻辑)
    missing_mels = {sid for sid in df['id'] if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    missing_muqs = {sid for sid in df['id'] if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
    all_missing = missing_mels.union(missing_muqs)
    if all_missing:
        df = df[~df['id'].isin(all_missing)]

    feature_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 
                      'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    
    all_genres_list = sorted(df["major_genre_name_en"].unique())
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    N_GENRES = len(all_genres_list)

    # Recreate the splits using the SAME random_state as training
    train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=VALIDATION_FRACTION, random_state=42)
    
    # Combine Val and Test
    val_test_df = pd.concat([val_df, test_df])
    print(f"Total Val+Test samples: {len(val_test_df)}")

    # 1. Create Romantic Subset (Songs containing 'Romantic' in emotion_sequence)
    # emotion_sequence 格式如 "Sadness,Romantic"
    romantic_subset = val_test_df[val_test_df['emotion_sequence'].apply(lambda x: 'Romantic' in str(x).split(','))]
    
    # 2. Create Jazz + Romantic Subset
    jazz_romantic_subset = romantic_subset[romantic_subset['major_genre_name_en'] == 'Jazz']

    # Load Model
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Model file {BEST_MODEL_PATH} not found.")
        sys.exit()

    model = QuadFusionNet(EMB_DIM, N_GENRES, len(feature_cols), N_EMOTIONS).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print("Model loaded successfully.")

    # Run Evaluations
    print(f"\n{'='*10} EVALUATION RESULTS (Val + Test Set) {'='*10}")
    
    evaluate_subset(model, romantic_subset, feature_cols, N_GENRES, genre_to_idx, "Subset: Romantic")
    evaluate_subset(model, jazz_romantic_subset, feature_cols, N_GENRES, genre_to_idx, "Subset: Jazz + Romantic")
    
    # Optional: Evaluate Full Val+Test for reference
    # evaluate_subset(model, val_test_df, feature_cols, N_GENRES, genre_to_idx, "Full Val+Test Set")