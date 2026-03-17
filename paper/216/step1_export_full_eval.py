import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.model_selection import train_test_split 
from tqdm import tqdm

# ════════════════════ 1. 配置区域 ════════════════════
# 请根据你的实际路径修改
MUQ_EMB_DIR = r"G:\13kmid30s_muq"
MEL_FEAT_DIR = r"G:\13kmid30smel"
DATASET_CSV = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon.csv"
MODEL_PATH = "best_ranking_model_by_ndcg.pth" 

# 所有的情感列表 (顺序必须固定)
ALL_EMOTIONS = [
    'Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 'Quiet',
    'Happiness', 'Loneliness', 'Touching', 'Missing', 'Fresh', 'Relaxation'
]

# 划分参数 (必须与训练一致)
TEST_SIZE = 0.2
VALIDATION_FRACTION = 0.2
RANDOM_STATE = 42

BATCH_SIZE = 64
EMB_DIM = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════ 2. 模型与数据类 (保持不变) ════════════════════
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
        mel_path = os.path.join(self.mel_dir, f"{file_id}.npy")
        muq_path = os.path.join(self.muq_dir, f"{file_id}.npy")
        
        mel_spec = torch.from_numpy(np.load(mel_path).astype(np.float32))
        embedding = torch.from_numpy(np.load(muq_path).astype(np.float32))
        
        genre_vector = torch.zeros(self.n_genres)
        if row.major_genre_name_en in self.genre_to_idx: genre_vector[self.genre_to_idx[row.major_genre_name_en]] = 1.0
        audio_features = torch.tensor(row[self.feature_cols].values.astype(np.float32))
        true_emotions = row.emotion_sequence
        return mel_spec, embedding, genre_vector, audio_features, file_id, true_emotions, row.major_genre_name_en

def collate_fn(batch):
    mels, embs, genres, audios, ids, labels, genre_names = zip(*batch)
    max_len = max(m.shape[1] for m in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len)
    for i, m in enumerate(mels): padded_mels[i, :, :m.shape[1]] = m
    return padded_mels, default_collate(embs), default_collate(genres), default_collate(audios), ids, labels, genre_names

# ════════════════════ 3. 执行流程 ════════════════════
if __name__ == '__main__':
    print("--- > Starting Full Eval Inference ---")
    
    # 1. Load CSV
    try: 
        df = pd.read_csv(DATASET_CSV)
    except FileNotFoundError as e: 
        print(f"FATAL ERROR: CSV file not found -> {e.filename}"); sys.exit()
    
    # 2. Clean Data
    df.dropna(subset=['id', 'emotion_sequence', 'major_genre_name_en'], inplace=True)
    df = df[df['id'] != 'id'].copy(); df['id'] = df['id'].astype(str)
    
    missing_mels = {sid for sid in df['id'] if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    missing_muqs = {sid for sid in df['id'] if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
    all_missing = missing_mels.union(missing_muqs)
    
    if all_missing:
        df = df[~df['id'].isin(all_missing)]

    # 3. Split Data (Eval Set Only)
    train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df, val_df = train_test_split(train_val_df, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE)
    
    # Eval Set (Validation + Test)
    eval_df = pd.concat([val_df, test_df])
    print(f"> Processing Eval Set: {len(eval_df)} records")

    # 4. Setup
    feature_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 
                    'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    
    full_genre_list = sorted(df["major_genre_name_en"].unique())
    genre_to_idx = {g: i for i, g in enumerate(full_genre_list)}
    
    dataset = FullInferenceDataset(eval_df, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, len(full_genre_list), genre_to_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
    
    # 5. Model
    model = QuadFusionNet(EMB_DIM, len(full_genre_list), len(feature_cols), len(ALL_EMOTIONS)).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found."); sys.exit()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 6. Inference Loop
    all_ids = []
    all_embeddings = []
    all_pred_scores = [] # Store all 12 emotion scores
    all_true_labels = []
    all_genre_names = []

    print("> Running Inference...")
    with torch.no_grad():
        for mel, emb, genre, audio, ids, labels, g_names in tqdm(loader):
            outputs = model(mel.to(device), emb.to(device), genre.to(device), audio.to(device))
            
            # Store everything on CPU
            all_pred_scores.append(outputs.cpu().numpy())
            all_embeddings.append(emb.cpu().numpy())
            all_ids.extend(ids)
            all_true_labels.extend(labels)
            all_genre_names.extend(g_names)

    # 7. Concatenate and Save
    all_pred_scores = np.concatenate(all_pred_scores, axis=0) # (N, 12)
    all_embeddings = np.concatenate(all_embeddings, axis=0)   # (N, 1024)
    all_ids = np.array(all_ids)
    all_genre_names = np.array(all_genre_names)
    # True labels is a list of strings, keep as object array or pickling
    
    print("> Saving full_eval_results.npz...")
    np.savez("full_eval_results.npz", 
             ids=all_ids,
             embeddings=all_embeddings,
             scores=all_pred_scores,
             genres=all_genre_names,
             true_labels=all_true_labels, # This might pickle
             emotion_names=ALL_EMOTIONS)
    
    print("  Done! Size of saved file should be manageable.")
    print("  Now run the 'Explorer UI' script.")