# ─────────────────── Inference Script with nDCG Check ───────────────────
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm

# ═══════════＝ 1. 配置 ＝═══════════
MUQ_EMB_DIR = "G:\\13kmid30s_muq"
MEL_FEAT_DIR = "G:\\13kmid30smel"
DATASET_CSV = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon.csv"
MODEL_PATH = "best_ranking_model_by_ndcg.pth"
OUTPUT_CSV = "full_model_predictions.csv"

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

# ═══════════＝ 2. nDCG 计算函数 (直接从训练脚本搬来的) ＝═══════════
def ndcg_at_k(y_scores, y_true_lists, k=3):
    batch_ndcg = []
    # y_scores is numpy array here, y_true_lists is list of lists
    for scores, true_list in zip(y_scores, y_true_lists):
        if not true_list:
            continue

        # Build relevance vector (1.0 for true emotions, 0.0 others)
        relevance = np.zeros(len(scores))
        true_indices = [EMOTION_TO_IDX[emo] for emo in true_list if emo in EMOTION_TO_IDX]
        if not true_indices: continue
        relevance[true_indices] = 1.0

        # Get top K indices
        top_k_indices = np.argsort(scores)[::-1][:k]

        # Calculate DCG
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            rel = relevance[idx]
            dcg += rel / np.log2(i + 2.0)

        # Calculate IDCG
        idcg = 0.0
        num_true = min(k, len(true_indices))
        for i in range(num_true):
            idcg += 1.0 / np.log2(i + 2.0)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        batch_ndcg.append(ndcg)
        
    return np.mean(batch_ndcg) if batch_ndcg else 0.0

# ═══════════＝ 3. 模型定义 ＝═══════════
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

# ═══════════＝ 4. 数据集 (修改：返回真值标签) ＝═══════════
class InferenceDataset(Dataset):
    def __init__(self, dataframe, mel_dir, muq_dir, feature_cols, n_genres, genre_to_idx):
        self.df = dataframe.reset_index(drop=True)
        self.mel_dir = mel_dir; self.muq_dir = muq_dir; self.feature_cols = feature_cols
        self.n_genres = n_genres; self.genre_to_idx = genre_to_idx

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]; file_id = str(row.id)
        
        mel_path = os.path.join(self.mel_dir, f"{file_id}.npy")
        mel_spec = torch.from_numpy(np.load(mel_path).astype(np.float32))
        emb_path = os.path.join(self.muq_dir, f"{file_id}.npy")
        embedding = torch.from_numpy(np.load(emb_path).astype(np.float32))
        
        genre_vector = torch.zeros(self.n_genres, dtype=torch.float32)
        genre = row.major_genre_name_en
        if genre in self.genre_to_idx: genre_vector[self.genre_to_idx[genre]] = 1.0
            
        audio_features = torch.tensor(row[self.feature_cols].values.astype(np.float32), dtype=torch.float32)

        # 【新增】解析真实情感标签，用于计算 nDCG
        emotion_str = row.emotion_sequence
        if pd.isna(emotion_str) or not emotion_str:
            true_emotions = []
        else:
            true_emotions = emotion_str.split(',')
        
        return mel_spec, embedding, genre_vector, audio_features, file_id, true_emotions

def inference_collate_fn(batch):
    mels, embs, genres, audios, ids, true_emotions = zip(*batch)
    max_len_mel = max(mel.shape[1] for mel in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len_mel)
    for i, mel in enumerate(mels): padded_mels[i, :, :mel.shape[1]] = mel
    return padded_mels, default_collate(embs), default_collate(genres), default_collate(audios), ids, true_emotions

# ═══════════＝ 5. 主逻辑 ＝═══════════
def main():
    print("--- > Loading Data ---")
    try: df = pd.read_csv(DATASET_CSV)
    except Exception as e: print(e); sys.exit()
    
    df.dropna(subset=['id', 'major_genre_name_en'], inplace=True)
    df = df[df['id'] != 'id'].copy(); df['id'] = df['id'].astype(str)
    
    # Check files
    missing_mels = {sid for sid in df['id'] if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    missing_muqs = {sid for sid in df['id'] if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
    all_missing = missing_mels.union(missing_muqs)
    if all_missing: df = df[~df['id'].isin(all_missing)]
    
    print(f"> Processing {len(df)} records...")
    
    feature_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 
                      'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    all_genres_list = sorted(df["major_genre_name_en"].unique())
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    
    dataset = InferenceDataset(df, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, len(all_genres_list), genre_to_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=inference_collate_fn, num_workers=0)
    
    model = QuadFusionNet(EMB_DIM, len(all_genres_list), len(feature_cols), N_EMOTIONS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    results = []
    all_pred_scores = []
    all_true_lists = []

    print("> Starting Inference...")
    with torch.no_grad():
        for mel, emb, genre, audio, file_ids, true_emos_batch in tqdm(loader):
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            outputs = model(mel, emb, genre, audio)
            scores = outputs.cpu().numpy()
            
            # 收集用于计算整体 nDCG 的数据
            all_pred_scores.append(scores)
            all_true_lists.extend(true_emos_batch)
            
            # 保存 CSV 结果
            for i, file_id in enumerate(file_ids):
                row_scores = scores[i]
                top_indices = np.argsort(row_scores)[::-1][:3]
                top_emotions = [IDX_TO_EMOTION[idx] for idx in top_indices]
                
                record = {'id': file_id}
                record['pred_top1'] = top_emotions[0]
                record['pred_top2'] = top_emotions[1]
                record['pred_top3'] = top_emotions[2]
                results.append(record)

    # ═══════════ 计算全量 nDCG ═══════════
    print("\n> Calculating Full Dataset nDCG...")
    full_pred_scores = np.concatenate(all_pred_scores, axis=0)
    full_ndcg3 = ndcg_at_k(full_pred_scores, all_true_lists, k=3)
    
    print(f"\n{'='*40}")
    print(f"📊 FULL DATASET nDCG@3: {full_ndcg3:.4f}")
    print(f"{'='*40}\n")
    
    # Save CSV
    res_df = pd.DataFrame(results)
    final_df = df[['id', 'major_genre_name_en', 'emotion_sequence']].merge(res_df, on='id', how='inner')
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"> Results saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()