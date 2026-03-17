# ─────────────────── Import Libraries ───────────────────
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate

# ═══════════ 配置路径 (请确认无误) ═══════════
MUQ_EMB_DIR = "G:\\13kmid30s_muq"
MEL_FEAT_DIR = "G:\\13kmid30smel"
DATASET_CSV = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon.csv"
BEST_MODEL_PATH = "best_ranking_model_by_ndcg.pth"
OUTPUT_FILE = "music_data.npz" # 结果将保存到这里
EMB_DIM = 1024

# ═══════════ 修正后的模型定义 ═══════════
# 必须与训练代码的层结构完全一致 (包括 Dropout)，否则 state_dict 索引会对不上
ALL_EMOTIONS = ['Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 'Quiet',
                'Happiness', 'Loneliness', 'Touching', 'Missing', 'Fresh', 'Relaxation']
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(ALL_EMOTIONS)}

class QuadFusionNet(nn.Module):
    def __init__(self, emb_dim, genre_dim, audio_feature_dim, n_classes):
        super().__init__()
        # 【关键修复】补回 Dropout 层，确保索引与 checkpoint 一致
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
        
        # 为了提取 Embedding，我们将 fusion_head 拆分定义
        # layer1 对应原 fusion_head 的前3层 (Linear -> BN -> ReLU)
        # 注意：原 fusion_head 的第3层是 ReLU，第4层是 Dropout。
        # 这里 layer1 包含 [0, 1, 2]，输出 128 维特征。
        self.fusion_layer1 = nn.Sequential(
            nn.Linear(fusion_input_dim, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU()
        )
        # layer2 包含剩余部分，仅用于占位以避免权重加载报错，实际推理不用它
        # 对应原 fusion_head 的 [3, 4, 5, 6, 7, 8, 9]
        self.fusion_layer2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, n_classes)
        )

    def forward(self, mel, emb, genre, audio_feats):
        mel = mel.unsqueeze(1)
        z_mel = self.mel_cnn_branch(mel)
        z_mel = z_mel.view(z_mel.size(0), -1)
        z_emb = self.emb_branch(emb)
        z_genre = self.genre_branch(genre)
        z_audio = self.audio_feature_branch(audio_feats)
        
        combined = torch.cat([z_mel, z_emb, z_genre, z_audio], dim=1)
        # 只跑前半截，拿到 Embedding 就返回
        return self.fusion_layer1(combined)

class RankingDataset(Dataset):
    def __init__(self, df, mel_dir, muq_dir, cols, n_genres, g_idx):
        self.df = df.reset_index(drop=True)
        self.mel_dir = mel_dir; self.muq_dir = muq_dir; self.cols = cols
        self.n_genres = n_genres; self.g_idx = g_idx
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        mel = torch.from_numpy(np.load(os.path.join(self.mel_dir, f"{row.id}.npy")).astype(np.float32))
        emb = torch.from_numpy(np.load(os.path.join(self.muq_dir, f"{row.id}.npy")).astype(np.float32))
        g_vec = torch.zeros(self.n_genres, dtype=torch.float32)
        if row.major_genre_name_en in self.g_idx: g_vec[self.g_idx[row.major_genre_name_en]] = 1.0
        audio = torch.tensor(row[self.cols].values.astype(np.float32), dtype=torch.float32)
        emo = row.emotion_sequence.split(',')[0] if pd.notna(row.emotion_sequence) else "Unknown"
        return mel, emb, g_vec, audio, row.major_genre_name_en, emo, str(row.id)

def simple_collate(batch):
    mels, embs, genres, audios, g_labels, e_labels, ids = zip(*batch)
    max_len = max(m.shape[1] for m in mels)
    padded = torch.zeros(len(mels), mels[0].shape[0], max_len)
    for i, m in enumerate(mels): padded[i, :, :m.shape[1]] = m
    return padded, default_collate(embs), default_collate(genres), default_collate(audios), g_labels, e_labels, ids

# ═══════════ 主程序 ═══════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading Data...")
    try:
        df = pd.read_csv(DATASET_CSV)
    except FileNotFoundError:
        print(f"ERROR: CSV not found: {DATASET_CSV}"); sys.exit()
        
    df.dropna(subset=['id', 'emotion_sequence', 'major_genre_name_en'], inplace=True)
    df = df[df['id'] != 'id']
    
    # 填充 NaNs
    feat_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    df[feat_cols] = df[feat_cols].fillna(0.0)

    # 完整流派列表 (确保与训练一致)
    all_genres = sorted(df["major_genre_name_en"].unique())
    g_to_idx = {g: i for i, g in enumerate(all_genres)}
    print(f"Detected {len(all_genres)} genres.")
    
    dataset = RankingDataset(df, MEL_FEAT_DIR, MUQ_EMB_DIR, feat_cols, len(all_genres), g_to_idx)
    loader = DataLoader(dataset, batch_size=64, collate_fn=simple_collate, num_workers=0)

    print("Loading Model...")
    model = QuadFusionNet(EMB_DIM, len(all_genres), len(feat_cols), 12).to(device)
    
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"ERROR: Model file not found: {BEST_MODEL_PATH}"); sys.exit()

    # 权重加载逻辑 (带 Key Mapping)
    sd = torch.load(BEST_MODEL_PATH, map_location=device)
    new_sd = {}
    for k, v in sd.items():
        # 映射 fusion_head 的前三层到 layer1
        if "fusion_head.0" in k: new_sd[k.replace("fusion_head.0", "fusion_layer1.0")] = v
        elif "fusion_head.1" in k: new_sd[k.replace("fusion_head.1", "fusion_layer1.1")] = v
        elif "fusion_head.2" in k: new_sd[k.replace("fusion_head.2", "fusion_layer1.2")] = v
        # 剩下的映射到 layer2 (虽然我们不跑它，但为了 load_state_dict 不报错)
        elif "fusion_head.3" in k: new_sd[k.replace("fusion_head.3", "fusion_layer2.0")] = v
        elif "fusion_head.4" in k: new_sd[k.replace("fusion_head.4", "fusion_layer2.1")] = v
        elif "fusion_head.5" in k: new_sd[k.replace("fusion_head.5", "fusion_layer2.2")] = v
        elif "fusion_head.6" in k: new_sd[k.replace("fusion_head.6", "fusion_layer2.3")] = v
        elif "fusion_head.7" in k: new_sd[k.replace("fusion_head.7", "fusion_layer2.4")] = v
        elif "fusion_head.8" in k: new_sd[k.replace("fusion_head.8", "fusion_layer2.5")] = v
        elif "fusion_head.9" in k: new_sd[k.replace("fusion_head.9", "fusion_layer2.6")] = v
        else:
            new_sd[k] = v
            
    model.load_state_dict(new_sd, strict=True) # 现在结构一致了，可以使用 strict=True
    model.eval()

    print("Extracting Embeddings (All Data)...")
    all_embs, all_genres, all_emos, all_ids = [], [], [], []
    
    with torch.no_grad():
        for i, (mel, emb, g_vec, audio, g_lbls, e_lbls, ids) in enumerate(loader):
            if i % 10 == 0: print(f"Processing batch {i}...")
            mel, emb, g_vec, audio = mel.to(device), emb.to(device), g_vec.to(device), audio.to(device)
            feats = model(mel, emb, g_vec, audio).cpu().numpy()
            all_embs.append(feats); all_genres.extend(g_lbls); all_emos.extend(e_lbls); all_ids.extend(ids)
    
    X = np.concatenate(all_embs, axis=0)
    # 最后清洗一次 NaN，以防万一
    X = np.nan_to_num(X, nan=0.0) 
    
    print(f"Saving {len(X)} samples to {OUTPUT_FILE}...")
    np.savez(OUTPUT_FILE, embeddings=X, genres=np.array(all_genres), emotions=np.array(all_emos), ids=np.array(all_ids))
    print("Done! You can now run step2_gui_cluster.py")