# ─────────────────── Import Libraries ───────────────────
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════＝ 1. Configurations (请确保路径正确) ＝═══════════
MUQ_EMB_DIR = "G:\\13kmid30s_muq"
MEL_FEAT_DIR = "G:\\13kmid30smel"
DATASET_CSV = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon.csv"
BEST_MODEL_PATH = "best_ranking_model_by_ndcg.pth" 

# 可视化参数
SAMPLE_SIZE = 3000  
BATCH_SIZE = 64
EMB_DIM = 1024      

# ═══════════＝ 2. 类定义 (Model & Dataset) ＝═══════════

ALL_EMOTIONS = [
    'Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 'Quiet',
    'Happiness', 'Loneliness', 'Touching', 'Missing', 'Fresh', 'Relaxation'
]
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(ALL_EMOTIONS)}

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
        
        # Split fusion head
        self.fusion_layer1 = nn.Sequential(nn.Linear(fusion_input_dim, 128), nn.BatchNorm1d(128), nn.ReLU())
        self.fusion_layer2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, n_classes))

    def forward(self, mel, emb, genre, audio_feats, return_embedding=False):
        mel = mel.unsqueeze(1)
        z_mel = self.mel_cnn_branch(mel)
        z_mel = z_mel.view(z_mel.size(0), -1)
        z_emb = self.emb_branch(emb)
        z_genre = self.genre_branch(genre)
        z_audio = self.audio_feature_branch(audio_feats)
        
        combined = torch.cat([z_mel, z_emb, z_genre, z_audio], dim=1)
        
        embedding_vector = self.fusion_layer1(combined)
        
        if return_embedding:
            return embedding_vector
        else:
            return self.fusion_layer2(embedding_vector)

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
        primary_emotion = "Unknown"
        if pd.notna(emotion_str) and emotion_str:
            primary_emotion = emotion_str.split(',')[0]
            
        return mel_spec, embedding, genre_vector, audio_features, genre, primary_emotion

def simple_collate(batch):
    mels, embs, genres, audios, genre_labels, emotion_labels = zip(*batch)
    max_len = max(mel.shape[1] for mel in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len)
    for i, mel in enumerate(mels): padded_mels[i, :, :mel.shape[1]] = mel
    return padded_mels, default_collate(embs), default_collate(genres), default_collate(audios), genre_labels, emotion_labels

# ═══════════＝ 3. 主逻辑 (Main Execution) ＝═══════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"> Using device: {device}")

    # 1. 加载完整数据
    print("Loading full dataset...")
    try:
        df = pd.read_csv(DATASET_CSV)
    except FileNotFoundError:
        print(f"Error: CSV not found at {DATASET_CSV}"); sys.exit()

    df.dropna(subset=['id', 'emotion_sequence', 'major_genre_name_en'], inplace=True)
    df = df[df['id'] != 'id'] 
    
    # ───【修复 1：填充特征列的 NaN】───
    # 防止 Audio Features 为空导致模型输出 NaN
    feature_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 
                      'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    
    # 检查并填充 (将空值填为 0)
    print("Checking for NaNs in feature columns...")
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    # 计算全量流派 (保持与训练一致)
    all_genres_list = sorted(df["major_genre_name_en"].unique())
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    N_GENRES = len(all_genres_list)
    print(f"> Detected {N_GENRES} genres (should be 23).")

    # 2. 采样数据
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"> Downsampled to {SAMPLE_SIZE} samples.")
    
    dataset = RankingDataset(df, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, N_GENRES, genre_to_idx, EMOTION_TO_IDX)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=simple_collate, shuffle=False)

    # 3. 加载模型
    print("Loading model...")
    model = QuadFusionNet(EMB_DIM, N_GENRES, len(feature_cols), len(ALL_EMOTIONS)).to(device)
    
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Model not found at {BEST_MODEL_PATH}"); sys.exit()

    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if "fusion_head.0" in k: new_state_dict[k.replace("fusion_head.0", "fusion_layer1.0")] = v
        elif "fusion_head.1" in k: new_state_dict[k.replace("fusion_head.1", "fusion_layer1.1")] = v
        elif "fusion_head.2" in k: new_state_dict[k.replace("fusion_head.2", "fusion_layer1.2")] = v
        elif "fusion_head.3" in k: new_state_dict[k.replace("fusion_head.3", "fusion_layer2.0")] = v
        elif "fusion_head.4" in k: new_state_dict[k.replace("fusion_head.4", "fusion_layer2.1")] = v
        elif "fusion_head.5" in k: new_state_dict[k.replace("fusion_head.5", "fusion_layer2.2")] = v
        elif "fusion_head.6" in k: new_state_dict[k.replace("fusion_head.6", "fusion_layer2.3")] = v
        elif "fusion_head.7" in k: new_state_dict[k.replace("fusion_head.7", "fusion_layer2.4")] = v
        elif "fusion_head.8" in k: new_state_dict[k.replace("fusion_head.8", "fusion_layer2.5")] = v
        elif "fusion_head.9" in k: new_state_dict[k.replace("fusion_head.9", "fusion_layer2.6")] = v
        else: new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()

    # 4. 提取 Embeddings
    print("Extracting embeddings...")
    extracted_embs = []
    labels_genre = []
    labels_emotion = []

    with torch.no_grad():
        for i, (mel, emb, genre_vec, audio, g_labels, e_labels) in enumerate(loader):
            mel, emb, genre_vec, audio = mel.to(device), emb.to(device), genre_vec.to(device), audio.to(device)
            features = model(mel, emb, genre_vec, audio, return_embedding=True)
            extracted_embs.append(features.cpu().numpy())
            labels_genre.extend(g_labels)
            labels_emotion.extend(e_labels)

    X = np.concatenate(extracted_embs, axis=0)
    
    # ───【修复 2：输出层清洗】───
    # 检查是否有 NaN，如果有，替换为 0，防止 t-SNE 报错
    if np.isnan(X).any():
        print(f"Warning: Found NaNs in extracted embeddings. Replacing with 0. (Count: {np.isnan(X).sum()})")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
    print(f"Embedding shape: {X.shape}")

    # 5. 运行 t-SNE
    print("Running t-SNE (this might take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    
    # 现在 X 是安全的
    X_embedded = tsne.fit_transform(X)

    # 6. 画图
    print("Plotting results...")
    
    plot_df = pd.DataFrame(X_embedded, columns=['x', 'y'])
    plot_df['Genre'] = labels_genre
    plot_df['Primary Emotion'] = labels_emotion

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: Genre
    sns.scatterplot(
        data=plot_df, x='x', y='y', hue='Genre', 
        palette='tab10', s=50, alpha=0.7, ax=axes[0]
    )
    axes[0].set_title("Song Embeddings by Genre")
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, ncol=2, fontsize='small')

    # Right: Emotion
    top_emotions = plot_df['Primary Emotion'].value_counts().nlargest(12).index
    plot_df_emo = plot_df[plot_df['Primary Emotion'].isin(top_emotions)]
    
    sns.scatterplot(
        data=plot_df_emo, x='x', y='y', hue='Primary Emotion', 
        palette='plasma', s=50, alpha=0.7, ax=axes[1]
    )
    axes[1].set_title("Song Embeddings by Primary Emotion")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout()
    plt.show()
    print("Done! Check charts.")