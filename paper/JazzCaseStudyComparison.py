# ─────────────────── Import Libraries ───────────────────
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score  # <--- 新增指标库
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════＝  1. Configurations (请确保路径正确) ＝═══════════
MEL_FEAT_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\mel"
MUQ_EMB_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\muq"
AUDIO_FEATURES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\audio_features_normalized.csv"
GENRES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\processed_genres.csv"
LABELS_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\quadrants_from_dynamic.csv"
BEST_MODEL_PATH = "best_model_quad_fusion-1.pth"  # 确保此文件在当前目录下

BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶ Using device: {device}")

# ═══════════＝  2. Model Definition (QuadFusionNet) ＝═══════════
class QuadFusionNet(nn.Module):
    def __init__(self, emb_dim, genre_dim, audio_feature_dim, n_classes):
        super().__init__()
        
        # Branch 1: Mel-spectrogram CNN
        self.mel_cnn_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Branch 2: MuQ Embedding
        self.emb_branch = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5)
        )
        
        # Branch 3: Genre
        self.genre_branch = nn.Sequential(
            nn.Linear(genre_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5)
        )
        
        # Branch 4: Audio Features
        self.audio_feature_branch = nn.Sequential(
            nn.BatchNorm1d(audio_feature_dim), nn.ReLU(), nn.Dropout(0.5)
        )

        # Fusion Head (Reconstructed to match the saved model keys "fusion_head.x")
        cnn_output_dim = 256
        emb_output_dim = 128
        genre_output_dim = 32
        fusion_input_dim = cnn_output_dim + emb_output_dim + genre_output_dim + audio_feature_dim
        
        # 你的模型里 fusion_head 是一个整体 Sequential
        # 根据错误提示：0是Linear, 1是BN, 4是Linear, 5是BN, 8是Linear
        # 推测结构如下：
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),      # layer 0
            nn.BatchNorm1d(128),                   # layer 1
            nn.ReLU(),                             # layer 2
            nn.Dropout(0.4),                       # layer 3
            nn.Linear(128, 64),                    # layer 4
            nn.BatchNorm1d(64),                    # layer 5
            nn.ReLU(),                             # layer 6
            nn.Dropout(0.3),                       # layer 7
            nn.Linear(64, n_classes)               # layer 8 (Classifier)
        )

    def forward(self, mel, emb, genre, audio_feats, return_embeds=False):
        mel = mel.unsqueeze(1)
        z_mel = self.mel_cnn_branch(mel)
        z_mel = z_mel.view(z_mel.size(0), -1)
        z_emb = self.emb_branch(emb)
        z_genre = self.genre_branch(genre)
        z_audio = self.audio_feature_branch(audio_feats)
        
        combined = torch.cat([z_mel, z_emb, z_genre, z_audio], dim=1)
        
        # 我们需要分别拿到 embedding (分类层之前的特征) 和 output (分类结果)
        # 这里的 embedding 取的是 fusion_head 中 layer 7 (Dropout) 的输出，也就是 layer 8 之前的输入
        # 对应的是 64维 的特征
        
        # 执行前8层 (0到7)
        embedding_feat = self.fusion_head[:-1](combined) 
        
        # 执行最后一层 (8)
        output = self.fusion_head[-1](embedding_feat)
        
        if return_embeds:
            return output, embedding_feat
        else:
            return output

# ═══════════＝  3. Dataset Class ＝═══════════
class QuadFusionDataset(Dataset):
    def __init__(self, dataframe, indices, mel_dir, muq_dir, feature_cols, n_genres, genre_to_idx):
        self.df = dataframe.iloc[indices]
        self.mel_dir = mel_dir
        self.muq_dir = muq_dir 
        self.feature_cols = feature_cols
        self.n_genres = n_genres
        self.genre_to_idx = genre_to_idx

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        song_id = str(row.song_id)
        
        mel_path = os.path.join(self.mel_dir, f"{song_id}.npy")
        mel_spec = torch.from_numpy(np.load(mel_path).astype(np.float32))
        
        emb_path = os.path.join(self.muq_dir, f"{song_id}.npy")
        embedding = torch.from_numpy(np.load(emb_path).astype(np.float32))
        if embedding.ndim > 1: embedding = embedding.squeeze()

        genre_vector = torch.zeros(self.n_genres, dtype=torch.float32)
        genres = str(row.normalized_genres).split(';')
        weights = 1 / np.arange(1, len(genres) + 1); weights /= weights.sum()
        for k, g in enumerate(genres):
            if g in self.genre_to_idx: genre_vector[self.genre_to_idx[g]] = weights[k]
        
        audio_features = torch.tensor(row[self.feature_cols].values.astype(np.float32), dtype=torch.float32)
        label = torch.tensor(row.label_idx, dtype=torch.long)
        
        return mel_spec, embedding, genre_vector, audio_features, label

def pad_collate_fn(batch):
    mels, embs, genres, audios, labels = zip(*batch)
    max_len = max(mel.shape[1] for mel in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len)
    for i, mel in enumerate(mels):
        length = mel.shape[1]
        padded_mels[i, :, :length] = mel
    return padded_mels, default_collate(embs), default_collate(genres), default_collate(audios), default_collate(labels)

# ═══════════＝  4. Main Execution Logic (Comparison) ＝═══════════
if __name__ == '__main__':
    print("\n" + "═"*10 + " STARTING JAZZ ABLATION STUDY " + "═"*10)
    
    # 1. Load Data
    try:
        df_labels = pd.read_csv(LABELS_CSV); df_genres = pd.read_csv(GENRES_CSV); df_audio_features = pd.read_csv(AUDIO_FEATURES_CSV)
    except FileNotFoundError: print("❌ CSV Files not found!"); sys.exit()
    
    df = pd.merge(df_labels, df_genres, on="song_id"); df = pd.merge(df, df_audio_features, on="song_id")
    df['song_id'] = pd.to_numeric(df['song_id'], errors='coerce'); df.dropna(subset=['song_id'], inplace=True); df['song_id'] = df['song_id'].astype(int).astype(str)
    df.dropna(subset=['normalized_genres'], inplace=True); df['normalized_genres'] = df['normalized_genres'].astype(str)
    
    # 2. Filter for JAZZ
    target_genre = "Jazz"
    print(f"▶ Filtering for genre: '{target_genre}'...")
    df_jazz = df[df['normalized_genres'].str.contains(target_genre, case=False, na=False)]
    
    if len(df_jazz) < 10:
        print(f"⚠️ Warning: Only {len(df_jazz)} Jazz songs found. Visualization might be sparse.")
        sys.exit()
    else:
        print(f"✅ Found {len(df_jazz)} Jazz songs.")

    # 3. Setup Labels & Mappings
    le = LabelEncoder()
    df_jazz = df_jazz.copy()
    
    quadrant_map = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}
    emotion_names = ['Happy', 'Tense', 'Sad', 'Relaxed'] # 0->Happy, 1->Tense...
    
    df_jazz["label_idx"] = df_jazz["Quadrant"].map(quadrant_map)
    df_jazz.dropna(subset=["label_idx"], inplace=True)
    df_jazz["label_idx"] = df_jazz["label_idx"].astype(int)
    
    y_jazz = df_jazz["label_idx"].values
    
    feature_cols = [col for col in df_audio_features.columns if col != 'song_id']
    all_genres_list = sorted({g for g in df.normalized_genres.str.cat(sep=';').split(';') if g})
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    N_GENRES = len(all_genres_list); EMB_DIM = 1024

    # 4. Load Model
    model = QuadFusionNet(EMB_DIM, N_GENRES, len(feature_cols), 4).to(device) 
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        print(f"✅ Model loaded from {BEST_MODEL_PATH}")
    else:
        print("❌ Model not found! Please run your training script first.")
        sys.exit()
        
    model.eval()
    
    jazz_dataset = QuadFusionDataset(df_jazz, np.arange(len(df_jazz)), MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, N_GENRES, genre_to_idx)
    jazz_loader = DataLoader(jazz_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    
    # 5. Extract Embeddings (Two Passes)
    
    # --- Pass 1: Without Genre ---
    print("▶ Pass 1: Extracting Embeddings with Genre MASKED...")
    emb_without_genre = []
    with torch.no_grad():
        for mel, emb, genre, audio, _ in jazz_loader:
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            genre_masked = torch.zeros_like(genre) 
            _, embeddings = model(mel, emb, genre_masked, audio, return_embeds=True)
            emb_without_genre.append(embeddings.cpu().numpy())
    X_without = np.concatenate(emb_without_genre)
    
    # --- Pass 2: With Genre (Full Model) ---
    print("▶ Pass 2: Extracting Embeddings with Full Model...")
    emb_with_genre = []
    with torch.no_grad():
        for mel, emb, genre, audio, _ in jazz_loader:
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            _, embeddings = model(mel, emb, genre, audio, return_embeds=True)
            emb_with_genre.append(embeddings.cpu().numpy())
    X_with = np.concatenate(emb_with_genre)
    
    # ─────────────────── CALCULATE CLUSTERING METRICS ───────────────────
    # 我们基于原始高维 Embedding 计算，这样更科学（t-SNE 只是降维可视化，会失真）
    print("\n" + "═"*10 + " CLUSTERING METRICS " + "═"*10)
    
    # Silhouette Score (Range: -1 to 1, Higher is Better)
    sil_score_no = silhouette_score(X_without, y_jazz)
    sil_score_yes = silhouette_score(X_with, y_jazz)
    
    # Calinski-Harabasz Index (Higher is Better)
    ch_score_no = calinski_harabasz_score(X_without, y_jazz)
    ch_score_yes = calinski_harabasz_score(X_with, y_jazz)
    
    print(f"Scenario A (Genre Masked) -> Silhouette: {sil_score_no:.4f} | CH-Index: {ch_score_no:.2f}")
    print(f"Scenario B (Genre Active) -> Silhouette: {sil_score_yes:.4f} | CH-Index: {ch_score_yes:.2f}")
    
    # 6. Visualization
    print("▶ Generating Plot...")
    sns.set_style("whitegrid") # 更清晰的网格背景
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(f"Impact of Genre Context on Jazz Emotion Clusters", fontsize=24, fontweight='bold', y=0.98)
    
    palette = sns.color_palette("bright", n_colors=4)
    
    # --- Plot 1 ---
    print("   Running t-SNE for Scenario A...")
    tsne_1 = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=min(30, len(df_jazz)-1))
    X_2d_without = tsne_1.fit_transform(X_without)
    
    sns.scatterplot(x=X_2d_without[:,0], y=X_2d_without[:,1], hue=y_jazz, palette=palette, style=y_jazz, s=150, ax=ax1, legend=False, edgecolor='k', alpha=0.8)
    
    # 把指标写在标题里
    title_str_a = (f"Scenario A: Without Genre Info\n"
                   f"Silhouette: {sil_score_no:.3f} | CH-Index: {ch_score_no:.1f}")
    ax1.set_title(title_str_a, fontsize=18, pad=15, color='#B22222') # 暗红色表示效果较差
    ax1.set_xlabel("Latent Dim 1", fontsize=14)
    ax1.set_ylabel("Latent Dim 2", fontsize=14)
    
    # --- Plot 2 ---
    print("   Running t-SNE for Scenario B...")
    tsne_2 = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=min(30, len(df_jazz)-1))
    X_2d_with = tsne_2.fit_transform(X_with)
    
    sns.scatterplot(x=X_2d_with[:,0], y=X_2d_with[:,1], hue=y_jazz, palette=palette, style=y_jazz, s=150, ax=ax2, edgecolor='k', alpha=0.9)
    
    # 把指标写在标题里
    title_str_b = (f"Scenario B: With Genre Info (Full Model)\n"
                   f"Silhouette: {sil_score_yes:.3f} | CH-Index: {ch_score_yes:.1f}")
    ax2.set_title(title_str_b, fontsize=18, pad=15, color='#006400') # 暗绿色表示效果较好
    ax2.set_xlabel("Latent Dim 1", fontsize=14)
    ax2.set_ylabel("Latent Dim 2", fontsize=14)
    
    # Legend
    handles, _ = ax2.get_legend_handles_labels()
    if len(handles) > 4: handles = handles[:4]
    ax2.legend(handles, emotion_names, title="Emotion Class", fontsize=14, title_fontsize=16, 
               loc='upper right', frameon=True, framealpha=0.95, shadow=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    print("✅ Visualization generated with Metrics!")
    plt.show()