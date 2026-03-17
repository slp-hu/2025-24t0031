# ─────────────────── Import Libraries ───────────────────
import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ═══════════＝  1. Configurations ＝═══════════
# 【请务必修改】你的原始音频目录 (.mp3/.wav 等)
RAW_AUDIO_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\audio_files" 

# 其他文件路径
MEL_FEAT_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\mel"
MUQ_EMB_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\muq"
AUDIO_FEATURES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\audio_features_normalized.csv"
GENRES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\processed_genres.csv"
LABELS_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\quadrants_from_dynamic.csv"
BEST_MODEL_PATH = "best_model_quad_fusion-1.pth" 

# 输出结果目录
ANALYSIS_OUTPUT_DIR = "analysis_cases_output"
TEST_SIZE = 0.2

# ═══════════＝  2. Device ＝═══════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶ Using device: {device}")

# ═══════════＝  3. Model Architecture (Must match training) ＝═══════════
class QuadFusionNet(nn.Module):
    def __init__(self, emb_dim, genre_dim, audio_feature_dim, n_classes):
        super().__init__()
        # Branch 1: Mel-Spectrogram CNN
        self.mel_cnn_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Branch 2: MuQ
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
        # Fusion Head
        cnn_output_dim = 256
        emb_output_dim = 128
        genre_output_dim = 32
        fusion_input_dim = cnn_output_dim + emb_output_dim + genre_output_dim + audio_feature_dim
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, mel, emb, genre, audio_feats):
        mel = mel.unsqueeze(1)
        z_mel = self.mel_cnn_branch(mel).view(mel.size(0), -1)
        z_emb = self.emb_branch(emb)
        z_genre = self.genre_branch(genre)
        z_audio = self.audio_feature_branch(audio_feats)
        combined = torch.cat([z_mel, z_emb, z_genre, z_audio], dim=1)
        return self.fusion_head(combined)

# ═══════════＝  4. Dataset Class ＝═══════════
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
        song_id = row.song_id
        
        mel_path = os.path.join(self.mel_dir, f"{song_id}.npy")
        mel_spec = torch.from_numpy(np.load(mel_path).astype(np.float32))
        
        emb_path = os.path.join(self.muq_dir, f"{song_id}.npy")
        embedding = torch.from_numpy(np.load(emb_path).astype(np.float32))
        
        genre_vector = torch.zeros(self.n_genres, dtype=torch.float32)
        genres = row.normalized_genres.split(';')
        weights = 1 / np.arange(1, len(genres) + 1); weights /= weights.sum()
        for k, g in enumerate(genres):
            if g in self.genre_to_idx: genre_vector[self.genre_to_idx[g]] = weights[k]
        
        audio_features = torch.tensor(row[self.feature_cols].values.astype(np.float32), dtype=torch.float32)
        label = torch.tensor(row.label_idx, dtype=torch.long)
        
        # 返回 6 个值，包含 song_id 用于复制文件
        return mel_spec, embedding, genre_vector, audio_features, label, song_id

# ═══════════＝  5. Core Logic: Spectrum Contrast Analysis ＝═══════════
def visualize_contrast_pair(model, dataset, class_names, label_map, idx_to_genre, 
                            raw_audio_dir, output_dir, device):
    """
    寻找双子星案例 (Spectrum Analysis):
    找到两首歌，它们的 Audio 模型都认为是 'Happy' (High Energy)。
    但加入 Genre 后，一个变成了 Tense (纠错)，一个保持 Happy (确认)。
    """
    print(f"\n🔍 [Spectrum Analysis] Searching for 'Visual Lookalikes' with Different Outcomes...")
    model.eval()
    
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    candidates_rap = []
    candidates_pop = []
    
    # 1. 自动识别标签索引
    idx_happy = -1; idx_tense = -1
    for i, name in enumerate(class_names):
        clean_name = label_map.get(name, name)
        if 'Happy' in clean_name: idx_happy = i
        if 'Tense' in clean_name: idx_tense = i
    
    print(f"   -> Target Indices: Happy={idx_happy}, Tense={idx_tense}")
    if idx_happy == -1 or idx_tense == -1:
        print("❌ Error: Cannot find Happy/Tense labels. Check CLASS_NAMES.")
        return

    # 2. 遍历搜索
    indices = list(range(len(dataset)))
    # np.random.shuffle(indices) # 如需随机结果可打开

    with torch.no_grad():
        for i in indices:
            # 找到一对就停止
            if len(candidates_rap) > 0 and len(candidates_pop) > 0: break 

            mel, emb, genre, audio, label_tensor, song_id = dataset[i]
            true_label_idx = label_tensor.item()
            
            # Prepare Batch
            mel_b = mel.unsqueeze(0).to(device)
            emb_b = emb.unsqueeze(0).to(device)
            genre_b = genre.unsqueeze(0).to(device)
            audio_b = audio.unsqueeze(0).to(device)
            
            # Audio-Only Prediction (Mask Genre)
            genre_zeros = torch.zeros_like(genre_b).to(device)
            pred_audio = model(mel_b, emb_b, genre_zeros, audio_b).argmax(1).item()
            
            # Full Model Prediction
            pred_full = model(mel_b, emb_b, genre_b, audio_b).argmax(1).item()
            
            # Genre String
            g_indices = genre.nonzero(as_tuple=False).squeeze().cpu().numpy()
            if g_indices.ndim == 0: g_indices = [g_indices.item()]
            genre_list = [idx_to_genre[idx] for idx in g_indices]
            genre_str = " ".join(genre_list).lower()
            
            # === 核心筛选条件: Audio 必须认为是 Happy (High Energy) ===
            if pred_audio == idx_happy:
                
                # Case A: Rap (Audio:Happy -> Full:Tense) - 纠错
                if len(candidates_rap) == 0:
                    if 'rap' in genre_str or 'hip' in genre_str:
                        if pred_full == idx_tense and true_label_idx == idx_tense:
                            print(f"✅ Found Case A (Rap Correction): Song {song_id} ({genre_str})")
                            candidates_rap.append({'id': song_id, 'mel': mel, 'genre': genre_str})
                            _copy_audio(song_id, raw_audio_dir, output_dir, "CaseA_Rap_IsTense")

                # Case B: Pop (Audio:Happy -> Full:Happy) - 确认
                if len(candidates_pop) == 0:
                    # 包含 Pop, Dance, Electronic
                    if 'pop' in genre_str or 'dance' in genre_str or 'electronic' in genre_str:
                        if pred_full == idx_happy and true_label_idx == idx_happy:
                            print(f"✅ Found Case B (Pop Confirmation): Song {song_id} ({genre_str})")
                            candidates_pop.append({'id': song_id, 'mel': mel, 'genre': genre_str})
                            _copy_audio(song_id, raw_audio_dir, output_dir, "CaseB_Pop_IsHappy")

    # 3. 绘图
    if candidates_rap and candidates_pop:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot Rap
        rap = candidates_rap[0]
        axes[0].imshow(rap['mel'].squeeze().numpy(), aspect='auto', origin='lower', cmap='magma')
        axes[0].set_title(f"Case A: {rap['genre'].title()}\nAudio Sees: 'Happy' (High Energy)\nGenre Corrects to: 'Tense' (Correct)", 
                          fontsize=11, fontweight='bold', bbox=dict(facecolor='#ffcccc', alpha=0.5, boxstyle='round'))
        axes[0].set_ylabel("Frequency")
        axes[0].set_xlabel("Time")

        # Plot Pop
        pop = candidates_pop[0]
        axes[1].imshow(pop['mel'].squeeze().numpy(), aspect='auto', origin='lower', cmap='magma')
        axes[1].set_title(f"Case B: {pop['genre'].title()}\nAudio Sees: 'Happy' (High Energy)\nGenre Confirms: 'Happy' (Correct)", 
                          fontsize=11, fontweight='bold', bbox=dict(facecolor='#ccffcc', alpha=0.5, boxstyle='round'))
        axes[1].set_xlabel("Time")
        axes[1].set_yticks([])

        plt.suptitle("The Spectrum of High Energy: How Genre Resolves Ambiguity", fontsize=16, y=1.02)
        plt.tight_layout()
        save_path = os.path.join(output_dir, "spectrum_contrast.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"📊 Contrast Plot saved to: {save_path}")
        plt.show()
    else:
        print("⚠️ Warning: Could not find a perfect Rap vs Pop pair in this pass.")

def _copy_audio(song_id, raw_dir, out_dir, prefix):
    for ext in ['.mp3', '.wav', '.au', '.flac', '.m4a']:
        src = os.path.join(raw_dir, f"{song_id}{ext}")
        if os.path.exists(src):
            dst = os.path.join(out_dir, f"{prefix}_{song_id}{ext}")
            shutil.copy2(src, dst)
            print(f"   -> Audio copied: {dst}")
            return

# ═══════════＝  6. Main Execution Block ＝═══════════
if __name__ == '__main__':
    print("▶ Loading Data...")
    try:
        df_labels = pd.read_csv(LABELS_CSV); df_genres = pd.read_csv(GENRES_CSV); df_audio_features = pd.read_csv(AUDIO_FEATURES_CSV)
    except FileNotFoundError as e: print(f"❌ Error: {e.filename} not found."); sys.exit()
    
    # Merge and Clean Data
    df = pd.merge(df_labels, df_genres, on="song_id"); df = pd.merge(df, df_audio_features, on="song_id")
    df['song_id'] = pd.to_numeric(df['song_id'], errors='coerce'); df.dropna(subset=['song_id'], inplace=True)
    df['song_id'] = df['song_id'].astype(int).astype(str)
    df.dropna(subset=['normalized_genres'], inplace=True); df['normalized_genres'] = df['normalized_genres'].astype(str)
    
    # Filter missing files
    all_missing = {sid for sid in df['song_id'] if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    if all_missing: df = df[~df['song_id'].isin(all_missing)]
    
    # Encoders
    feature_cols = [col for col in df_audio_features.columns if col != 'song_id']
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["Quadrant"])
    CLASS_NAMES = list(le.classes_)
    
    all_genres_list = sorted({g for g in df.normalized_genres.str.cat(sep=';').split(';') if g})
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    idx_to_genre = {i: g for g, i in genre_to_idx.items()}
    N_GENRES = len(all_genres_list)
    EMB_DIM = 1024 

    # Clean Labels for Display
    label_display_map = {}
    for name in CLASS_NAMES:
        if 'Q1' in name or 'Happy' in name: label_display_map[name] = 'Happy'
        elif 'Q2' in name or 'Tense' in name: label_display_map[name] = 'Tense'
        elif 'Q3' in name or 'Sad' in name: label_display_map[name] = 'Sad'
        elif 'Q4' in name or 'Relaxed' in name: label_display_map[name] = 'Relaxed'
        else: label_display_map[name] = str(name).encode('ascii', 'ignore').decode('ascii')

    # Load Model
    print(f"▶ Loading Model Weights: {BEST_MODEL_PATH}")
    if not os.path.exists(BEST_MODEL_PATH):
        print("❌ Error: .pth model file not found."); sys.exit()
        
    model = QuadFusionNet(EMB_DIM, N_GENRES, len(feature_cols), len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH))

    # Prepare Analysis Dataset (using test split logic for consistency)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    _, analysis_indices = next(splitter.split(df, df["label_idx"]))
    analysis_dataset = QuadFusionDataset(df, analysis_indices, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, N_GENRES, genre_to_idx)

    # Run Analysis
    visualize_contrast_pair(
        model=model,
        dataset=analysis_dataset,
        class_names=CLASS_NAMES,
        label_map=label_display_map,
        idx_to_genre=idx_to_genre,
        raw_audio_dir=RAW_AUDIO_DIR,
        output_dir=ANALYSIS_OUTPUT_DIR,
        device=device
    )

    print("\n▶ Script finished successfully.")