# ─────────────────── Import Libraries ───────────────────
import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# ═══════════＝  1. Configurations and Paths ＝═══════════
# 【请修改】你的原始音频目录
RAW_AUDIO_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\audio_files" 

# 其他路径
MEL_FEAT_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\mel"
MUQ_EMB_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\muq"
AUDIO_FEATURES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\audio_features_normalized.csv"
GENRES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\processed_genres.csv"
LABELS_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\quadrants_from_dynamic.csv"
ANALYSIS_OUTPUT_DIR = "analysis_cases_output"

NUM_RUNS = 3
TEST_SIZE = 0.2
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 50
USE_EARLY_STOPPING = True
ES_PATIENCE = 8
BEST_MODEL_PATH = "best_model_quad_fusion-1.pth"

# ═══════════＝  2. Device Configuration ＝═══════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶ Using device: {device}")


# ═══════════＝  3. Model Definition ＝═══════════
class QuadFusionNet(nn.Module):
    def __init__(self, emb_dim, genre_dim, audio_feature_dim, n_classes):
        super().__init__()
        self.mel_cnn_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.emb_branch = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5)
        )
        self.genre_branch = nn.Sequential(
            nn.Linear(genre_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5)
        )
        self.audio_feature_branch = nn.Sequential(
            nn.BatchNorm1d(audio_feature_dim), nn.ReLU(), nn.Dropout(0.5)
        )
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

# ═══════════＝  4. Dataset and Collate Function ＝═══════════
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
        return mel_spec, embedding, genre_vector, audio_features, label, song_id

def pad_collate_fn(batch):
    # 解包 batch (含 song_ids)
    mels, embs, genres, audios, labels, song_ids = zip(*batch)
    
    # Pad Mel Spectrograms
    max_len = max(mel.shape[1] for mel in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len)
    for i, mel in enumerate(mels):
        length = mel.shape[1]
        padded_mels[i, :, :length] = mel
        
    # Collate others
    embs_collated = default_collate(embs)
    genres_collated = default_collate(genres)
    audios_collated = default_collate(audios)
    labels_collated = default_collate(labels)
    
    # 【修复重点】：必须返回 song_ids，即使它是 tuple 且训练时不用
    return padded_mels, embs_collated, genres_collated, audios_collated, labels_collated, song_ids

# ═══════════＝  5. Visualization & Audio Helper (SAFE MODE) ＝═══════════
def visualize_correction_cases(model, dataset, class_names, label_map, idx_to_genre, 
                               raw_audio_dir, output_dir, device, max_cases=5):
    print(f"\n🔍 [Case Study] Finding correction cases (Target: {max_cases})...")
    model.eval()
    found_cases = []
    
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    indices = list(range(len(dataset)))
    np.random.shuffle(indices) 

    with torch.no_grad():
        for i in indices:
            if len(found_cases) >= max_cases: break
            
            mel, emb, genre, audio, label_tensor, song_id = dataset[i]
            true_label_idx = label_tensor.item()
            
            mel_b = mel.unsqueeze(0).to(device)
            emb_b = emb.unsqueeze(0).to(device)
            genre_b = genre.unsqueeze(0).to(device)
            audio_b = audio.unsqueeze(0).to(device)

            # Audio-Only (Masked)
            genre_zeros = torch.zeros_like(genre_b).to(device)
            logits_audio = model(mel_b, emb_b, genre_zeros, audio_b)
            pred_audio_idx = logits_audio.argmax(1).item()

            # Full Model
            logits_full = model(mel_b, emb_b, genre_b, audio_b)
            pred_full_idx = logits_full.argmax(1).item()

            # Condition: Audio Wrong AND Full Correct
            if (pred_audio_idx != true_label_idx) and (pred_full_idx == true_label_idx):
                
                # Format Genre
                g_indices = genre.nonzero(as_tuple=False).squeeze().cpu().numpy()
                if g_indices.ndim == 0: g_indices = [g_indices.item()]
                genre_names = [idx_to_genre[idx] for idx in g_indices]
                genre_str = ", ".join(genre_names)

                # Format Labels (Clean Text)
                true_label_str = class_names[true_label_idx]
                wrong_label_str = class_names[pred_audio_idx]
                
                display_true = label_map.get(true_label_str, "Unknown")
                display_wrong = label_map.get(wrong_label_str, "Unknown")
                display_correct = display_true 

                print(f"✅ Case Found: {song_id} | Genre: {genre_str} | True: {display_true} | Audio saw: {display_wrong}")

                # Copy Audio
                for ext in ['.mp3', '.wav', '.au', '.flac', '.m4a']:
                    src_path = os.path.join(raw_audio_dir, f"{song_id}{ext}")
                    if os.path.exists(src_path):
                        # Safe filename
                        safe_genre = genre_names[0].replace(' ','_').replace('/','-') if genre_names else "Unknown"
                        dst_filename = f"Case{len(found_cases)+1}_{song_id}_{safe_genre}_{display_true}.mp3"
                        shutil.copy2(src_path, os.path.join(output_dir, dst_filename))
                        break

                found_cases.append({
                    'song_id': song_id, 'mel': mel.cpu(),
                    'genre_str': genre_str,
                    'display_true': display_true,
                    'display_wrong': display_wrong,
                    'display_correct': display_correct
                })

    if not found_cases:
        print("⚠️ No cases found.")
        return

    # Plotting
    num_cases = len(found_cases)
    fig, axes = plt.subplots(1, num_cases, figsize=(5 * num_cases, 5))
    if num_cases == 1: axes = [axes]
    
    fig.suptitle(f"Genre Integration Analysis: Correcting Audio Ambiguity", fontsize=16, y=1.05)

    for idx, case in enumerate(found_cases):
        ax = axes[idx]
        mel_spec = case['mel'].squeeze().numpy()
        im = ax.imshow(mel_spec, aspect='auto', origin='lower', cmap='magma')
        
        title_text = (f"Case #{idx+1} (Song {case['song_id']})\n"
                      f"Truth: {case['display_true']}\n"
                      f"Audio-Only: {case['display_wrong']}\n"
                      f"With Genre ({case['genre_str']}): {case['display_correct']}")
        
        ax.set_title(title_text, fontsize=11, fontweight='bold', pad=12, 
                     bbox=dict(facecolor='#f8f9fa', edgecolor='gray', boxstyle='round,pad=0.4', alpha=0.9))
        ax.set_ylabel("Frequency" if idx==0 else "")
        ax.set_xlabel("Time Frames")
        if idx > 0: ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correction_analysis.png"), bbox_inches='tight', dpi=150)
    print(f"📊 Plot saved to: {os.path.join(output_dir, 'correction_analysis.png')}")
    plt.show()

# ═══════════＝  6. Run Experiment (Fixed) ＝═══════════
def run_experiment(run_num, full_df, feature_cols, emb_dim, n_genres, genre_to_idx, class_names, class_weights):
    print(f"\n{'═'*25} STARTING RUN {run_num}/{NUM_RUNS} {'═'*25}")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=None)
    train_indices, test_indices = next(splitter.split(full_df, full_df["label_idx"]))
    train_val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=None)
    train_sub_indices, val_indices = next(train_val_splitter.split(full_df.iloc[train_indices], full_df.iloc[train_indices]["label_idx"]))
    final_train_indices, final_val_indices = train_indices[train_sub_indices], train_indices[val_indices]
    
    train_dataset = QuadFusionDataset(full_df, final_train_indices, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx)
    val_dataset = QuadFusionDataset(full_df, final_val_indices, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx)
    test_dataset = QuadFusionDataset(full_df, test_indices, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate_fn)

    model = QuadFusionNet(emb_dim, n_genres, len(feature_cols), len(class_names)).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float('inf')
    wait_for_es = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct = 0, 0
        for batch in train_loader:
            # 训练循环：忽略最后一个 song_id 变量
            mel, emb, genre, audio, labels, _ = batch 
            mel, emb, genre, audio, labels = mel.to(device), emb.to(device), genre.to(device), audio.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mel, emb, genre, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
        history["train_loss"].append(train_loss / len(train_dataset))
        history["train_acc"].append(train_correct / len(train_dataset))

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                mel, emb, genre, audio, labels, _ = batch 
                mel, emb, genre, audio, labels = mel.to(device), emb.to(device), genre.to(device), audio.to(device), labels.to(device)
                outputs = model(mel, emb, genre, audio)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = val_correct / len(val_dataset)
        scheduler.step(avg_val_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            wait_for_es = 0
        else:
            wait_for_es += 1
            if USE_EARLY_STOPPING and wait_for_es >= ES_PATIENCE: break
            
    # Evaluation
    model.load_state_dict(torch.load(BEST_MODEL_PATH)); model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            mel, emb, genre, audio, labels, _ = batch 
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            outputs = model(mel, emb, genre, audio)
            y_pred.extend(outputs.argmax(1).cpu().numpy()); y_true.extend(labels.cpu().numpy())
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Run {run_num} Macro-F1: {macro_f1:.4f}")
    return macro_f1, None, history

# ═══════════＝  7. Main Block ＝═══════════
if __name__ == '__main__':
    try:
        df_labels = pd.read_csv(LABELS_CSV); df_genres = pd.read_csv(GENRES_CSV); df_audio_features = pd.read_csv(AUDIO_FEATURES_CSV)
    except FileNotFoundError as e: print(f"❌ Error: File not found {e.filename}"); sys.exit()
    df = pd.merge(df_labels, df_genres, on="song_id"); df = pd.merge(df, df_audio_features, on="song_id")
    df['song_id'] = pd.to_numeric(df['song_id'], errors='coerce'); df.dropna(subset=['song_id'], inplace=True); df['song_id'] = df['song_id'].astype(int).astype(str)
    df.dropna(subset=['normalized_genres'], inplace=True); df['normalized_genres'] = df['normalized_genres'].astype(str)
    
    missing_mels = {sid for sid in df['song_id'] if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    missing_muqs = {sid for sid in df['song_id'] if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
    all_missing = missing_mels.union(missing_muqs)
    if all_missing: df = df[~df['song_id'].isin(all_missing)]
    if df.empty: sys.exit("No valid data.")
    
    feature_cols = [col for col in df_audio_features.columns if col != 'song_id']
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["Quadrant"])
    CLASS_NAMES = list(le.classes_)
    all_genres_list = sorted({g for g in df.normalized_genres.str.cat(sep=';').split(';') if g})
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    idx_to_genre = {i: g for g, i in genre_to_idx.items()}
    N_GENRES = len(all_genres_list)
    EMB_DIM = 1024 
    
    # Label Map for Clean Text
    label_display_map = {}
    for name in CLASS_NAMES:
        if 'Q1' in name or 'Happy' in name: label_display_map[name] = 'Happy'
        elif 'Q2' in name or 'Tense' in name: label_display_map[name] = 'Tense'
        elif 'Q3' in name or 'Sad' in name: label_display_map[name] = 'Sad'
        elif 'Q4' in name or 'Relaxed' in name: label_display_map[name] = 'Relaxed'
        else: label_display_map[name] = str(name).encode('ascii', 'ignore').decode('ascii')

    class_weights = compute_class_weight('balanced', classes=np.unique(df["label_idx"]), y=df["label_idx"].values)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    all_macro_f1s = []
    for i in range(1, NUM_RUNS + 1):
        macro_f1, _, _ = run_experiment(i, df, feature_cols, EMB_DIM, N_GENRES, genre_to_idx, CLASS_NAMES, class_weights)
        all_macro_f1s.append(macro_f1)

    print(f"\nFinal Macro F1: {np.mean(all_macro_f1s):.4f}")

    # Case Study Analysis
    print("\n" + "═"*20 + " CASE STUDY ANALYSIS " + "═"*20)
    best_model = QuadFusionNet(EMB_DIM, N_GENRES, len(feature_cols), len(CLASS_NAMES)).to(device)
    if os.path.exists(BEST_MODEL_PATH): best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    else: sys.exit("Best model not found.")

    analysis_splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    _, analysis_indices = next(analysis_splitter.split(df, df["label_idx"]))
    analysis_dataset = QuadFusionDataset(df, analysis_indices, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, N_GENRES, genre_to_idx)

    visualize_correction_cases(
        model=best_model,
        dataset=analysis_dataset,
        class_names=CLASS_NAMES,
        label_map=label_display_map,
        idx_to_genre=idx_to_genre,
        raw_audio_dir=RAW_AUDIO_DIR,
        output_dir=ANALYSIS_OUTPUT_DIR,
        device=device,
        max_cases=5
    )

    print("\n▶ Script finished.")