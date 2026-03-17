# ─────────────────── Import Libraries ───────────────────
import os
import sys
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
MEL_FEAT_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\mel"
MUQ_EMB_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\muq"
AUDIO_FEATURES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\audio_features_normalized.csv"
GENRES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\processed_genres.csv"
LABELS_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\quadrants_from_dynamic.csv"

# 注意：这里最好改一下保存路径，以免覆盖掉你之前最好的全模型
BEST_MODEL_PATH = "best_model_no_genre_ablation.pth" 

NUM_RUNS = 3
TEST_SIZE = 0.2
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 50
USE_EARLY_STOPPING = True
ES_PATIENCE = 8

# ═══════════＝  2. Device Configuration ＝═══════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶ Using device: {device}")

# ═══════════＝  3. Model Definition (NO GENRE - ABLATION) ＝═══════════
class NoGenreFusionNet(nn.Module):
    # 【修改1】移除了 genre_dim 参数，因为不需要了
    def __init__(self, emb_dim, audio_feature_dim, n_classes):
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
        
        # 【修改2】Branch 3: Genre 被移除了 (Deleted Genre Branch)
        
        # Branch 4: Audio Features
        self.audio_feature_branch = nn.Sequential(
            nn.BatchNorm1d(audio_feature_dim), nn.ReLU(), nn.Dropout(0.5)
        )

        # Updated Fusion Head
        cnn_output_dim = 256
        emb_output_dim = 128
        # genre_output_dim = 32  <-- 移除了
        
        # 【修改3】融合层的输入维度不再包含 Genre
        fusion_input_dim = cnn_output_dim + emb_output_dim + audio_feature_dim
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    # 【修改4】forward 不再接收 genre 参数
    def forward(self, mel, emb, audio_feats):
        # Process each branch
        mel = mel.unsqueeze(1)
        z_mel = self.mel_cnn_branch(mel)
        z_mel = z_mel.view(z_mel.size(0), -1)
        
        z_emb = self.emb_branch(emb)
        # z_genre = self.genre_branch(genre) <-- 移除了
        z_audio = self.audio_feature_branch(audio_feats)
        
        # 【修改5】Concatenate 时只拼接 Mel, Emb, Audio
        combined = torch.cat([z_mel, z_emb, z_audio], dim=1)
        output = self.fusion_head(combined)
        return output

# ═══════════＝  4. Dataset and Collate Function ＝═══════════
# Dataset 保持不变，我们依然加载流派数据，只是在训练时不喂给模型
# 这样可以最大程度减少代码改动带来的风险
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
        
        # 1. Mel
        mel_path = os.path.join(self.mel_dir, f"{song_id}.npy")
        mel_spec = torch.from_numpy(np.load(mel_path).astype(np.float32))
        
        # 2. MuQ
        emb_path = os.path.join(self.muq_dir, f"{song_id}.npy")
        embedding = torch.from_numpy(np.load(emb_path).astype(np.float32))

        # 3. Genre (依然加载，但后面会忽略)
        genre_vector = torch.zeros(self.n_genres, dtype=torch.float32)
        genres = row.normalized_genres.split(';')
        weights = 1 / np.arange(1, len(genres) + 1); weights /= weights.sum()
        for k, g in enumerate(genres):
            if g in self.genre_to_idx: genre_vector[self.genre_to_idx[g]] = weights[k]
        
        # 4. Audio
        audio_features = torch.tensor(row[self.feature_cols].values.astype(np.float32), dtype=torch.float32)
        
        # 5. Label
        label = torch.tensor(row.label_idx, dtype=torch.long)
        
        return mel_spec, embedding, genre_vector, audio_features, label

def pad_collate_fn(batch):
    mels, embs, genres, audios, labels = zip(*batch)
    max_len = max(mel.shape[1] for mel in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len)
    for i, mel in enumerate(mels):
        length = mel.shape[1]
        padded_mels[i, :, :length] = mel
        
    embs_collated = default_collate(embs)
    genres_collated = default_collate(genres)
    audios_collated = default_collate(audios)
    labels_collated = default_collate(labels)
    
    return padded_mels, embs_collated, genres_collated, audios_collated, labels_collated

# ═══════════＝  5. Experiment Execution Function ＝═══════════
def run_experiment(run_num, full_df, feature_cols, emb_dim, n_genres, genre_to_idx, class_names, class_weights):
    print(f"\n{'═'*15} STARTING ABLATION RUN (NO GENRE) {run_num}/{NUM_RUNS} {'═'*15}")
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
    
    # 【修改6】实例化 NoGenreFusionNet (不需要传 n_genres 了)
    model = NoGenreFusionNet(emb_dim, len(feature_cols), len(class_names)).to(device)
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float('inf')
    wait_for_es = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct = 0, 0
        for mel, emb, genre, audio, labels in train_loader:
            # 【修改7】训练循环中，虽然 dataset 返回了 genre，但我们不把它 .to(device) 也不传给模型
            mel, emb, audio, labels = mel.to(device), emb.to(device), audio.to(device), labels.to(device)
            
            optimizer.zero_grad()
            # 【修改8】Forward 只传3个参数
            outputs = model(mel, emb, audio)
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
            for mel, emb, genre, audio, labels in val_loader:
                # 【修改9】验证循环同理，忽略 genre
                mel, emb, audio, labels = mel.to(device), emb.to(device), audio.to(device), labels.to(device)
                outputs = model(mel, emb, audio)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = val_correct / len(val_dataset)
        scheduler.step(avg_val_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {history['train_loss'][-1]:.4f}, Acc: {history['train_acc'][-1]:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            wait_for_es = 0
        else:
            wait_for_es += 1
            if USE_EARLY_STOPPING and wait_for_es >= ES_PATIENCE:
                print(f"--- Early stopping triggered at epoch {epoch} ---"); break
    
    print(f"Run {run_num} training finished. Evaluating on the test set...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH)); model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for mel, emb, genre, audio, labels in test_loader:
            # 【修改10】测试循环同理，忽略 genre
            mel, emb, audio = mel.to(device), emb.to(device), audio.to(device)
            outputs = model(mel, emb, audio)
            y_pred.extend(outputs.argmax(1).cpu().numpy()); y_true.extend(labels.cpu().numpy())
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0)
    print(f"Run {run_num} [NO GENRE] Results -> Macro-F1: {macro_f1:.4f}")
    return macro_f1, class_f1, history

# ═══════════＝  6. Main Execution Block ＝═══════════
if __name__ == '__main__':
    try:
        df_labels = pd.read_csv(LABELS_CSV); df_genres = pd.read_csv(GENRES_CSV); df_audio_features = pd.read_csv(AUDIO_FEATURES_CSV)
    except FileNotFoundError as e: print(f"❌ Error: File not found {e.filename}. Please check the file paths."); sys.exit()
    df = pd.merge(df_labels, df_genres, on="song_id"); df = pd.merge(df, df_audio_features, on="song_id")
    df['song_id'] = pd.to_numeric(df['song_id'], errors='coerce'); df.dropna(subset=['song_id'], inplace=True); df['song_id'] = df['song_id'].astype(int).astype(str)
    df.dropna(subset=['normalized_genres'], inplace=True); df['normalized_genres'] = df['normalized_genres'].astype(str)
    
    missing_mels = {sid for sid in df['song_id'] if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    missing_muqs = {sid for sid in df['song_id'] if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
    all_missing = missing_mels.union(missing_muqs)
    if all_missing:
        print(f"⚠️ Warning: Removing {len(all_missing)} records with missing feature files.")
        df = df[~df['song_id'].isin(all_missing)]
    if df.empty: print("\n❌ FATAL ERROR: No valid data entries remaining."); sys.exit()
    
    feature_cols = [col for col in df_audio_features.columns if col != 'song_id']
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["Quadrant"])
    CLASS_NAMES = list(le.classes_)
    all_genres_list = sorted({g for g in df.normalized_genres.str.cat(sep=';').split(';') if g})
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    N_GENRES = len(all_genres_list)
    EMB_DIM = 1024
    print(f"\n▶ Data loading complete. Found {len(df)} valid entries.")
    
    class_weights = compute_class_weight('balanced', classes=np.unique(df["label_idx"]), y=df["label_idx"].values)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    print("\n⚠️ STARTING ABLATION STUDY: NO GENRE BRANCH ⚠️")
    
    all_macro_f1s, all_class_f1s, all_histories = [], [], []
    start_time = time.time()
    for i in range(1, NUM_RUNS + 1):
        macro_f1, class_f1, history = run_experiment(
            run_num=i, full_df=df, feature_cols=feature_cols, emb_dim=EMB_DIM,
            n_genres=N_GENRES, genre_to_idx=genre_to_idx, 
            class_names=CLASS_NAMES, class_weights=class_weights
        )
        all_macro_f1s.append(macro_f1); all_class_f1s.append(class_f1); all_histories.append(history)
    
    end_time = time.time()
    print(f"\nTotal execution time for {NUM_RUNS} runs: {end_time - start_time:.2f} seconds.")

    print(f"\n\n{'═'*20} FINAL ABLATION RESULTS (NO GENRE) {'═'*20}")
    mean_macro_f1 = np.mean(all_macro_f1s); std_macro_f1 = np.std(all_macro_f1s); best_macro_f1 = np.max(all_macro_f1s)
    print(f"Overall Macro F1-Score: {mean_macro_f1:.4f} ± {std_macro_f1:.4f} (best-run: {best_macro_f1:.4f})")
    print("-" * 70); print("Per-Class F1-Scores:")
    class_f1_matrix = np.array(all_class_f1s); mean_class_f1s = np.mean(class_f1_matrix, axis=0); std_class_f1s = np.std(class_f1_matrix, axis=0); best_class_f1s = np.max(class_f1_matrix, axis=0)
    for i, class_name in enumerate(CLASS_NAMES): print(f"  - {class_name:<6}: {mean_class_f1s[i]:.4f} ± {std_class_f1s[i]:.4f} (best-run: {best_class_f1s[i]:.4f})")
    
    print("\n▶ Generating visualizations...")
    best_run_index = np.argmax(all_macro_f1s)
    best_history = all_histories[best_run_index]
    
    fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig_curves.suptitle(f"Training Curves (NO GENRE) - Best Run #{best_run_index + 1}", fontsize=16)
    ax1.plot(best_history["train_loss"], label="Train Loss"); ax1.plot(best_history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss Curves"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)
    ax2.plot(best_history["train_acc"], label="Train Accuracy"); ax2.plot(best_history["val_acc"], label="Validation Accuracy")
    ax2.set_title("Accuracy Curves"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
    fig_metrics, ax_metrics = plt.subplots(figsize=(8, 6))
    fig_metrics.suptitle(f"F1-Score Statistics (NO GENRE) Over {NUM_RUNS} Runs", fontsize=16)
    metrics_text = f"Overall Macro F1-Score:\n{mean_macro_f1:.4f} ± {std_macro_f1:.4f} (best: {best_macro_f1:.4f})\n\n"
    metrics_text += "-----------------------------------\n\n"; metrics_text += "Per-Class F1-Scores:\n"
    for i, class_name in enumerate(CLASS_NAMES): metrics_text += f"  - {class_name:<6}: {mean_class_f1s[i]:.4f} ± {std_class_f1s[i]:.4f} (best: {best_class_f1s[i]:.4f})\n"
    ax_metrics.text(0.5, 0.5, metrics_text, ha='center', va='center', fontfamily='monospace', fontsize=12, bbox=dict(boxstyle="round,pad=1", fc='mistyrose', alpha=0.8))
    ax_metrics.axis('off'); plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
    print("\n▶ Ablation study finished.")