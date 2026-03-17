# ─────────────────── Import Libraries ───────────────────
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight # <--- 导入新库
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import time

# ═══════════＝  1. Configurations and Paths ＝═══════════

# --- Data File Paths ---
MUQ_EMB_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\muq"
AUDIO_FEATURES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\audio_features_normalized.csv"
GENRES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\processed_genres.csv"
LABELS_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\quadrants_from_dynamic.csv"

# --- Model and Training Hyperparameters ---
NUM_RUNS = 3
EMB_DIM = 1024
TEST_SIZE = 0.2
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 50
USE_EARLY_STOPPING = True
ES_PATIENCE = 8
BEST_MODEL_PATH = "best_model_muq.pth"

# ═══════════＝  2. Device Configuration ＝═══════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶ Using device: {device}")


# ═══════════＝  3. Model Definition ＝═══════════
class TripleInputFusionNet(nn.Module):
    def __init__(self, emb_dim, genre_dim, audio_feature_dim, n_classes):
        super().__init__()
        
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 【DROPOUT VALUES ADJUSTED】 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # NOTE: Original Dropout values (0.6, 0.7, 0.8) were very high and likely
        #       hindered learning. These have been adjusted to more standard values.
        
        # Branch 1: MuQ embedding
        self.emb_branch = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)  # Adjusted from 0.6
        )
        # Branch 2: Genre
        self.genre_branch = nn.Sequential(
            nn.Linear(genre_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5)  # Adjusted from 0.7
        )
        # Branch 3: Audio Features
        self.audio_feature_branch = nn.Sequential(
            nn.BatchNorm1d(audio_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5)  # Adjusted from 0.8
        )

        # Fusion Head (two fully-connected layers)
        self.fusion_head = nn.Sequential(
            nn.Linear(128 + 32 + audio_feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4), # Adjusted from 0.3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3), # Adjusted from 0.2
            nn.Linear(32, n_classes)
        )
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 【END OF ADJUSTMENT】 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    def forward(self, emb, genre, audio_feats):
        z_emb = self.emb_branch(emb)
        z_genre = self.genre_branch(genre)
        z_audio = self.audio_feature_branch(audio_feats)
        combined = torch.cat([z_emb, z_genre, z_audio], dim=1)
        output = self.fusion_head(combined)
        return output

# ═══════════＝  4. Dataset Definition ＝═══════════
class MusicEmotionDataset(Dataset):
    def __init__(self, dataframe, indices, muq_dir, feature_cols, n_genres, genre_to_idx):
        self.df = dataframe.iloc[indices]
        self.muq_dir = muq_dir
        self.feature_cols = feature_cols
        self.n_genres = n_genres
        self.genre_to_idx = genre_to_idx
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        song_id = row.song_id
        emb_path = os.path.join(self.muq_dir, f"{song_id}.npy")
        embedding = torch.from_numpy(np.load(emb_path).astype(np.float32))
        genre_vector = torch.zeros(self.n_genres, dtype=torch.float32)
        genres = row.normalized_genres.split(';')
        weights = 1 / np.arange(1, len(genres) + 1); weights /= weights.sum()
        for k, g in enumerate(genres):
            if g in self.genre_to_idx: genre_vector[self.genre_to_idx[g]] = weights[k]
        audio_features = torch.tensor(row[self.feature_cols].values.astype(np.float32), dtype=torch.float32)
        label = torch.tensor(row.label_idx, dtype=torch.long)
        return embedding, genre_vector, audio_features, label

# ═══════════＝  5. Experiment Execution Function ＝═══════════
# MODIFIED: a new 'class_weights' parameter is passed
def run_experiment(run_num, full_df, feature_cols, n_genres, genre_to_idx, class_names, class_weights):
    print(f"\n{'═'*25} STARTING RUN {run_num}/{NUM_RUNS} {'═'*25}")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=None)
    train_indices, test_indices = next(splitter.split(full_df, full_df["label_idx"]))
    
    train_val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=None)
    train_sub_indices, val_indices = next(train_val_splitter.split(full_df.iloc[train_indices], full_df.iloc[train_indices]["label_idx"]))
    
    final_train_indices, final_val_indices = train_indices[train_sub_indices], train_indices[val_indices]

    train_dataset = MusicEmotionDataset(full_df, final_train_indices, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx)
    val_dataset = MusicEmotionDataset(full_df, final_val_indices, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx)
    test_dataset = MusicEmotionDataset(full_df, test_indices, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    print(f"Run {run_num} Data Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    n_classes = len(class_names)
    model = TripleInputFusionNet(EMB_DIM, n_genres, len(feature_cols), n_classes).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # MODIFIED: Pass the calculated weights to the loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    scheduler = OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.2)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float('inf')
    wait_for_es = 0

    # ... [Training loop is the same] ...
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct = 0, 0
        for emb, genre, audio, labels in train_loader:
            emb, genre, audio, labels = emb.to(device), genre.to(device), audio.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(emb, genre, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
        history["train_loss"].append(train_loss / len(train_dataset))
        history["train_acc"].append(train_correct / len(train_dataset))
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for emb, genre, audio, labels in val_loader:
                emb, genre, audio, labels = emb.to(device), genre.to(device), audio.to(device), labels.to(device)
                outputs = model(emb, genre, audio)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        avg_val_loss = val_loss / len(val_dataset)
        avg_val_acc = val_correct / len(val_dataset)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(avg_val_acc)
        if (epoch) % 10 == 0: print(f"Epoch {epoch:02d}/{EPOCHS} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            wait_for_es = 0
        else:
            wait_for_es += 1
            if USE_EARLY_STOPPING and wait_for_es >= ES_PATIENCE:
                print(f"--- Early stopping triggered at epoch {epoch} ---"); break
    
    print(f"Run {run_num} training finished. Evaluating on the test set...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for emb, genre, audio, labels in test_loader:
            emb, genre, audio = emb.to(device), genre.to(device), audio.to(device)
            outputs = model(emb, genre, audio)
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0)
    print(f"Run {run_num} Results -> Macro-F1: {macro_f1:.4f}")
    return macro_f1, class_f1, history

# ═══════════＝  6. Main Execution Block ＝═══════════
if __name__ == '__main__':
    try:
        df_labels = pd.read_csv(LABELS_CSV)
        df_genres = pd.read_csv(GENRES_CSV)
        df_audio_features = pd.read_csv(AUDIO_FEATURES_CSV)
    except FileNotFoundError as e:
        print(f"❌ Error: File not found {e.filename}. Please check the file paths."); sys.exit()

    df = pd.merge(df_labels, df_genres, on="song_id")
    df = pd.merge(df, df_audio_features, on="song_id")
    
    # Standardize song_id to prevent mismatches
    df['song_id'] = pd.to_numeric(df['song_id'], errors='coerce')
    df.dropna(subset=['song_id'], inplace=True)
    df['song_id'] = df['song_id'].astype(int).astype(str)

    df.dropna(subset=['normalized_genres'], inplace=True)
    df['normalized_genres'] = df['normalized_genres'].astype(str)

    missing_embs = [sid for sid in df['song_id'] if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))]
    if missing_embs: df = df[~df['song_id'].isin(missing_embs)]
    if df.empty: print("\n❌ FATAL ERROR: No valid data entries remaining."); sys.exit()

    feature_cols = [col for col in df_audio_features.columns if col != 'song_id']
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["Quadrant"])
    CLASS_NAMES = list(le.classes_)
    
    all_genres_list = sorted({g for genres_str in df.normalized_genres for g in genres_str.split(';')})
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    N_GENRES = len(all_genres_list)

    print(f"\n▶ Data loading complete. Found {len(df)} valid entries.")
    
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 【NEW CODE HERE】 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # Calculate class weights for the entire dataset
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df["label_idx"]),
        y=df["label_idx"].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    print("▶ Calculated Class Weights to handle imbalance:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  - {name}: {class_weights[i]:.2f}")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 【END OF NEW CODE】 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    all_macro_f1s, all_class_f1s, all_histories = [], [], []
    start_time = time.time()
    for i in range(1, NUM_RUNS + 1):
        # Pass the calculated weights to the experiment function
        macro_f1, class_f1, history = run_experiment(
            run_num=i, full_df=df, feature_cols=feature_cols,
            n_genres=N_GENRES, genre_to_idx=genre_to_idx, class_names=CLASS_NAMES,
            class_weights=class_weights
        )
        all_macro_f1s.append(macro_f1)
        all_class_f1s.append(class_f1)
        all_histories.append(history)
    
    end_time = time.time()
    print(f"\nTotal execution time for {NUM_RUNS} runs: {end_time - start_time:.2f} seconds.")

    # ... [Reporting and Visualization code remains the same] ...
    print(f"\n\n{'═'*20} FINAL RESULTS (AVERAGED OVER {NUM_RUNS} RUNS) {'═'*20}")
    mean_macro_f1 = np.mean(all_macro_f1s)
    std_macro_f1 = np.std(all_macro_f1s)
    best_macro_f1 = np.max(all_macro_f1s)
    print(f"Overall Macro F1-Score: {mean_macro_f1:.4f} ± {std_macro_f1:.4f} (best-run: {best_macro_f1:.4f})")
    print("-" * 70)
    print("Per-Class F1-Scores:")
    class_f1_matrix = np.array(all_class_f1s)
    mean_class_f1s = np.mean(class_f1_matrix, axis=0)
    std_class_f1s = np.std(class_f1_matrix, axis=0)
    best_class_f1s = np.max(class_f1_matrix, axis=0)
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  - {class_name:<6}: {mean_class_f1s[i]:.4f} ± {std_class_f1s[i]:.4f} (best-run: {best_class_f1s[i]:.4f})")
    
    print("\n▶ Generating visualizations...")
    best_run_index = np.argmax(all_macro_f1s)
    best_history = all_histories[best_run_index]
    fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig_curves.suptitle(f"Training Curves from Best Performing Run (Run #{best_run_index + 1})", fontsize=16)
    ax1.plot(best_history["train_loss"], label="Train Loss"); ax1.plot(best_history["val_loss"], label="Validation Loss")
    ax1.set_title("Loss Curves"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)
    ax2.plot(best_history["train_acc"], label="Train Accuracy"); ax2.plot(best_history["val_acc"], label="Validation Accuracy")
    ax2.set_title("Accuracy Curves"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
    fig_metrics, ax_metrics = plt.subplots(figsize=(8, 6))
    fig_metrics.suptitle(f"F1-Score Statistics Over {NUM_RUNS} Runs", fontsize=16)
    metrics_text = f"Overall Macro F1-Score:\n{mean_macro_f1:.4f} ± {std_macro_f1:.4f} (best: {best_macro_f1:.4f})\n\n"
    metrics_text += "-----------------------------------\n\n"
    metrics_text += "Per-Class F1-Scores:\n"
    for i, class_name in enumerate(CLASS_NAMES):
        metrics_text += f"  - {class_name:<6}: {mean_class_f1s[i]:.4f} ± {std_class_f1s[i]:.4f} (best: {best_class_f1s[i]:.4f})\n"
    ax_metrics.text(0.5, 0.5, metrics_text, ha='center', va='center', fontfamily='monospace', fontsize=12,
                    bbox=dict(boxstyle="round,pad=1", fc='lightcyan', alpha=0.8))
    ax_metrics.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("\n▶ Script finished.")