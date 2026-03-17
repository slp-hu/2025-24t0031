# ─────────────────── Import Libraries ───────────────────
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns # For better plotting
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ═══════════＝  1. Configurations and Paths ＝═══════════
MUQ_EMB_DIR = "G:\\13kmid30s_muq"
MEL_FEAT_DIR = "G:\\13kmid30smel"
DATASET_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\newdata\\merged_joyful_melon_with_new_features.csv"
NUM_RUNS = 1
TEST_SIZE = 0.2
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
USE_EARLY_STOPPING = True
ES_PATIENCE = 5 # 提早停止的耐心值
BEST_MODEL_PATH = "best_model_new_dataset.pth"

# ═══════════＝  2. Device Configuration ＝═══════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶ Using device: {device}")


# ═══════════＝  3. Model Definition ＝═══════════
class QuadFusionNet(nn.Module):
    # (模型定义部分无需修改)
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
            nn.Linear(fusion_input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    def forward(self, mel, emb, genre, audio_feats):
        mel = mel.unsqueeze(1); z_mel = self.mel_cnn_branch(mel); z_mel = z_mel.view(z_mel.size(0), -1)
        z_emb = self.emb_branch(emb)
        z_genre = self.genre_branch(genre)
        z_audio = self.audio_feature_branch(audio_feats)
        combined = torch.cat([z_mel, z_emb, z_genre, z_audio], dim=1); output = self.fusion_head(combined)
        return output

# ═══════════＝  4. Dataset and Collate Function (Corrected Logic) ＝═══════════
class NewDataset(Dataset):
    def __init__(self, dataframe, mel_dir, muq_dir, feature_cols, n_genres, genre_to_idx):
        self.df = dataframe.reset_index(drop=True)
        self.mel_dir = mel_dir; self.muq_dir = muq_dir; self.feature_cols = feature_cols
        self.n_genres = n_genres; self.genre_to_idx = genre_to_idx

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
        label = torch.tensor(row.label_idx, dtype=torch.long)
        return mel_spec, embedding, genre_vector, audio_features, label

def pad_collate_fn(batch):
    mels, embs, genres, audios, labels = zip(*batch)
    max_len_mel = max(mel.shape[1] for mel in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len_mel)
    for i, mel in enumerate(mels): padded_mels[i, :, :mel.shape[1]] = mel
    embs_collated = default_collate(embs)
    genres_collated = default_collate(genres)
    audios_collated = default_collate(audios)
    labels_collated = default_collate(labels)
    return padded_mels, embs_collated, genres_collated, audios_collated, labels_collated

# ═══════════＝  5. Experiment Execution Function ＝═══════════
def run_experiment(run_num, full_df, feature_cols, emb_dim, n_genres, genre_to_idx, class_names, class_weights):
    print(f"\n{'═'*25} STARTING RUN {run_num}/{NUM_RUNS} {'═'*25}")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    train_indices, test_indices = next(splitter.split(full_df, full_df["label_idx"]))
    train_df = full_df.iloc[train_indices]
    train_val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=42)
    train_sub_indices, val_indices = next(train_val_splitter.split(train_df, train_df["label_idx"]))
    
    train_dataset = NewDataset(train_df.iloc[train_sub_indices], MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx)
    val_dataset = NewDataset(train_df.iloc[val_indices], MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx)
    test_dataset = NewDataset(full_df.iloc[test_indices], MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate_fn, num_workers=0)

    print(f"Run {run_num} Data Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    model = QuadFusionNet(emb_dim, n_genres, len(feature_cols), len(class_names)).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, min_lr=1e-6)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float('inf'); wait_for_es = 0
    for epoch in range(1, EPOCHS + 1):
        # (训练循环无需修改)
        model.train(); train_loss, train_correct, train_samples = 0, 0, 0
        for mel, emb, genre, audio, labels in train_loader:
            if mel is None: continue
            mel, emb, genre, audio, labels = mel.to(device), emb.to(device), genre.to(device), audio.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(mel, emb, genre, audio); loss = criterion(outputs, labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += loss.item() * labels.size(0); train_correct += (outputs.argmax(1) == labels).sum().item(); train_samples += labels.size(0)
        history["train_loss"].append(train_loss / train_samples if train_samples > 0 else 0)
        history["train_acc"].append(train_correct / train_samples if train_samples > 0 else 0)
        model.eval(); val_loss, val_correct, val_samples = 0, 0, 0
        with torch.no_grad():
            for mel, emb, genre, audio, labels in val_loader:
                if mel is None: continue
                mel, emb, genre, audio, labels = mel.to(device), emb.to(device), genre.to(device), audio.to(device), labels.to(device)
                outputs = model(mel, emb, genre, audio); loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0); val_correct += (outputs.argmax(1) == labels).sum().item(); val_samples += labels.size(0)
        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0
        avg_val_acc = val_correct / val_samples if val_samples > 0 else 0
        scheduler.step(avg_val_loss)
        history["val_loss"].append(avg_val_loss); history["val_acc"].append(avg_val_acc)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {history['train_loss'][-1]:.4f}, Acc: {history['train_acc'][-1]:.4f} | Val Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        if avg_val_loss < best_val_loss and val_samples > 0:
            best_val_loss = avg_val_loss; torch.save(model.state_dict(), BEST_MODEL_PATH); wait_for_es = 0
        else:
            wait_for_es += 1
            if USE_EARLY_STOPPING and wait_for_es >= ES_PATIENCE: print(f"--- Early stopping triggered at epoch {epoch} ---"); break
    print(f"\nRun {run_num} training finished. Evaluating on the test set...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH)); model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for mel, emb, genre, audio, labels in test_loader:
            if mel is None: continue
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            outputs = model(mel, emb, genre, audio)
            y_pred.extend(outputs.argmax(1).cpu().numpy()); y_true.extend(labels.cpu().numpy())
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0)
    print(f"Run {run_num} Results -> Macro-F1: {macro_f1:.4f}")
    return macro_f1, class_f1, history, y_true, y_pred

# ═══════════＝  6. Main Execution Block ＝═══════════
if __name__ == '__main__':
    print("--- ▶ Starting Data Preparation ---")
    try: df = pd.read_csv(DATASET_CSV)
    except FileNotFoundError as e: print(f"❌ FATAL ERROR: CSV file not found -> {e.filename}. Please check your file paths."); sys.exit()
    df.dropna(subset=['id', 'final_label', 'major_genre_name_en'], inplace=True)
    df = df[df['id'] != 'id'].copy(); df['id'] = df['id'].astype(str)
    initial_count = len(df)
    missing_mels = {sid for sid in df['id'] if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    missing_muqs = {sid for sid in df['id'] if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
    all_missing = missing_mels.union(missing_muqs)
    if all_missing:
        print(f"▶ Info: Removing {len(all_missing)} records with missing feature files.")
        df = df[~df['id'].isin(all_missing)]
    if df.empty: print("\n❌ FATAL ERROR: No valid data entries remaining after checking for feature files."); sys.exit()
    print(f"▶ Data preparation complete. Using {len(df)} valid entries.")
    EMB_DIM = 1024
    feature_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 
                    'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    le_label = LabelEncoder(); df["label_idx"] = le_label.fit_transform(df["final_label"])
    CLASS_NAMES = list(le_label.classes_)
    all_genres_list = sorted(df["major_genre_name_en"].unique())
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    N_GENRES = len(all_genres_list)
    class_weights = compute_class_weight('balanced', classes=np.unique(df["label_idx"]), y=df["label_idx"].values)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    all_macro_f1s, all_class_f1s, all_histories, all_y_trues, all_y_preds = [], [], [], [], []
    start_time = time.time()
    for i in range(1, NUM_RUNS + 1):
        macro_f1, class_f1, history, y_true, y_pred = run_experiment(
            run_num=i, full_df=df, feature_cols=feature_cols, emb_dim=EMB_DIM, n_genres=N_GENRES, 
            genre_to_idx=genre_to_idx, class_names=CLASS_NAMES, class_weights=class_weights
        )
        all_macro_f1s.append(macro_f1); all_class_f1s.append(class_f1); all_histories.append(history)
        all_y_trues.append(y_true); all_y_preds.append(y_pred)
    
    end_time = time.time(); print(f"\nTotal execution time for {NUM_RUNS} runs: {end_time - start_time:.2f} seconds.")

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 【NEW VISUALIZATION BLOCK】 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    print(f"\n\n{'═'*20} FINAL RESULTS (FROM BEST RUN) {'═'*20}")
    
    if not all_macro_f1s:
        print("No runs were completed successfully. Exiting.")
        sys.exit()

    best_run_index = np.argmax(all_macro_f1s)
    best_macro_f1 = all_macro_f1s[best_run_index]
    best_class_f1 = all_class_f1s[best_run_index]
    best_history = all_histories[best_run_index]
    best_y_true = all_y_trues[best_run_index]
    best_y_pred = all_y_preds[best_run_index]

    print(f"Best Run: #{best_run_index + 1}")
    print(f"Overall Macro F1-Score: {best_macro_f1:.4f}")
    print("-" * 50)
    print("Per-Class F1-Scores:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  - {class_name:<15}: {best_class_f1[i]:.4f}")

    print("\n▶ Generating visualizations...")
    
    # --- 1. Plotting Loss and Accuracy Curves ---
    fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig_curves.suptitle(f"Training Curves from Best Run (Run #{best_run_index + 1})", fontsize=16)
    
    epochs_ran = len(best_history["train_loss"])
    ax1.plot(range(1, epochs_ran + 1), best_history["train_loss"], 'o-', label="Train Loss")
    ax1.plot(range(1, epochs_ran + 1), best_history["val_loss"], 'o-', label="Validation Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(1, epochs_ran + 1), best_history["train_acc"], 'o-', label="Train Accuracy")
    ax2.plot(range(1, epochs_ran + 1), best_history["val_acc"], 'o-', label="Validation Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    # --- 2. Plotting Confusion Matrix ---
    cm = confusion_matrix(best_y_true, best_y_pred, labels=le_label.transform(CLASS_NAMES))
    fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax_cm)
    ax_cm.set_title(f"Confusion Matrix from Best Run (Run #{best_run_index + 1})")
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")

    # --- 3. Plotting F1 Scores ---
    fig_f1, ax_f1 = plt.subplots(figsize=(12, 8))
    f1_scores_df = pd.DataFrame({'Class': CLASS_NAMES, 'F1-Score': best_class_f1})
    f1_scores_df = f1_scores_df.sort_values('F1-Score', ascending=False)
    
    sns.barplot(x='F1-Score', y='Class', data=f1_scores_df, ax=ax_f1, palette='viridis')
    ax_f1.set_title(f"Per-Class F1-Scores (Macro-F1: {best_macro_f1:.4f})")
    ax_f1.set_xlim(0, 1.0)
    # Add F1 scores as text on the bars
    for index, value in enumerate(f1_scores_df['F1-Score']):
        ax_f1.text(value + 0.01, index, f'{value:.3f}', va='center')

    plt.tight_layout()
    plt.show()

    print("\n▶ Script finished.")
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 【NEW VISUALIZATION BLOCK】 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲