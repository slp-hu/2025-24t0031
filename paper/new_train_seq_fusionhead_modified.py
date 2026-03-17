# ─────────────────── Import Libraries ───────────────────
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

# ═══════════＝  1. Configurations and Paths ＝═══════════
MUQ_EMB_DIR = "G:\\13kmid30s_muq"
MEL_FEAT_DIR = "G:\\13kmid30smel"
DATASET_CSV = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon.csv"

ALL_EMOTIONS = [
    'Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 'Quiet',
    'Happiness', 'Loneliness', 'Touching', 'Missing', 'Fresh', 'Relaxation'
]
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(ALL_EMOTIONS)}
IDX_TO_EMOTION = {i: emo for i, emo in enumerate(ALL_EMOTIONS)}
N_EMOTIONS = len(ALL_EMOTIONS)

NUM_RUNS = 1
TEST_SIZE = 0.2
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-4 # Best LR from previous experiments
EPOCHS = 50
USE_EARLY_STOPPING = True
ES_PATIENCE = 7
BEST_MODEL_PATH = "best_ranking_model_attention.pth" # New model name for this experiment

# ═══════════＝  2. Device Configuration ＝═══════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"> Using device: {device}")


# ═══════════＝  3. ListNet Loss Function ＝═══════════
class ListNetLoss(nn.Module):
    """
    A robust, vectorized implementation of ListNet loss using the Plackett-Luce model.
    """
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, y_pred_scores, y_true_seqs):
        total_loss = 0.0
        batch_size = y_pred_scores.shape[0]
        
        for i in range(batch_size):
            scores = y_pred_scores[i]
            true_seq = y_true_seqs[i]
            
            valid_seq = true_seq[true_seq != -1]
            if len(valid_seq) == 0:
                continue

            ground_truth_scores = scores[valid_seq]

            flipped_scores = torch.flip(ground_truth_scores, dims=[0])
            log_denominators = torch.logcumsumexp(flipped_scores, dim=0)
            log_denominators = torch.flip(log_denominators, dims=[0])

            log_likelihood = torch.sum(ground_truth_scores - log_denominators)
            
            total_loss -= log_likelihood

        return total_loss / batch_size

# ═══════════＝ 4. Ranking Evaluation Metric (nDCG) ＝═══════════
def ndcg_at_k(y_scores, y_true_lists, k=3):
    batch_ndcg = []
    for scores, true_list in zip(y_scores, y_true_lists):
        if not true_list:
            continue

        relevance = torch.zeros_like(scores)
        true_indices = [EMOTION_TO_IDX[emo] for emo in true_list if emo in EMOTION_TO_IDX]
        if not true_indices: continue
        relevance[true_indices] = 1.0

        _, top_k_indices = torch.topk(scores, k=k)

        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            dcg += relevance[idx] / torch.log2(torch.tensor(i + 2.0, device=scores.device))

        idcg = 0.0
        num_true = min(k, len(true_indices))
        for i in range(num_true):
             idcg += 1.0 / torch.log2(torch.tensor(i + 2.0, device=scores.device))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        batch_ndcg.append(ndcg)
        
    return torch.mean(torch.tensor(batch_ndcg)) if batch_ndcg else torch.tensor(0.0)

# ═══════════＝  5. 【UPGRADED】 Model Definition with Attention ＝═══════════
class QuadFusionNet(nn.Module):
    def __init__(self, emb_dim, genre_dim, audio_feature_dim, n_classes):
        super().__init__()
        
        # --- 1. Feature Extraction Branches (Best config from Run #4) ---
        self.mel_cnn_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.emb_branch = nn.Sequential(nn.Linear(emb_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5))
        self.genre_branch = nn.Sequential(nn.Linear(genre_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5))
        self.audio_feature_branch = nn.Sequential(nn.BatchNorm1d(audio_feature_dim), nn.ReLU(), nn.Dropout(0.5))
        
        # --- 2. 【NEW】 Feature Projection Layers ---
        # Project all branch outputs to a common dimension for weighted fusion
        attention_dim = 128
        self.project_mel = nn.Linear(256, attention_dim)
        self.project_emb = nn.Linear(128, attention_dim)
        self.project_genre = nn.Linear(32, attention_dim)
        self.project_audio = nn.Linear(audio_feature_dim, attention_dim)
        
        # --- 3. 【NEW】 Attention Network ---
        # Takes all concatenated original features as input
        fusion_input_dim = 256 + 128 + 32 + audio_feature_dim
        self.attention_net = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4), # Outputs 4 weights, one for each branch
            nn.Softmax(dim=1)  # Ensures the 4 weights sum to 1
        )
        
        # --- 4. 【MODIFIED】 Fusion Head ---
        # Its input dimension is now the common attention_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(attention_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, mel, emb, genre, audio_feats):
        # --- 1. Extract original features ---
        mel = mel.unsqueeze(1); z_mel = self.mel_cnn_branch(mel); z_mel = z_mel.view(z_mel.size(0), -1)
        z_emb = self.emb_branch(emb)
        z_genre = self.genre_branch(genre)
        z_audio = self.audio_feature_branch(audio_feats)
        
        # --- 2. 【NEW】 Calculate attention weights ---
        # Concatenate original features and feed to the attention network
        combined_original = torch.cat([z_mel, z_emb, z_genre, z_audio], dim=1)
        attention_weights = self.attention_net(combined_original) # Shape: (batch_size, 4)
        
        # --- 3. 【NEW】 Project features and apply weighted fusion ---
        # (a) Project each branch's feature to the common dimension
        p_mel = self.project_mel(z_mel)
        p_emb = self.project_emb(z_emb)
        p_genre = self.project_genre(z_genre)
        p_audio = self.project_audio(z_audio)
        
        # (b) Apply weights and sum
        # We need to unsqueeze the weights for broadcast multiplication
        # attention_weights[:, 0] is the weight for mel, shape: (batch_size) -> (batch_size, 1)
        # p_mel shape: (batch_size, 128)
        attended_features = (
            p_mel * attention_weights[:, 0].unsqueeze(1) +
            p_emb * attention_weights[:, 1].unsqueeze(1) +
            p_genre * attention_weights[:, 2].unsqueeze(1) +
            p_audio * attention_weights[:, 3].unsqueeze(1)
        )
        
        # --- 4. Final decision ---
        output = self.fusion_head(attended_features)
        return output

# ═══════════＝  6. Dataset and Collate Function ＝═══════════
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
        if pd.isna(emotion_str) or not emotion_str:
            emotion_indices = []
        else:
            emotions = emotion_str.split(',')
            emotion_indices = [self.emotion_to_idx[emo] for emo in emotions if emo in self.emotion_to_idx]
        
        label_seq = torch.tensor(emotion_indices, dtype=torch.long)
        
        original_emotion_list = emotion_str.split(',') if pd.notna(emotion_str) and emotion_str else []

        return mel_spec, embedding, genre_vector, audio_features, label_seq, original_emotion_list

def ranking_collate_fn(batch):
    mels, embs, genres, audios, label_seqs, original_lists = zip(*batch)
    
    max_len_mel = max(mel.shape[1] for mel in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len_mel)
    for i, mel in enumerate(mels): padded_mels[i, :, :mel.shape[1]] = mel
    
    embs_collated = default_collate(embs)
    genres_collated = default_collate(genres)
    audios_collated = default_collate(audios)
    
    max_len_seq = max(len(seq) for seq in label_seqs if len(seq) > 0) if any(len(seq) > 0 for seq in label_seqs) else 0
    padded_labels = torch.full((len(label_seqs), max_len_seq), -1, dtype=torch.long)
    for i, seq in enumerate(label_seqs):
        if len(seq) > 0:
            padded_labels[i, :len(seq)] = seq
            
    return padded_mels, embs_collated, genres_collated, audios_collated, padded_labels, original_lists


# ═══════════＝  7. Experiment Execution Function ＝═══════════
def run_experiment(run_num, full_df, feature_cols, emb_dim, n_genres, genre_to_idx):
    print(f"\n{'='*25} STARTING RUN {run_num}/{NUM_RUNS} {'='*25}")
    
    train_val_df, test_df = train_test_split(full_df, test_size=TEST_SIZE, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=VALIDATION_FRACTION, random_state=42)

    train_dataset = RankingDataset(train_df, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx, EMOTION_TO_IDX)
    val_dataset = RankingDataset(val_df, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx, EMOTION_TO_IDX)
    test_dataset = RankingDataset(test_df, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, n_genres, genre_to_idx, EMOTION_TO_IDX)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ranking_collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=ranking_collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=ranking_collate_fn, num_workers=0)

    print(f"Run {run_num} Data Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    model = QuadFusionNet(emb_dim, n_genres, len(feature_cols), N_EMOTIONS).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = ListNetLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, min_lr=1e-7)
    
    history = {"train_loss": [], "val_loss": [], "val_ndcg_3": []}
    best_val_ndcg = 0.0
    wait_for_es = 0
    
    for epoch in range(1, EPOCHS + 1):
        model.train(); train_loss, train_samples = 0, 0
        for mel, emb, genre, audio, labels, _ in train_loader:
            if mel is None: continue
            mel, emb, genre, audio, labels = mel.to(device), emb.to(device), genre.to(device), audio.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(mel, emb, genre, audio)
            loss = criterion(outputs, labels)
            
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            train_samples += labels.size(0)
            
        history["train_loss"].append(train_loss / train_samples if train_samples > 0 else 0)

        model.eval(); val_loss, val_samples = 0, 0
        all_val_scores = []
        all_val_true_lists = []
        with torch.no_grad():
            for mel, emb, genre, audio, labels, original_lists in val_loader:
                if mel is None: continue
                mel, emb, genre, audio, labels = mel.to(device), emb.to(device), genre.to(device), audio.to(device), labels.to(device)
                outputs = model(mel, emb, genre, audio)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                val_samples += labels.size(0)
                
                all_val_scores.append(outputs.cpu())
                all_val_true_lists.extend(original_lists)

        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0
        
        val_scores_tensor = torch.cat(all_val_scores, dim=0)
        avg_val_ndcg = ndcg_at_k(val_scores_tensor, all_val_true_lists, k=3)
        
        scheduler.step(avg_val_ndcg)
        history["val_loss"].append(avg_val_loss)
        history["val_ndcg_3"].append(avg_val_ndcg.item())
        
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {avg_val_loss:.4f}, Val nDCG@3: {avg_val_ndcg:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if avg_val_ndcg > best_val_ndcg and val_samples > 0:
            best_val_ndcg = avg_val_ndcg
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            wait_for_es = 0
            print(f"  -> New best model saved with Val nDCG@3: {best_val_ndcg:.4f}")
        else:
            wait_for_es += 1
            if USE_EARLY_STOPPING and wait_for_es >= ES_PATIENCE:
                print(f"--- Early stopping triggered at epoch {epoch} ---")
                break
                
    print(f"\nRun {run_num} training finished. Evaluating on the test set...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    
    all_test_scores = []
    all_test_true_lists = []
    with torch.no_grad():
        for mel, emb, genre, audio, _, original_lists in test_loader:
            if mel is None: continue
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            outputs = model(mel, emb, genre, audio)
            all_test_scores.append(outputs.cpu())
            all_test_true_lists.extend(original_lists)
            
    test_scores_tensor = torch.cat(all_test_scores, dim=0)
    test_ndcg_1 = ndcg_at_k(test_scores_tensor, all_test_true_lists, k=1)
    test_ndcg_3 = ndcg_at_k(test_scores_tensor, all_test_true_lists, k=3)
    test_ndcg_5 = ndcg_at_k(test_scores_tensor, all_test_true_lists, k=5)
    
    print(f"Run {run_num} Test Results -> nDCG@1: {test_ndcg_1:.4f} | nDCG@3: {test_ndcg_3:.4f} | nDCG@5: {test_ndcg_5:.4f}")
    
    return test_ndcg_3.item(), history

# ═══════════＝  8. Main Execution Block ＝═══════════
if __name__ == '__main__':
    print("--- > Starting Data Preparation ---")
    try: 
        df = pd.read_csv(DATASET_CSV)
    except FileNotFoundError as e: 
        print(f"FATAL ERROR: CSV file not found -> {e.filename}. Please check your file paths."); sys.exit()
    
    df.dropna(subset=['id', 'emotion_sequence', 'major_genre_name_en'], inplace=True)
    df = df[df['id'] != 'id'].copy(); df['id'] = df['id'].astype(str)
    
    initial_count = len(df)
    missing_mels = {sid for sid in df['id'] if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    missing_muqs = {sid for sid in df['id'] if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
    all_missing = missing_mels.union(missing_muqs)
    
    if all_missing:
        print(f"> Info: Removing {len(all_missing)} records with missing feature files.")
        df = df[~df['id'].isin(all_missing)]
        
    if df.empty: 
        print(f"\nFATAL ERROR: No valid data entries remaining after checking for feature files."); sys.exit()
    
    print(f"> Data preparation complete. Using {len(df)} valid entries.")
    
    EMB_DIM = 1024
    feature_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 
                      'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    
    all_genres_list = sorted(df["major_genre_name_en"].unique())
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    N_GENRES = len(all_genres_list)
    
    all_final_ndcgs, all_histories = [], []
    start_time = time.time()
    
    for i in range(1, NUM_RUNS + 1):
        final_ndcg, history = run_experiment(
            run_num=i, full_df=df, feature_cols=feature_cols, emb_dim=EMB_DIM, 
            n_genres=N_GENRES, genre_to_idx=genre_to_idx
        )
        all_final_ndcgs.append(final_ndcg)
        all_histories.append(history)
        
    end_time = time.time(); print(f"\nTotal execution time for {NUM_RUNS} runs: {end_time - start_time:.2f} seconds.")
    
    print(f"\n\n{'='*20} FINAL RESULTS (FROM BEST RUN) {'='*20}")
    
    if not all_final_ndcgs:
        print("No runs were completed successfully. Exiting."); sys.exit()

    best_run_index = np.argmax(all_final_ndcgs)
    best_history = all_histories[best_run_index]

    print(f"Best Run: #{best_run_index + 1}")
    
    fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig_curves.suptitle(f"Training Curves from Best Run (Run #{best_run_index + 1})", fontsize=16)
    
    epochs_ran = len(best_history["train_loss"])
    ax1.plot(range(1, epochs_ran + 1), best_history["train_loss"], 'o-', label="Train Loss")
    ax1.plot(range(1, epochs_ran + 1), best_history["val_loss"], 'o-', label="Validation Loss")
    ax1.set_title("Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ListNet Loss")
    ax1.legend(); ax1.grid(True)

    ax2.plot(range(1, epochs_ran + 1), best_history["val_ndcg_3"], 'o-', label="Validation nDCG@3", color='green')
    ax2.set_title("Validation nDCG@3 Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("nDCG@3 Score")
    ax2.legend(); ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

    print("\n> Script finished.")