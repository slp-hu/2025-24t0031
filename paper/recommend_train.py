# -*- coding: utf-8 -*-
# ===================== Pre-split Version (NaN-hardened) ======================
# Uses pre-made splits: train_split.csv / val_split.csv / dev_val_plus_test.csv
# - Cleans NaN/Inf; robust ListNet; feature standardization (train-only stats)
# - Saves/loads metadata (genre_to_idx, feature means/stds)
# - Inference CSV schema: id, name, artist, major_genre_name_en, emotion_vector(JSON array of 12 logits),
#   emotion_sequence (pred top-3), gt_emotion_sequence (from CSV), file_path (G:\13kmid30s\<id>.mp3)

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─────────────────── Configs ───────────────────
# Feature directories
MUQ_EMB_DIR = r"G:\13kmid30s_muq"
MEL_FEAT_DIR = r"G:\13kmid30smel"

# Pre-split CSVs
DATA_DIR = r"C:\Users\YAO\Desktop\genre ml\newdata\split"
TRAIN_CSV = os.path.join(DATA_DIR, "train_split.csv")
VAL_CSV = os.path.join(DATA_DIR, "val_split.csv")
LIBRARY_INPUT_CSV = os.path.join(DATA_DIR, "dev_val_plus_test.csv")

# Outputs
BEST_MODEL_PATH = "best_ranking_model.pth"
LIB_META_PATH   = "training_meta.json"
LIBRARY_OUTPUT_FILE = "emotion_library.csv"

# Labels
ALL_EMOTIONS = [
    'Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 'Quiet',
    'Happiness', 'Loneliness', 'Touching', 'Missing', 'Fresh', 'Relaxation'
]
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(ALL_EMOTIONS)}
IDX_TO_EMOTION = {i: emo for i, emo in enumerate(ALL_EMOTIONS)}
N_EMOTIONS = len(ALL_EMOTIONS)

# Feature columns (tabular)
FEATURE_COLS = [
    'median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio',
    'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux'
]

# Hyper-params
EMB_DIM = 1024
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 50
USE_EARLY_STOPPING = True
ES_PATIENCE = 7

# Use LayerNorm for audio branch (more stable than BN for tabular)
USE_LAYER_NORM_FOR_AUDIO = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"> Using device: {device}")

# ─────────────────── Helpers ───────────────────
def np_sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def torch_sanitize(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

def assert_finite(name, t, ids=None):
    if not torch.isfinite(t).all():
        print(f"[BAD] {name} has non-finite values. shape={tuple(t.shape)}")
        if ids is not None:
            print(" sample ids (first 8):", ids[:8])
        return False
    return True

# ─────────────────── Loss & Metric ───────────────────
class ListNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, y_pred_scores, y_true_seqs):
        total_ll = 0.0
        used = 0
        B = y_pred_scores.shape[0]
        for i in range(B):
            scores = torch_sanitize(y_pred_scores[i])
            true_seq = y_true_seqs[i]
            valid = true_seq[true_seq != -1]
            if valid.numel() == 0:
                continue
            gt = scores[valid]
            if not torch.isfinite(gt).all():
                continue
            flipped = torch.flip(gt, dims=[0])
            log_den = torch.logcumsumexp(flipped, dim=0)
            log_den = torch.flip(log_den, dims=[0])
            ll = torch.sum(gt - log_den)
            if torch.isfinite(ll):
                total_ll += ll
                used += 1
        if used == 0:
            return y_pred_scores.new_tensor(0.0)
        return - total_ll / used

def ndcg_at_k(y_scores, y_true_lists, k=3):
    batch_ndcg = []
    for scores, true_list in zip(y_scores, y_true_lists):
        if not true_list:
            continue
        relevance = torch.zeros_like(scores)
        true_indices = [EMOTION_TO_IDX.get(emo, None) for emo in true_list]
        true_indices = [i for i in true_indices if i is not None]
        if not true_indices:
            continue
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
    return torch.mean(torch.stack(batch_ndcg)) if batch_ndcg else torch.tensor(0.0, device=y_scores.device)

# ─────────────────── Model ───────────────────
class QuadFusionNet(nn.Module):
    def __init__(self, emb_dim, genre_dim, audio_feature_dim, n_classes, use_ln_audio=True):
        super().__init__()
        self.mel_cnn_branch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)), nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.emb_branch = nn.Sequential(nn.Linear(emb_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5))
        self.genre_branch = nn.Sequential(nn.Linear(genre_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5))
        if use_ln_audio:
            self.audio_feature_branch = nn.Sequential(nn.LayerNorm(audio_feature_dim), nn.ReLU(), nn.Dropout(0.5))
        else:
            self.audio_feature_branch = nn.Sequential(nn.BatchNorm1d(audio_feature_dim), nn.ReLU(), nn.Dropout(0.5))
        fusion_input_dim = 256 + 128 + 32 + audio_feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, n_classes)
        )
    def forward(self, mel, emb, genre, audio_feats):
        mel = mel.unsqueeze(1)                 # (B, 1, F, T)
        z_mel = self.mel_cnn_branch(mel)      # (B, 256, 1, 1)
        z_mel = z_mel.view(z_mel.size(0), -1) # (B, 256)
        z_emb = self.emb_branch(emb)
        z_genre = self.genre_branch(genre)
        z_audio = self.audio_feature_branch(audio_feats)
        combined = torch.cat([z_mel, z_emb, z_genre, z_audio], dim=1)
        output = self.fusion_head(combined)
        return output

# ─────────────────── Dataset ───────────────────
class RankingDataset(Dataset):
    def __init__(self, dataframe, mel_dir, muq_dir, feature_cols,
                 n_genres, genre_to_idx, emotion_to_idx,
                 feat_means=None, feat_stds=None, eps=1e-8):
        self.df = dataframe.reset_index(drop=True)
        self.mel_dir, self.muq_dir = mel_dir, muq_dir
        self.feature_cols = list(feature_cols)
        self.n_genres = n_genres
        self.genre_to_idx = dict(genre_to_idx)
        self.emotion_to_idx = dict(emotion_to_idx)
        self.eps = float(eps)
        if feat_means is not None and feat_stds is not None:
            self.feat_means = np.asarray([feat_means[c] for c in self.feature_cols], dtype=np.float32)
            stds = np.asarray([feat_stds[c] for c in self.feature_cols], dtype=np.float32)
            stds[stds == 0] = 1.0
            self.feat_stds = stds
        else:
            self.feat_means, self.feat_stds = None, None

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        file_id = str(row.id)

        # mel
        mel_path = os.path.join(self.mel_dir, f"{file_id}.npy")
        mel_spec = np_sanitize(np.load(mel_path).astype(np.float32))
        mel_spec = torch.from_numpy(mel_spec)

        # embedding
        emb_path = os.path.join(self.muq_dir, f"{file_id}.npy")
        embedding = np_sanitize(np.load(emb_path).astype(np.float32))
        embedding = torch.from_numpy(embedding)

        # genre one-hot
        genre_vector = torch.zeros(self.n_genres, dtype=torch.float32)
        genre = str(row.major_genre_name_en) if pd.notna(row.major_genre_name_en) else ""
        if genre in self.genre_to_idx:
            genre_vector[self.genre_to_idx[genre]] = 1.0

        # tabular features -> sanitize -> (x - mean)/std
        feat_vals = row[self.feature_cols].astype(np.float32).values
        feat_vals = np_sanitize(feat_vals)
        if self.feat_means is not None and self.feat_stds is not None:
            feat_vals = (feat_vals - self.feat_means) / (self.feat_stds + self.eps)
        audio_features = torch.from_numpy(feat_vals).to(torch.float32)

        # labels (sequence of emotions)
        emotion_str = row.emotion_sequence if pd.notna(row.emotion_sequence) else ""
        emotion_indices = []
        if emotion_str:
            emotions = [e.strip() for e in str(emotion_str).split(',') if e.strip()]
            emotion_indices = [self.emotion_to_idx[e] for e in emotions if e in self.emotion_to_idx]
        label_seq = torch.tensor(emotion_indices, dtype=torch.long)
        original_emotion_list = [e for e in str(emotion_str).split(',') if e] if emotion_str else []

        return mel_spec, embedding, genre_vector, audio_features, label_seq, original_emotion_list, file_id

def ranking_collate_fn(batch):
    mels, embs, genres, audios, label_seqs, original_lists, ids = zip(*batch)
    # pad mel along time
    max_len_mel = max(mel.shape[1] for mel in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len_mel)
    for i, mel in enumerate(mels):
        padded_mels[i, :, :mel.shape[1]] = mel
    embs_collated = default_collate(embs)
    genres_collated = default_collate(genres)
    audios_collated = default_collate(audios)
    # pad label sequences
    max_len_seq = max((len(seq) for seq in label_seqs), default=0)
    padded_labels = torch.full((len(label_seqs), max_len_seq), -1, dtype=torch.long)
    for i, seq in enumerate(label_seqs):
        if len(seq) > 0:
            padded_labels[i, :len(seq)] = seq
    return padded_mels, embs_collated, genres_collated, audios_collated, padded_labels, original_lists, list(ids)

# ─────────────────── Sanity Check ───────────────────
def quick_sanity_check(dloader, emb_dim, n_genres, audio_dim, n_emotions, n_batches=2):
    print("[SANITY] Running quick sanity check ...")
    model = QuadFusionNet(emb_dim, n_genres, audio_dim, n_emotions, use_ln_audio=USE_LAYER_NORM_FOR_AUDIO).to(device)
    model.eval()
    with torch.no_grad():
        it = iter(dloader)
        for _ in range(n_batches):
            mel, emb, genre, audio, labels, orig, ids = next(it)
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            ok = True
            ok &= assert_finite("mel", mel, ids)
            ok &= assert_finite("emb", emb, ids)
            ok &= assert_finite("genre", genre, ids)
            ok &= assert_finite("audio", audio, ids)
            out = model(mel, emb, genre, audio)
            out = torch_sanitize(out)
            if not torch.isfinite(out).all():
                print("[SANITY] model outputs contain non-finite values")
            crit = ListNetLoss()
            l = crit(out, labels.to(device))
            print(f"[SANITY] dummy loss = {float(l):.6f}")
    print("[SANITY] Done.")

# ─────────────────── Train ───────────────────
def train(train_csv_path, val_csv_path):
    print(f"\n{'='*25} PHASE 1: MODEL TRAINING {'='*25}")

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    for df in (train_df, val_df):
        if 'id' in df.columns: df['id'] = df['id'].astype(str)

    # Drop rows missing feature files
    def drop_missing_files(frame: pd.DataFrame) -> pd.DataFrame:
        ids = frame['id'].astype(str).tolist()
        missing_mels = {sid for sid in ids if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
        missing_muqs = {sid for sid in ids if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
        missing = missing_mels.union(missing_muqs)
        if missing:
            print(f"> Info: dropping {len(missing)} rows with missing feature files.")
            frame = frame[~frame['id'].isin(missing)].copy()
        return frame

    train_df = drop_missing_files(train_df)
    val_df   = drop_missing_files(val_df)
    if train_df.empty or val_df.empty:
        print("FATAL: Empty train/val after file check."); sys.exit(1)

    # Train-only stats for tabular features
    feat_means = train_df[FEATURE_COLS].mean(numeric_only=True)
    feat_stds  = train_df[FEATURE_COLS].std(numeric_only=True).replace(0, 1.0)

    # Unified genre mapping (train+val)
    all_genres_list = sorted(pd.concat([train_df, val_df])["major_genre_name_en"].dropna().astype(str).unique())
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    N_GENRES = len(all_genres_list)

    # Datasets / Loaders
    train_dataset = RankingDataset(train_df, MEL_FEAT_DIR, MUQ_EMB_DIR, FEATURE_COLS,
                                   N_GENRES, genre_to_idx, EMOTION_TO_IDX,
                                   feat_means=feat_means, feat_stds=feat_stds)
    val_dataset   = RankingDataset(val_df,   MEL_FEAT_DIR, MUQ_EMB_DIR, FEATURE_COLS,
                                   N_GENRES, genre_to_idx, EMOTION_TO_IDX,
                                   feat_means=feat_means, feat_stds=feat_stds)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=ranking_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=ranking_collate_fn, num_workers=0)

    print(f"Data Loaded: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # Quick sanity
    quick_sanity_check(train_loader, EMB_DIM, N_GENRES, len(FEATURE_COLS), N_EMOTIONS, n_batches=2)

    # Model / Optim
    model = QuadFusionNet(EMB_DIM, N_GENRES, len(FEATURE_COLS), N_EMOTIONS,
                          use_ln_audio=USE_LAYER_NORM_FOR_AUDIO).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = ListNetLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3, min_lr=1e-7)

    history = {"train_loss": [], "val_loss": [], "val_ndcg_3": []}
    best_val_ndcg = 0.0
    wait_for_es = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_samples = 0.0, 0
        for mel, emb, genre, audio, labels, _, ids in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]"):
            mel, emb, genre, audio, labels = mel.to(device), emb.to(device), genre.to(device), audio.to(device), labels.to(device)

            ok = True
            ok &= assert_finite("mel", mel, ids)
            ok &= assert_finite("emb", emb, ids)
            ok &= assert_finite("genre", genre, ids)
            ok &= assert_finite("audio", audio, ids)
            if not ok:
                continue

            optimizer.zero_grad()
            outputs = model(mel, emb, genre, audio)
            outputs = torch_sanitize(outputs)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                print("[WARN] non-finite loss encountered; skipping this batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_samples += labels.size(0)

        history["train_loss"].append(train_loss / train_samples if train_samples > 0 else 0.0)

        # Validation
        model.eval()
        val_loss, val_samples = 0.0, 0
        all_val_scores, all_val_true_lists = [], []
        with torch.no_grad():
            for mel, emb, genre, audio, labels, original_lists, _ in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{EPOCHS} [Val]"):
                mel, emb, genre, audio, labels = mel.to(device), emb.to(device), genre.to(device), audio.to(device), labels.to(device)
                outputs = model(mel, emb, genre, audio)
                outputs = torch_sanitize(outputs)
                loss = criterion(outputs, labels)
                if not torch.isfinite(loss):
                    continue
                val_loss += loss.item() * labels.size(0)
                val_samples += labels.size(0)
                all_val_scores.append(outputs.detach())
                all_val_true_lists.extend(original_lists)

        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0.0
        val_scores_tensor = torch.cat(all_val_scores, dim=0) if all_val_scores else torch.empty(0, N_EMOTIONS, device=device)
        avg_val_ndcg = ndcg_at_k(val_scores_tensor, all_val_true_lists, k=3) if val_scores_tensor.numel() else torch.tensor(0.0, device=device)

        scheduler.step(avg_val_ndcg)
        history["val_loss"].append(avg_val_loss)
        history["val_ndcg_3"].append(avg_val_ndcg.item())

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {history['train_loss'][-1]:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val nDCG@3: {avg_val_ndcg:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_ndcg > best_val_ndcg and val_samples > 0:
            best_val_ndcg = avg_val_ndcg
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            # Save meta for inference reproducibility
            meta = {
                "genre_to_idx": genre_to_idx,
                "feature_cols": FEATURE_COLS,
                "feature_means": {c: float(feat_means[c]) for c in FEATURE_COLS},
                "feature_stds":  {c: float(feat_stds[c])  for c in FEATURE_COLS},
                "emb_dim": EMB_DIM,
                "n_emotions": N_EMOTIONS,
                "use_ln_audio": USE_LAYER_NORM_FOR_AUDIO
            }
            with open(LIB_META_PATH, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            wait_for_es = 0
            print(f"  -> New best model saved with Val nDCG@3: {best_val_ndcg:.4f}")
        else:
            wait_for_es += 1
            if USE_EARLY_STOPPING and wait_for_es >= ES_PATIENCE:
                print(f"--- Early stopping triggered at epoch {epoch} ---")
                break

    return BEST_MODEL_PATH, history

# ─────────────────── Inference ───────────────────
def generate_emotion_library(model_path, library_csv_path, output_path, meta_path=LIB_META_PATH):
    print(f"\n{'='*25} PHASE 2: GENERATING EMOTION LIBRARY {'='*25}")
    # Load training meta
    if not os.path.exists(meta_path):
        print(f"FATAL: cannot find meta file: {meta_path}")
        sys.exit(1)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    genre_to_idx = meta["genre_to_idx"]
    feat_means = pd.Series(meta["feature_means"])
    feat_stds  = pd.Series(meta["feature_stds"])
    n_genres = len(genre_to_idx)
    use_ln_audio = bool(meta.get("use_ln_audio", True))

    # Load library CSV
    library_df = pd.read_csv(library_csv_path)
    if 'id' in library_df.columns: library_df['id'] = library_df['id'].astype(str)
    print(f"Library data loaded: {len(library_df)} rows from '{library_csv_path}'")

    # Drop rows missing feature files
    ids = library_df['id'].astype(str).tolist()
    missing_mels = {sid for sid in ids if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    missing_muqs = {sid for sid in ids if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
    missing = missing_mels.union(missing_muqs)
    if missing:
        print(f"> Info: dropping {len(missing)} library rows with missing feature files.")
        library_df = library_df[~library_df['id'].isin(missing)].copy()

    # Dataset/Loader with training stats
    lib_dataset = RankingDataset(
        library_df, MEL_FEAT_DIR, MUQ_EMB_DIR, FEATURE_COLS,
        n_genres, genre_to_idx, EMOTION_TO_IDX,
        feat_means=feat_means, feat_stds=feat_stds
    )
    lib_loader = DataLoader(lib_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=ranking_collate_fn, num_workers=0)

    # Model
    model = QuadFusionNet(EMB_DIM, n_genres, len(FEATURE_COLS), N_EMOTIONS,
                          use_ln_audio=use_ln_audio).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Best model '{model_path}' loaded for inference.")

    # Forward
    all_logits = []
    with torch.no_grad():
        for mel, emb, genre, audio, _, _, _ in tqdm(lib_loader, desc="Generating logits"):
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            outputs = model(mel, emb, genre, audio)     # (B, 12)
            outputs = torch_sanitize(outputs)
            all_logits.append(outputs.cpu().numpy())
    if not all_logits:
        print("No vectors generated. Exiting.")
        return
    logits = np.concatenate(all_logits, axis=0)         # (N, 12)

    # Assemble output rows with requested schema
    out_rows = []
    for i in range(len(library_df)):
        sid = str(library_df.iloc[i].get('id', ''))
        name = library_df.iloc[i].get('name', '')
        artist = library_df.iloc[i].get('artist', '')
        genre_name = library_df.iloc[i].get('major_genre_name_en', '')
        gt_seq = library_df.iloc[i].get('emotion_sequence', '')

        vec = logits[i].astype(float)
        topk_idx = np.argsort(vec)[-3:][::-1]
        pred_labels = [IDX_TO_EMOTION[j] for j in topk_idx]
        pred_seq = ",".join(pred_labels)

        file_path = f"G:\\13kmid30s\\{sid}.mp3"
        vec_json = json.dumps([float(x) for x in vec], ensure_ascii=False)

        out_rows.append({
            "id": sid,
            "name": name,
            "artist": artist,
            "major_genre_name_en": genre_name,
            "emotion_vector": vec_json,
            "emotion_sequence": pred_seq,          # predicted Top-3
            "gt_emotion_sequence": gt_seq,         # ground truth from CSV
            "file_path": file_path
        })

    out_df = pd.DataFrame(out_rows, columns=[
        "id", "name", "artist", "major_genre_name_en",
        "emotion_vector", "emotion_sequence", "gt_emotion_sequence", "file_path"
    ])
    out_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Emotion library saved to '{output_path}'")
    print(out_df.head())

# ─────────────────── Main ───────────────────
if __name__ == '__main__':
    # Phase 1: Train
    best_model_path, history = train(train_csv_path=TRAIN_CSV, val_csv_path=VAL_CSV)
    print(f"\n--- Training complete. Best model at '{best_model_path}' ---")

    # Phase 2: Inference on dev (val+test)
    generate_emotion_library(
        model_path=best_model_path,
        library_csv_path=LIBRARY_INPUT_CSV,
        output_path=LIBRARY_OUTPUT_FILE
    )

    # Optional: visualize curves
    print("\n--- Generating training visualizations ---")
    fig_curves, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig_curves.suptitle(f"Training Curves of the Best Model", fontsize=16)
    epochs_ran = len(history["train_loss"])
    ax1.plot(range(1, epochs_ran + 1), history["train_loss"], 'o-', label="Train Loss")
    ax1.plot(range(1, epochs_ran + 1), history["val_loss"], 'o-', label="Validation Loss")
    ax1.set_title("Loss Curves"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("ListNet Loss"); ax1.legend(); ax1.grid(True)
    ax2.plot(range(1, epochs_ran + 1), history["val_ndcg_3"], 'o-', label="Validation nDCG@3", color='green')
    ax2.set_title("Validation nDCG@3 Curve"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("nDCG@3 Score"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n> Script finished.")
