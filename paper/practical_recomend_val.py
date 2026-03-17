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

# ═══════════＝  1. Configurations and Paths ＝═══════════
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

# 你的原始训练参数 (保留它们以便模型加载)
NUM_RUNS = 1
TEST_SIZE = 0.2
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 128 # 评估时可以调大
LEARNING_RATE = 1e-4
EPOCHS = 50
USE_EARLY_STOPPING = True
ES_PATIENCE = 7
BEST_MODEL_PATH = "best_ranking_model_by_ndcg.pth" # 确保这个文件存在!

# ═══════════＝  2. Device Configuration ＝═══════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"> Using device: {device}")


# ═══════════＝  3. ListNet Loss Function ＝═══════════
# (在评估模式下不需要，但保留它以便完整性)
class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()
    def forward(self, y_pred_scores, y_true_seqs):
        total_loss = 0.0; batch_size = y_pred_scores.shape[0]
        for i in range(batch_size):
            scores = y_pred_scores[i]; true_seq = y_true_seqs[i]
            valid_seq = true_seq[true_seq != -1]
            if len(valid_seq) == 0: continue
            ground_truth_scores = scores[valid_seq]
            flipped_scores = torch.flip(ground_truth_scores, dims=[0])
            log_denominators = torch.logcumsumexp(flipped_scores, dim=0)
            log_denominators = torch.flip(log_denominators, dims=[0])
            log_likelihood = torch.sum(ground_truth_scores - log_denominators)
            total_loss -= log_likelihood
        return total_loss / batch_size

# ═══════════＝ 4. Ranking Evaluation Metric (nDCG) ＝═══════════
# (在搜索实验中不需要，但保留)
def ndcg_at_k(y_scores, y_true_lists, k=3):
    batch_ndcg = []
    for scores, true_list in zip(y_scores, y_true_lists):
        if not true_list: continue
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

# ═══════════＝  5. Model Definition ＝═══════════
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
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, n_classes)
        )
    def forward(self, mel, emb, genre, audio_feats):
        mel = mel.unsqueeze(1); z_mel = self.mel_cnn_branch(mel); z_mel = z_mel.view(z_mel.size(0), -1)
        z_emb = self.emb_branch(emb)
        z_genre = self.genre_branch(genre)
        z_audio = self.audio_feature_branch(audio_feats)
        combined = torch.cat([z_mel, z_emb, z_genre, z_audio], dim=1); output = self.fusion_head(combined)
        return output

# ═══════════＝  6. Dataset and Collate Function ＝═══════════
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


# ═══════════＝  7. Experiment Execution Function ＝═══════════
# (在评估模式下我们不调用这个函数，但保留它以防你需要重新训练)
def run_experiment(run_num, full_df, feature_cols, emb_dim, n_genres, genre_to_idx):
    # ... (此函数的全部内容保持不变, 在评估模式下不会被调用) ...
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
            optimizer.zero_grad(); outputs = model(mel, emb, genre, audio); loss = criterion(outputs, labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += loss.item() * labels.size(0); train_samples += labels.size(0)
        history["train_loss"].append(train_loss / train_samples if train_samples > 0 else 0)

        model.eval(); val_loss, val_samples = 0, 0
        all_val_scores = []; all_val_true_lists = []
        with torch.no_grad():
            for mel, emb, genre, audio, labels, original_lists in val_loader:
                if mel is None: continue
                mel, emb, genre, audio, labels = mel.to(device), emb.to(device), genre.to(device), audio.to(device), labels.to(device)
                outputs = model(mel, emb, genre, audio); loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0); val_samples += labels.size(0)
                all_val_scores.append(outputs.cpu()); all_val_true_lists.extend(original_lists)
        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0
        val_scores_tensor = torch.cat(all_val_scores, dim=0)
        avg_val_ndcg = ndcg_at_k(val_scores_tensor, all_val_true_lists, k=3)
        
        scheduler.step(avg_val_ndcg)
        history["val_loss"].append(avg_val_loss); history["val_ndcg_3"].append(avg_val_ndcg.item())
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {avg_val_loss:.4f}, Val nDCG@3: {avg_val_ndcg:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if avg_val_ndcg > best_val_ndcg and val_samples > 0:
            best_val_ndcg = avg_val_ndcg; torch.save(model.state_dict(), BEST_MODEL_PATH); wait_for_es = 0
            print(f"  -> New best model saved with Val nDCG@3: {best_val_ndcg:.4f}")
        else:
            wait_for_es += 1
            if USE_EARLY_STOPPING and wait_for_es >= ES_PATIENCE:
                print(f"--- Early stopping triggered at epoch {epoch} ---"); break
                
    print(f"\nRun {run_num} training finished. Evaluating on the test set...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    
    all_test_scores = []; all_test_true_lists = []
    with torch.no_grad():
        for mel, emb, genre, audio, _, original_lists in test_loader:
            if mel is None: continue
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            outputs = model(mel, emb, genre, audio)
            all_test_scores.append(outputs.cpu()); all_test_true_lists.extend(original_lists)
            
    test_scores_tensor = torch.cat(all_test_scores, dim=0)
    test_ndcg_1 = ndcg_at_k(test_scores_tensor, all_test_true_lists, k=1)
    test_ndcg_3 = ndcg_at_k(test_scores_tensor, all_test_true_lists, k=3)
    test_ndcg_5 = ndcg_at_k(test_scores_tensor, all_test_true_lists, k=5)
    
    print(f"Run {run_num} Test Results -> nDCG@1: {test_ndcg_1:.4f} | nDCG@3: {test_ndcg_3:.4f} | nDCG@5: {test_ndcg_5:.4f}")
    
    return test_ndcg_3.item(), history

# ═══════════＝  8. Main Execution Block (MODIFIED FOR EVALUATION-ONLY) ＝═══════════
if __name__ == '__main__':
    
    # ---------------------------------------------------------------------
    # ❗ START: 搜索实验 (Search Experiment) 评估函数
    # ---------------------------------------------------------------------
    def precision_at_k(ranked_ids, ground_truth_ids, k=10):
        """计算 Precision@k"""
        if not isinstance(ranked_ids, (list, set)): ranked_ids = list(ranked_ids)
        if not isinstance(ground_truth_ids, set): ground_truth_ids = set(ground_truth_ids)
        
        if not ground_truth_ids: return 0.0 # 数据集中没有标准答案
        
        top_k = ranked_ids[:k]
        if not top_k: return 0.0 # 模型没有返回任何结果
        
        hits = len(set(top_k) & ground_truth_ids)
        return hits / k
    # ---------------------------------------------------------------------
    # ❗ END: 搜索实验 (Search Experiment) 评估函数
    # ---------------------------------------------------------------------


    print("--- > Starting Data Preparation (EVALUATION-ONLY MODE) ---")
    try: 
        df = pd.read_csv(DATASET_CSV)
    except FileNotFoundError as e: 
        print(f"FATAL ERROR: CSV file not found -> {e.filename}. Please check your file paths."); sys.exit()
    
    # --- [START] 这一部分的预处理代码必须和训练时完全一致 ---
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
    # --- [END] 预处理代码结束 ---


    # 1. --- 【修改】重新创建 *固定* 的数据集划分 (Test + Val) ---
    # 使用 random_state=42 来确保我们得到与训练时 *完全相同* 的测试集
    print(f"\n--- > Re-creating fixed data split (random_state=42) ---")
    train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    # 从 train_val_df 中进一步划分出 val_df
    train_df, val_df = train_test_split(train_val_df, test_size=VALIDATION_FRACTION, random_state=42)
    
    # 【新增】合并 test 和 val
    eval_df = pd.concat([test_df, val_df])
    print(f"> Evaluation set (Test + Val) size: {len(eval_df)} songs")

    # 2. --- 【修改】创建 Eval Loader ---
    eval_dataset = RankingDataset(eval_df, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, N_GENRES, genre_to_idx, EMOTION_TO_IDX)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, collate_fn=ranking_collate_fn, num_workers=0)

    # 3. --- 加载你已经训练好的模型 ---
    print(f"--- > Loading model and weights from {BEST_MODEL_PATH} ---")
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"FATAL ERROR: Model file not found: {BEST_MODEL_PATH}")
        print("Please run the training first (by un-commenting the code below) or check the path.")
        sys.exit()

    model = QuadFusionNet(EMB_DIM, N_GENRES, len(feature_cols), N_EMOTIONS).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval() # 切换到评估模式
    print("> Model loaded successfully.")

    # 4. --- 【修改】在评估集上运行模型以获取所有预测分数 ---
    print("--- > Running model on evaluation set to get predictions ---")
    all_eval_scores = []
    with torch.no_grad():
        for mel, emb, genre, audio, _, _ in eval_loader: # 我们不需要标签，只评估
            if mel is None: continue
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            outputs = model(mel, emb, genre, audio)
            all_eval_scores.append(outputs.cpu())
            
    eval_scores_tensor = torch.cat(all_eval_scores, dim=0)
    print(f"> Predictions generated for {len(eval_scores_tensor)} songs.")

    # 5. --- ❗ 开始执行搜索 (IR) 实验 ❗ ---
    print(f"\n{'='*20} STARTING SEARCH (IR) EXPERIMENT {'='*20}")

    # 【修改】修复重复列名错误
    # 将模型预测的列重命名 (e.g., 'Romantic_pred')
    pred_cols = [f"{emo}_pred" for emo in ALL_EMOTIONS]
    eval_scores_df = pd.DataFrame(eval_scores_tensor.numpy(), columns=pred_cols)
    
    # 【修改】使用 eval_df
    best_eval_df = eval_df.reset_index(drop=True) # 确保索引对齐
    results_df = pd.concat([best_eval_df, eval_scores_df], axis=1)

    # --- ❗ 【已修改】在这里定义你的查询 ❗ ---
    QUERIES = [
        ('Romantic', 'Jazz'),           # 教授的例子
        ('Loneliness', 'R&B/Soul'),     # <--- 这是你要求的新查询
        ('Healing', 'POP')            # 你的例子 (已修复 'Pop' -> 'POP')
    ]
    
    # --- 检查测试集中有哪些流派，方便你修正查询 ---
    available_genres = set(results_df['major_genre_name_en'])
    print(f"\nAvailable genres in eval set: {available_genres}")
    for _, g in QUERIES:
        if g not in available_genres:
            print(f"WARNING: Query genre '{g}' not found in eval set. This query will return 0 results.")
    print(f"{'-'*60}")

    all_p_at_10 = []

    for (query_emotion, query_genre) in QUERIES:
        
        # A. 找到标准答案 (Ground Truth)
        # 找出评估集中 *真正* 同时拥有这两个标签的歌曲
        ground_truth_df = results_df[
            (results_df['major_genre_name_en'] == query_genre) & 
            (results_df['emotion_sequence'].str.contains(query_emotion, case=False, na=False))
        ]
        ground_truth_ids = set(ground_truth_df['id'])

        # B. 模拟搜索：筛选流派 + 按情感排序
        # 1. 筛选出所有 'query_genre' 的歌曲
        search_pool_df = results_df[results_df['major_genre_name_en'] == query_genre].copy()
        
        if search_pool_df.empty:
            print(f"Query: '{query_emotion} + {query_genre}' | P@10: 0.0000 (No songs found for genre '{query_genre}')")
            continue

        # 2. 关键: 【修改】按照模型对 'query_emotion_pred' 的预测分数进行降序排序
        search_pool_df = search_pool_df.sort_values(by=f"{query_emotion}_pred", ascending=False)
        ranked_ids = list(search_pool_df['id'])

        # C. 评估 P@10
        p_at_10 = precision_at_k(ranked_ids, ground_truth_ids, k=10)
        all_p_at_10.append(p_at_10)
        
        print(f"Query: '{query_emotion} + {query_genre}' | P@10: {p_at_10:.4f} "
              f" (Found {len(ground_truth_ids)} correct songs in eval set)")
        
        # (可选) 打印排名前5的歌曲ID和它们的分数
        # print("  ... Top 5 ranked song IDs:", ranked_ids[:5])

    print(f"{'-'*60}")
    if all_p_at_10:
         print(f"Mean Precision@10 (mP@10) across all queries: {np.mean(all_p_at_10):.4f}")
    else:
        print("No valid queries were run.")


    # -----------------------------------------------------------------
    # --- 原始的训练和绘图代码 (已注释掉，因为我们只做评估) ---
    # -----------------------------------------------------------------
    
    print("\n> Evaluation script finished.")