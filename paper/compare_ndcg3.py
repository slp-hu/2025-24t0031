# ─────────────────── Import Libraries ───────────────────
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import random
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ═══════════＝ 1. Configurations ＝═══════════
# 路径配置
MUQ_EMB_DIR = "G:\\13kmid30s_muq"
MEL_FEAT_DIR = "G:\\13kmid30smel"
DATASET_CSV = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon.csv"

ALL_EMOTIONS = [
    'Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 'Quiet',
    'Happiness', 'Loneliness', 'Touching', 'Missing', 'Fresh', 'Relaxation'
]
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(ALL_EMOTIONS)}
N_EMOTIONS = len(ALL_EMOTIONS)

# --- 关键超参数调整 ---
# 为了跑完整 30s 音频，必须降低 Batch Size
BATCH_SIZE = 8  # 建议从 8 开始尝试，如果显存够大可以加到 12 或 16
NUM_RUNS = 1
TEST_SIZE = 0.2
VALIDATION_FRACTION = 0.2
LEARNING_RATE = 1e-4
EPOCHS = 50
MARGIN = 0.2
EMB_DIM_SHARED = 128
USE_EARLY_STOPPING = True
ES_PATIENCE = 5
BEST_MODEL_PATH = "best_metric_baseline_full_audio.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"> Using device: {device} | Batch Size: {BATCH_SIZE}")

# ═══════════＝ 2. Dataset (无裁剪模式) ＝═══════════
class TripletDataset(Dataset):
    def __init__(self, dataframe, mel_dir, emotion_to_idx, is_train=True):
        self.df = dataframe.reset_index(drop=True)
        self.mel_dir = mel_dir
        self.emotion_to_idx = emotion_to_idx
        self.is_train = is_train
        
        # 预处理索引
        self.tag_to_song_indices = {i: [] for i in range(len(emotion_to_idx))}
        self.song_idx_to_tags = {}
        
        for idx, row in self.df.iterrows():
            emotion_str = row.emotion_sequence
            if pd.isna(emotion_str) or not emotion_str: continue
            
            tags = [self.emotion_to_idx[t] for t in emotion_str.split(',') if t in self.emotion_to_idx]
            self.song_idx_to_tags[idx] = set(tags)
            for t in tags:
                self.tag_to_song_indices[t].append(idx)
                
        self.valid_tags = [t for t in self.tag_to_song_indices if len(self.tag_to_song_indices[t]) > 0]

    def __len__(self):
        return len(self.df)

    def load_mel(self, idx):
        file_id = self.df.iloc[idx].id
        mel_path = os.path.join(self.mel_dir, f"{file_id}.npy")
        # 直接加载完整音频，不裁剪
        mel = np.load(mel_path).astype(np.float32) 
        return torch.from_numpy(mel).unsqueeze(0) # [1, Freq, Time]

    def __getitem__(self, index):
        if self.is_train:
            # Triplet Sampling
            anchor_tag_idx = random.choice(self.valid_tags)
            pos_song_idx = random.choice(self.tag_to_song_indices[anchor_tag_idx])
            
            while True:
                neg_song_idx = random.randint(0, len(self.df) - 1)
                # 确保是 Negative (该歌曲不包含 anchor tag)
                if neg_song_idx in self.song_idx_to_tags:
                    if anchor_tag_idx not in self.song_idx_to_tags[neg_song_idx]:
                        break
            
            mel_pos = self.load_mel(pos_song_idx)
            mel_neg = self.load_mel(neg_song_idx)
            return torch.tensor(anchor_tag_idx, dtype=torch.long), mel_pos, mel_neg
        else:
            # Eval Mode
            mel = self.load_mel(index)
            true_tags = list(self.song_idx_to_tags.get(index, []))
            labels_tensor = torch.full((len(self.emotion_to_idx),), -1, dtype=torch.long)
            if len(true_tags) > 0:
                labels_tensor[:len(true_tags)] = torch.tensor(true_tags)
            return mel, labels_tensor, true_tags

# ═══════════＝ 3. Collate Function (关键修复：Padding) ＝═══════════
def pad_to_max(tensor_list):
    """
    找到 Batch 中最长的时间步，将所有 Tensor 补齐到该长度。
    tensor shape: [1, Freq, Time]
    """
    max_len = max(t.shape[2] for t in tensor_list)
    padded_list = []
    for t in tensor_list:
        pad_amount = max_len - t.shape[2]
        if pad_amount > 0:
            # F.pad 参数顺序: (左, 右, 上, 下...)
            t = F.pad(t, (0, pad_amount), "constant", 0)
        padded_list.append(t)
    return torch.stack(padded_list)

def triplet_collate_fn(batch):
    first_item = batch[0]
    
    # 判断是否为 Eval 模式 (第一项是 Tensor 且是 Mel 图)
    is_eval = torch.is_tensor(first_item[0]) and first_item[0].ndim > 1
    
    if is_eval:
        # Eval: (mel, labels, list)
        mels, labels, original_lists = zip(*batch)
        # 修复 stack error: 先 Pad 再 Stack
        mels_pad = pad_to_max(mels)
        labels_stack = torch.stack(labels)
        return mels_pad, labels_stack, original_lists
    else:
        # Train: (anchor, pos, neg)
        anchors, pos_mels, neg_mels = zip(*batch)
        anchors = torch.stack(anchors)
        
        # 修复 stack error: Pos 和 Neg 都要 Pad 到当前 Batch 的最大长度
        # 为了高效，我们可以让 Pos 和 Neg 统一 Pad 到 Pos+Neg 所有样本的最大值
        combined_mels = pos_mels + neg_mels
        max_len = max(t.shape[2] for t in combined_mels)
        
        def pad_one(t, length):
            pad_amt = length - t.shape[2]
            return F.pad(t, (0, pad_amt)) if pad_amt > 0 else t
            
        pos_mels_pad = torch.stack([pad_one(m, max_len) for m in pos_mels])
        neg_mels_pad = torch.stack([pad_one(m, max_len) for m in neg_mels])
        
        return anchors, pos_mels_pad, neg_mels_pad

# ═══════════＝ 4. Model Architecture ＝═══════════
class MetricBaselineNet(nn.Module):
    def __init__(self, n_tags, tag_emb_dim=300, shared_dim=128):
        super().__init__()
        
        # Branch A: Audio Encoder (CNN)
        # AdaptiveAvgPool2d 使得模型可以接受任意长度的输入
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.AdaptiveAvgPool2d((1, 1)), # 这里把任意长度 (Time) 压缩为 1x1
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, shared_dim)
        )
        
        # Branch B: Tag Encoder
        self.word_embeddings = nn.Embedding(n_tags, tag_emb_dim)
        self.tag_encoder = nn.Sequential(
            nn.Linear(tag_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, shared_dim)
        )
        
    def forward_audio(self, mel):
        return self.audio_encoder(mel)

    def forward_tag(self, tag_indices):
        vecs = self.word_embeddings(tag_indices)
        return self.tag_encoder(vecs)

# ═══════════＝ 5. Evaluation Logic ＝═══════════
def calculate_retrieval_ndcg(model, val_loader, all_tag_indices, device, k=5):
    model.eval()
    all_song_embs = []
    all_true_labels = [] 
    
    with torch.no_grad():
        for mels, _, true_tags_list in val_loader:
            mels = mels.to(device)
            embs = model.forward_audio(mels)
            embs = F.normalize(embs, p=2, dim=1)
            all_song_embs.append(embs)
            all_true_labels.extend([set(t) for t in true_tags_list])
            
    if not all_song_embs: return 0.0
    songs_tensor = torch.cat(all_song_embs, dim=0)
    
    tag_indices_tensor = torch.tensor(all_tag_indices, device=device)
    with torch.no_grad():
        tag_embs = model.forward_tag(tag_indices_tensor)
        tag_embs = F.normalize(tag_embs, p=2, dim=1)
        
    similarity_matrix = torch.matmul(tag_embs, songs_tensor.t())
    
    mean_ndcg = 0.0
    valid_count = 0
    
    for i, tag_idx in enumerate(all_tag_indices):
        scores = similarity_matrix[i]
        relevance = torch.tensor(
            [1.0 if tag_idx in song_labels else 0.0 for song_labels in all_true_labels], 
            device=device
        )
        
        if relevance.sum() == 0: continue
            
        _, pred_indices = torch.topk(scores, k)
        pred_rel = relevance[pred_indices]
        dcg = (pred_rel / torch.log2(torch.arange(2, k + 2, device=device).float())).sum()
        
        true_rel_sorted, _ = torch.sort(relevance, descending=True)
        idcg = (true_rel_sorted[:k] / torch.log2(torch.arange(2, k + 2, device=device).float())).sum()
        
        ndcg = (dcg / idcg).item() if idcg > 0 else 0.0
        mean_ndcg += ndcg
        valid_count += 1
        
    return mean_ndcg / valid_count if valid_count > 0 else 0.0

# ═══════════＝ 6. Main Execution ＝═══════════
def run_metric_baseline():
    print("--- > Starting Metric Learning Baseline (Full Audio Version) ---")
    
    try: 
        df = pd.read_csv(DATASET_CSV)
    except FileNotFoundError:
        print("CSV not found."); return

    df.dropna(subset=['id', 'emotion_sequence'], inplace=True)
    df = df[df['id'] != 'id'].copy(); df['id'] = df['id'].astype(str)
    
    # Filter missing files
    valid_ids = []
    for x in df['id']:
        if os.path.exists(os.path.join(MEL_FEAT_DIR, f"{x}.npy")):
            valid_ids.append(x)
    df = df[df['id'].isin(valid_ids)]
    print(f"Valid songs: {len(df)}")
    
    train_val_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=VALIDATION_FRACTION, random_state=42)
    
    # Dataset Init (Train/Eval split)
    train_ds = TripletDataset(train_df, MEL_FEAT_DIR, EMOTION_TO_IDX, is_train=True)
    val_ds = TripletDataset(val_df, MEL_FEAT_DIR, EMOTION_TO_IDX, is_train=False)
    test_ds = TripletDataset(test_df, MEL_FEAT_DIR, EMOTION_TO_IDX, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=triplet_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=triplet_collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=triplet_collate_fn, num_workers=0)
    
    model = MetricBaselineNet(n_tags=N_EMOTIONS, shared_dim=EMB_DIM_SHARED).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_ndcg = 0.0
    wait = 0
    all_tag_indices = list(range(N_EMOTIONS))
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        steps = 0
        
        for batch in train_loader:
            anchor_idx, mel_pos, mel_neg = batch
            anchor_idx = anchor_idx.to(device)
            mel_pos = mel_pos.to(device)
            mel_neg = mel_neg.to(device)
            
            optimizer.zero_grad()
            
            anchor_emb = model.forward_tag(anchor_idx)
            pos_emb = model.forward_audio(mel_pos)
            neg_emb = model.forward_audio(mel_neg)
            
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        avg_loss = total_loss / steps if steps > 0 else 0
        
        val_ndcg = calculate_retrieval_ndcg(model, val_loader, all_tag_indices, device, k=3)
        scheduler.step(val_ndcg)
        
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val nDCG@3: {val_ndcg:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New Best Saved: {best_ndcg:.4f}")
            wait = 0
        else:
            wait += 1
            if USE_EARLY_STOPPING and wait >= ES_PATIENCE:
                print("Early stopping."); break

    print("\nEvaluating Best Model on Test Set...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    ndcg_3 = calculate_retrieval_ndcg(model, test_loader, all_tag_indices, device, k=3)
    print(f"Final Test nDCG@3: {ndcg_3:.4f}")

if __name__ == '__main__':
    run_metric_baseline()