# ─────────────────── Full Evaluation: Genre & Emotion Breakdown ───────────────────
import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════＝ 1. 全局配置 ＝═══════════
MUQ_EMB_DIR = "G:\\13kmid30s_muq"
MEL_FEAT_DIR = "G:\\13kmid30smel"
DATASET_CSV = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon.csv"
MODEL_PATH = "best_ranking_model_by_ndcg2.pth"
OUTPUT_CSV = "evaluation_details.csv"

ALL_EMOTIONS = [
    'Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 'Quiet',
    'Happiness', 'Loneliness', 'Touching', 'Missing', 'Fresh', 'Relaxation'
]
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(ALL_EMOTIONS)}
IDX_TO_EMOTION = {i: emo for i, emo in enumerate(ALL_EMOTIONS)}
N_EMOTIONS = len(ALL_EMOTIONS)

BATCH_SIZE = 64
EMB_DIM = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════＝ 2. 评估指标函数 ＝═══════════
def calculate_ndcg_batch(y_scores, y_true_lists, k=3):
    """ 计算 Batch 中每个样本的 nDCG """
    ndcg_list = []
    for scores, true_list in zip(y_scores, y_true_lists):
        if not true_list:
            ndcg_list.append(0.0); continue
        
        relevance = np.zeros(len(scores))
        true_indices = [EMOTION_TO_IDX[emo] for emo in true_list if emo in EMOTION_TO_IDX]
        if not true_indices:
            ndcg_list.append(0.0); continue
        relevance[true_indices] = 1.0

        top_k_indices = np.argsort(scores)[::-1][:k]
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            dcg += relevance[idx] / np.log2(i + 2.0)

        idcg = 0.0
        num_true = min(k, len(true_indices))
        for i in range(num_true):
            idcg += 1.0 / np.log2(i + 2.0)
        
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
    return np.array(ndcg_list)

# ═══════════＝ 3. 模型定义 ＝═══════════
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

# ═══════════＝ 4. 数据加载 ＝═══════════
class EvalDataset(Dataset):
    def __init__(self, dataframe, mel_dir, muq_dir, feature_cols, n_genres, genre_to_idx):
        self.df = dataframe.reset_index(drop=True)
        self.mel_dir = mel_dir; self.muq_dir = muq_dir; self.feature_cols = feature_cols
        self.n_genres = n_genres; self.genre_to_idx = genre_to_idx

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]; file_id = str(row.id)
        mel_path = os.path.join(self.mel_dir, f"{file_id}.npy")
        mel_spec = torch.from_numpy(np.load(mel_path).astype(np.float32))
        emb_path = os.path.join(self.muq_dir, f"{file_id}.npy")
        embedding = torch.from_numpy(np.load(emb_path).astype(np.float32))
        
        genre_vec = torch.zeros(self.n_genres, dtype=torch.float32)
        if row.major_genre_name_en in self.genre_to_idx:
            genre_vec[self.genre_to_idx[row.major_genre_name_en]] = 1.0
            
        audio_feats = torch.tensor(row[self.feature_cols].values.astype(np.float32), dtype=torch.float32)
        true_emotions = row.emotion_sequence.split(',') if pd.notna(row.emotion_sequence) else []
        
        return mel_spec, embedding, genre_vec, audio_feats, file_id, true_emotions, row.major_genre_name_en

def eval_collate(batch):
    mels, embs, genres, audios, ids, true_emos, genre_names = zip(*batch)
    max_len = max(mel.shape[1] for mel in mels)
    padded_mels = torch.zeros(len(mels), mels[0].shape[0], max_len)
    for i, mel in enumerate(mels): padded_mels[i, :, :mel.shape[1]] = mel
    return padded_mels, default_collate(embs), default_collate(genres), default_collate(audios), ids, true_emos, genre_names

# ═══════════＝ 5. 主逻辑 ＝═══════════
def main():
    print(f"\n{'='*40}\n STARTING FULL EVALUATION (GENRE & EMOTION)\n{'='*40}")
    
    # --- 1. Load Data ---
    try: df = pd.read_csv(DATASET_CSV)
    except: sys.exit("CSV Not Found")
    df.dropna(subset=['id', 'major_genre_name_en', 'emotion_sequence'], inplace=True)
    df = df[df['id'] != 'id'].copy(); df['id'] = df['id'].astype(str)
    
    # 快速过滤文件
    valid_ids = [sid for sid in df['id'] if os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))]
    df = df[df['id'].isin(valid_ids)]
    print(f"> Valid samples: {len(df)}")

    feature_cols = ['median_dE_dt', 'mean_dissonance', 'p90_centroid', 'high_band_ratio', 
                    'low_band_ratio', 'energy_std', 'p10_centroid', 'median_flux']
    all_genres = sorted(df["major_genre_name_en"].unique())
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}
    
    loader = DataLoader(EvalDataset(df, MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, len(all_genres), genre_to_idx),
                        batch_size=BATCH_SIZE, collate_fn=eval_collate)
    
    # --- 2. Load Model ---
    model = QuadFusionNet(EMB_DIM, len(all_genres), len(feature_cols), N_EMOTIONS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- 3. Inference & Accumulation ---
    results_buffer = []
    
    # 初始化情感统计字典
    # Hits@3: 该情感作为True Label出现时，被模型预测进前3的次数
    # Count: 该情感作为True Label出现的总次数
    # Song_nDCG_Sum: 该情感出现的所有歌曲的nDCG总和（用于计算平均nDCG）
    emo_stats = {emo: {'hits_at_3': 0, 'count': 0, 'ndcg_sum': 0.0} for emo in ALL_EMOTIONS}

    print("> Running Inference...")
    with torch.no_grad():
        for mel, emb, genre_vec, audio, fids, true_emos_batch, genre_names in tqdm(loader):
            mel, emb, genre_vec, audio = mel.to(DEVICE), emb.to(DEVICE), genre_vec.to(DEVICE), audio.to(DEVICE)
            scores = model(mel, emb, genre_vec, audio).cpu().numpy()
            
            batch_ndcg3 = calculate_ndcg_batch(scores, true_emos_batch, k=3)
            
            for i, fid in enumerate(fids):
                # 获取 Top 3 预测
                top3_idx = np.argsort(scores[i])[::-1][:3]
                pred_top3 = set([IDX_TO_EMOTION[idx] for idx in top3_idx])
                true_set = set(true_emos_batch[i])
                
                # 记录单条结果
                results_buffer.append({
                    'id': fid, 'genre': genre_names[i], 
                    'true_emotions': ",".join(true_set), 
                    'pred_top3': ",".join(pred_top3), 
                    'ndcg_3': batch_ndcg3[i]
                })

                # --- 核心：按情感统计 ---
                for emo in true_set:
                    if emo in emo_stats:
                        emo_stats[emo]['count'] += 1
                        emo_stats[emo]['ndcg_sum'] += batch_ndcg3[i]
                        if emo in pred_top3:
                            emo_stats[emo]['hits_at_3'] += 1

    # --- 4. 生成分析报表 ---
    res_df = pd.DataFrame(results_buffer)
    global_ndcg = res_df['ndcg_3'].mean()
    print(f"\n🌟 Global nDCG@3: {global_ndcg:.4f}")

    # A. 流派分析 (Genre Analysis)
    genre_df = res_df.groupby('genre')['ndcg_3'].agg(['mean', 'count']).reset_index()
    genre_df.columns = ['Genre', 'nDCG@3', 'Sample_Count']
    genre_df = genre_df.sort_values(by='nDCG@3', ascending=False)
    
    # B. 情感分析 (Emotion Analysis)
    emotion_rows = []
    for emo, stats in emo_stats.items():
        if stats['count'] > 0:
            recall_3 = stats['hits_at_3'] / stats['count'] # 召回率
            avg_ndcg = stats['ndcg_sum'] / stats['count']  # 该情感相关歌曲的平均nDCG
        else:
            recall_3 = 0.0; avg_ndcg = 0.0
        
        emotion_rows.append({
            'Emotion': emo,
            'Recall@3': recall_3,
            'Avg_Song_nDCG': avg_ndcg,
            'Occurrences': stats['count']
        })
    
    emo_df = pd.DataFrame(emotion_rows).sort_values(by='Recall@3', ascending=False)

    print(f"\n{'='*20} REPORT 1: BY GENRE {'='*20}")
    print(genre_df)
    
    print(f"\n{'='*20} REPORT 2: BY EMOTION {'='*20}")
    print(emo_df)

    # --- 5. 可视化 (保存为一张大图) ---
    print("> Generating Visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot 1: Genre Performance
    sns.barplot(data=genre_df, x='nDCG@3', y='Genre', palette='viridis', ax=ax1)
    ax1.set_title(f'Performance by Genre (nDCG@3)', fontsize=14)
    ax1.set_xlim(0, 1.0)
    for i, v in enumerate(genre_df['nDCG@3']):
        ax1.text(v + 0.01, i, f"{v:.2f}", va='center')

    # Plot 2: Emotion Recall (Hardness)
    # 使用 Recall@3 因为它直观反映了模型"捕获"该情感的能力
    sns.barplot(data=emo_df, x='Recall@3', y='Emotion', palette='magma', ax=ax2)
    ax2.set_title(f'Recall@3 by Emotion (Ability to retrieve specific emotion)', fontsize=14)
    ax2.set_xlim(0, 1.0)
    for i, v in enumerate(emo_df['Recall@3']):
        ax2.text(v + 0.01, i, f"{v:.2f}", va='center')

    plt.tight_layout()
    plt.savefig("full_evaluation_report.png")
    print("> Report saved to 'full_evaluation_report.png'")
    
    # Save CSV
    res_df.to_csv(OUTPUT_CSV, index=False)
    print("> Detailed CSV saved.")

if __name__ == '__main__':
    main()