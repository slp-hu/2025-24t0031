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
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# ═══════════＝  1. Configurations and Paths ＝═══════════
MEL_FEAT_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\mel"
MUQ_EMB_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\muq"
AUDIO_FEATURES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\audio_features_normalized.csv"
GENRES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\processed_genres.csv"
LABELS_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\quadrants_from_dynamic.csv"

NUM_RUNS = 1
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


# ═══════════＝  3. Model Definition (QuadFusionNet) ＝═══════════
class QuadFusionNet(nn.Module):
    def __init__(self, emb_dim, genre_dim, audio_feature_dim, n_classes):
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
        
        # Branch 3: Genre
        self.genre_branch = nn.Sequential(
            nn.Linear(genre_dim, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5)
        )
        
        # Branch 4: Audio Features
        self.audio_feature_branch = nn.Sequential(
            nn.BatchNorm1d(audio_feature_dim), nn.ReLU(), nn.Dropout(0.5)
        )

        # Updated Fusion Head
        cnn_output_dim = 256
        emb_output_dim = 128
        genre_output_dim = 32
        fusion_input_dim = cnn_output_dim + emb_output_dim + genre_output_dim + audio_feature_dim
        
        # Split fully connected layers for feature extraction
        self.fc1 = nn.Sequential(nn.Linear(fusion_input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4))
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, mel, emb, genre, audio_feats, return_embeds=False):
        mel = mel.unsqueeze(1)
        z_mel = self.mel_cnn_branch(mel)
        z_mel = z_mel.view(z_mel.size(0), -1)
        z_emb = self.emb_branch(emb)
        z_genre = self.genre_branch(genre)
        z_audio = self.audio_feature_branch(audio_feats)
        
        combined = torch.cat([z_mel, z_emb, z_genre, z_audio], dim=1)
        x = self.fc1(combined)
        embedding = self.fc2(x) 
        output = self.classifier(embedding)
        
        if return_embeds:
            return output, embedding
        else:
            return output

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
        song_id = str(row.song_id)
        
        mel_path = os.path.join(self.mel_dir, f"{song_id}.npy")
        mel_spec = torch.from_numpy(np.load(mel_path).astype(np.float32))
        
        emb_path = os.path.join(self.muq_dir, f"{song_id}.npy")
        embedding = torch.from_numpy(np.load(emb_path).astype(np.float32))
        if embedding.ndim > 1: embedding = embedding.squeeze()

        genre_vector = torch.zeros(self.n_genres, dtype=torch.float32)
        genres = str(row.normalized_genres).split(';')
        weights = 1 / np.arange(1, len(genres) + 1); weights /= weights.sum()
        for k, g in enumerate(genres):
            if g in self.genre_to_idx: genre_vector[self.genre_to_idx[g]] = weights[k]
        
        audio_features = torch.tensor(row[self.feature_cols].values.astype(np.float32), dtype=torch.float32)
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

# ═══════════＝  5. Visualization Helpers (Optimized) ＝═══════════
def extract_learned_features(model, loader, device):
    model.eval()
    all_embeds, all_labels = [], []
    print("Extracting learned features from QuadFusionNet...")
    with torch.no_grad():
        for mel, emb, genre, audio, labels in loader:
            mel, emb, genre, audio, labels = mel.to(device), emb.to(device), genre.to(device), audio.to(device), labels.to(device)
            _, embeddings = model(mel, emb, genre, audio, return_embeds=True)
            all_embeds.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_embeds), np.concatenate(all_labels)

def plot_decision_boundary(X, y, class_names, title="Learned Feature Space & Decision Boundaries"):
    print(f"Generating Decision Boundary Plot: {title}...")
    
    # 1. t-SNE
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto', perplexity=30)
    X_2d = tsne.fit_transform(X)
    
    # 2. SVM for boundary simulation
    clf = SVC(kernel='rbf', C=10, gamma='auto')
    clf.fit(X_2d, y)

    # 3. Grid
    x_min, x_max = X_2d[:, 0].min() - 5, X_2d[:, 0].max() + 5
    y_min, y_max = X_2d[:, 1].min() - 5, X_2d[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 0.5))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 4. Plotting Setup
    plt.figure(figsize=(12, 10))
    
    # 设置纯白背景
    plt.gca().set_facecolor('white')
    plt.gcf().set_facecolor('white')
    
    # 绘制决策边界线 (淡灰色，不填充颜色，只留线)
    plt.contour(xx, yy, Z, colors='lightgray', linewidths=1.5, alpha=0.7, linestyles='--')
    
    # 绘制数据点
    # 使用 bright 调色板，确保 Happy/Tense/Sad/Relaxed 颜色区分明显
    # 这里的 palette 顺序将对应 y (0,1,2,3) 的顺序
    # 假设 0=Happy, 1=Tense, 2=Sad, 3=Relaxed
    palette = sns.color_palette("bright", n_colors=len(class_names))
    
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1],
        hue=y,
        palette=palette,
        style=y,
        s=100, # 点稍微大一点
        alpha=0.85,
        edgecolor='w',
        linewidth=0.8
    )
    
    # 处理图例
    handles, _ = plt.gca().get_legend_handles_labels()
    # 确保只取前N个handle (避免 SVM 边界产生的额外图例)
    if len(handles) > len(class_names): 
        handles = handles[:len(class_names)]
    
    # 使用语义化的 class_names (Happy, Tense, etc.)
    plt.legend(handles, class_names, title='Emotion Category', fontsize=12, title_fontsize=14, loc='best', frameon=True, framealpha=0.9)
    
    plt.title(title, fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("Latent Dimension 1 (t-SNE)", fontsize=12)
    plt.ylabel("Latent Dimension 2 (t-SNE)", fontsize=12)
    
    # 去掉边框刻度线，保持干净
    plt.grid(False)
    plt.tight_layout()
    
    print("✅ 图表已生成！背景已设为纯白。请在弹出的窗口中点击保存按钮。")
    plt.show() # 阻塞式弹窗

# ═══════════＝  6. Experiment Execution ＝═══════════
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
        for mel, emb, genre, audio, labels in train_loader:
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
            for mel, emb, genre, audio, labels in val_loader:
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
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{EPOCHS} | Val Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            wait_for_es = 0
        else:
            wait_for_es += 1
            if USE_EARLY_STOPPING and wait_for_es >= ES_PATIENCE:
                print(f"--- Early stopping at epoch {epoch} ---"); break
    
    print(f"Run {run_num} Eval...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH)); model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for mel, emb, genre, audio, labels in test_loader:
            mel, emb, genre, audio = mel.to(device), emb.to(device), genre.to(device), audio.to(device)
            outputs = model(mel, emb, genre, audio)
            y_pred.extend(outputs.argmax(1).cpu().numpy()); y_true.extend(labels.cpu().numpy())
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0)
    return macro_f1, class_f1, history

# ═══════════＝  7. Main Execution ＝═══════════
if __name__ == '__main__':
    try:
        df_labels = pd.read_csv(LABELS_CSV); df_genres = pd.read_csv(GENRES_CSV); df_audio_features = pd.read_csv(AUDIO_FEATURES_CSV)
    except FileNotFoundError as e: print(f"❌ Error: File not found {e.filename}."); sys.exit()
    
    df = pd.merge(df_labels, df_genres, on="song_id"); df = pd.merge(df, df_audio_features, on="song_id")
    df['song_id'] = pd.to_numeric(df['song_id'], errors='coerce'); df.dropna(subset=['song_id'], inplace=True); df['song_id'] = df['song_id'].astype(int).astype(str)
    df.dropna(subset=['normalized_genres'], inplace=True); df['normalized_genres'] = df['normalized_genres'].astype(str)
    
    missing_mels = {sid for sid in df['song_id'] if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    missing_muqs = {sid for sid in df['song_id'] if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
    all_missing = missing_mels.union(missing_muqs)
    if all_missing: df = df[~df['song_id'].isin(all_missing)]
    if df.empty: print("\n❌ FATAL ERROR: No valid data."); sys.exit()
    
    feature_cols = [col for col in df_audio_features.columns if col != 'song_id']
    
    # ─── Label Encoding & Mapping Logic (Updated!) ───
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["Quadrant"])
    
    # 获取原始的类名顺序 (e.g., ['Q1', 'Q2', 'Q3', 'Q4'])
    original_quadrants = list(le.classes_) 
    
    # 建立映射字典
    quadrant_map = {
        'Q1': 'Happy',
        'Q2': 'Tense',
        'Q3': 'Sad',
        'Q4': 'Relaxed'
    }
    
    # 按照原始编码顺序生成新的语义化类名列表
    CLASS_NAMES = [quadrant_map[q] for q in original_quadrants]
    print(f"▶ Mapped Classes: {original_quadrants} -> {CLASS_NAMES}")

    all_genres_list = sorted({g for g in df.normalized_genres.str.cat(sep=';').split(';') if g})
    genre_to_idx = {g: i for i, g in enumerate(all_genres_list)}
    N_GENRES = len(all_genres_list)
    EMB_DIM = 1024 
    print(f"\n▶ Data loaded. {len(df)} valid samples.")
    
    class_weights = compute_class_weight('balanced', classes=np.unique(df["label_idx"]), y=df["label_idx"].values)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    all_macro_f1s, all_class_f1s = [], []
    
    # ─── 1. Training Loop ───
    for i in range(1, NUM_RUNS + 1):
        macro_f1, class_f1, history = run_experiment(
            run_num=i, full_df=df, feature_cols=feature_cols, emb_dim=EMB_DIM,
            n_genres=N_GENRES, genre_to_idx=genre_to_idx, 
            class_names=CLASS_NAMES, class_weights=class_weights
        )
        all_macro_f1s.append(macro_f1); all_class_f1s.append(class_f1)
    
    print(f"\nBest Macro F1: {np.max(all_macro_f1s):.4f}")

    # ─── 2. Visualization Stage (Optimized) ───
    print("\n" + "═"*15 + " GENERATING DECISION BOUNDARY " + "═"*15)
    
    best_model = QuadFusionNet(EMB_DIM, N_GENRES, len(feature_cols), len(CLASS_NAMES)).to(device)
    if os.path.exists(BEST_MODEL_PATH):
        best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
        print("✅ Best model loaded.")
    else:
        print("⚠️ Model file not found, using last run weights.")
        
    # 🔴 Fix: Use np.arange(len(df)) to avoid IndexError
    vis_dataset = QuadFusionDataset(df, np.arange(len(df)), MEL_FEAT_DIR, MUQ_EMB_DIR, feature_cols, N_GENRES, genre_to_idx)
    vis_loader = DataLoader(vis_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
    
    # Extract and Plot
    X_learned, y_labels = extract_learned_features(best_model, vis_loader, device)
    
    plot_decision_boundary(
        X_learned, 
        y_labels, 
        CLASS_NAMES, 
        title="Learned Feature Space & Decision Boundaries"
    )
    
    print("\n▶ ALL DONE.")