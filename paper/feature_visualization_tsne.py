import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

# ═══════════＝ 1. 你的配置信息 (保持不变) ＝═══════════
MEL_FEAT_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\mel"
MUQ_EMB_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\muq"
AUDIO_FEATURES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\audio_features_normalized.csv"
GENRES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\processed_genres.csv"
LABELS_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\quadrants_from_dynamic.csv"

# ═══════════＝ 2. 降维和绘图函数 (保持不变) ＝═══════════
def plot_visualization(X, y, title, class_names, method='tsne', n_components=2):
    print(f"正在处理 {title}...")
    if X.shape[0] == 0: print(f"跳过 {title}: 没有数据。"); return
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, 
                       perplexity=30, max_iter=1000, init='pca', 
                       learning_rate='auto')
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
        
    X_reduced = reducer.fit_transform(X_scaled)
    
    plot_df = pd.DataFrame()
    plot_df['dim_1'] = X_reduced[:, 0]
    plot_df['dim_2'] = X_reduced[:, 1]
    plot_df['emotion'] = y
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='dim_1', y='dim_2',
        hue='emotion',
        hue_order=class_names,
        palette=sns.color_palette("deep", n_colors=len(class_names)),
        data=plot_df,
        legend="full",
        alpha=0.8 
    )
    plt.title(title, fontsize=16)
    
    if method == 'pca':
        explained_variance = reducer.explained_variance_ratio_
        plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.2f}%)")
        plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.2f}%)")
    
    plt.show()

# ═══════════＝ 3. 特征提取函数 (已修正) ＝═══════════
def extract_features_from_df(df, feature_cols, mel_dir, muq_dir):
    """
    从合并的 DataFrame 中迭代并加载所有特征。
    """
    
    # --- ⬇️ 这里是修正的地方 ⬇️ ---
    all_mels, all_muqs, all_audios, all_label_names = [], [], [], []
    # --- ⬆️ 修正完毕 ⬆️ ---
    
    print(f"正在从 {len(df)} 条(已下采样)记录中提取特征...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="加载特征文件"):
        song_id = str(row.song_id); label_name = row.Quadrant
        try:
            mel_path = os.path.join(mel_dir, f"{song_id}.npy")
            mel_spec = np.load(mel_path); mel_feat = np.mean(mel_spec, axis=1) 
            
            muq_path = os.path.join(muq_dir, f"{song_id}.npy")
            muq_feat = np.load(muq_path)
            if muq_feat.ndim > 1: muq_feat = muq_feat.squeeze() 

            audio_feat = row[feature_cols].values.astype(np.float32)
            
            all_mels.append(mel_feat); all_muqs.append(muq_feat); all_audios.append(audio_feat)
            all_label_names.append(label_name)
        except Exception as e:
            print(f"处理 {song_id} 时出错 (已跳过): {e}")

    print(f"\n成功加载 {len(all_label_names)} 个数据点。")
    X_mel = np.array(all_mels); X_muq = np.array(all_muqs)
    X_audio = np.array(all_audios); y_labels = np.array(all_label_names)
    return X_mel, X_muq, X_audio, y_labels

# ═══════════＝ 4. 主执行函数 (保持不变) ＝═══════════
if __name__ == '__main__':
    print("--- 启动特征可视化实验 (平衡 + 下采样版本) ---")
    
    # --- 步骤 1: 加载和准备数据 ---
    try:
        df_labels = pd.read_csv(LABELS_CSV); df_genres = pd.read_csv(GENRES_CSV); df_audio_features = pd.read_csv(AUDIO_FEATURES_CSV)
    except FileNotFoundError as e: print(f"❌ 致命错误: 找不到 CSV 文件 {e.filename}。"); sys.exit()
    
    df = pd.merge(df_labels, df_genres, on="song_id"); df = pd.merge(df, df_audio_features, on="song_id")
    df['song_id'] = pd.to_numeric(df['song_id'], errors='coerce'); df.dropna(subset=['song_id'], inplace=True); df['song_id'] = df['song_id'].astype(int).astype(str)
    
    feature_cols = [col for col in df_audio_features.columns if col != 'song_id']

    label_mapping = {'Q1': 'Happy', 'Q2': 'Tense', 'Q3': 'Sad', 'Q4': 'Relaxed'}
    NEW_CLASS_NAMES_ORDER = ['Happy', 'Tense', 'Sad', 'Relaxed']
    df['Quadrant'] = df['Quadrant'].map(label_mapping); df.dropna(subset=['Quadrant'], inplace=True)
    
    le = LabelEncoder(); df["label_idx"] = le.fit_transform(df["Quadrant"]); CLASS_NAMES = NEW_CLASS_NAMES_ORDER
    
    missing_mels = {sid for sid in df['song_id'] if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    missing_muqs = {sid for sid in df['song_id'] if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
    all_missing = missing_mels.union(missing_muqs)
    if all_missing: df = df[~df['song_id'].isin(all_missing)]
    if df.empty: print("\n❌ 致命错误: 没有可用的有效数据。"); sys.exit()

    print(f"数据准备完成。过滤前总计 {len(df)} 条有效记录。")

    # --- 新增步骤: 下采样 (Downsampling) ---
    print("\n--- 正在检查类别不平衡 ---")
    print("原始样本数量：")
    class_counts = df['Quadrant'].value_counts()
    print(class_counts)
    
    min_samples = class_counts.min()
    print(f"检测到最少样本数为: {min_samples}。将对所有类别下采样至此数量。")

    df_balanced = df.groupby('Quadrant').sample(n=min_samples, random_state=42)
    
    print("\n下采样后样本数量：")
    print(df_balanced['Quadrant'].value_counts())
    print(f"下采样后总样本数: {len(df_balanced)}")

    
    # --- 步骤 2: 提取所有特征 ---
    X_mel, X_muq, X_audio, y_labels = extract_features_from_df(
        df=df_balanced, # <--- 使用平衡后的数据
        feature_cols=feature_cols,
        mel_dir=MEL_FEAT_DIR,
        muq_dir=MUQ_EMB_DIR
    )
    
    print("\n原始特征维度检查 (下采样后):")
    print(f"  Mel (X_mel):    {X_mel.shape}")
    print(f"  MUQ (X_muq):    {X_muq.shape}")
    print(f"  Audio (X_audio): {X_audio.shape}")

    # --- 步骤 3: 绘制单独图表 ---
    VIS_METHOD = 'tsne' 
    N_DIMS = 2
    
    print(f"\n--- 正在运行 步骤 1-3: 单独特征图表 ({VIS_METHOD.upper()}) ---")
    plot_visualization(X_mel, y_labels, 
                       f"Mel Features ({VIS_METHOD.upper()}) [Downsampled]", 
                       CLASS_NAMES, method=VIS_METHOD, n_components=N_DIMS)

    plot_visualization(X_muq, y_labels, 
                       f"MUQ Features ({VIS_METHOD.upper()}) [Downsampled]", 
                       CLASS_NAMES, method=VIS_METHOD, n_components=N_DIMS)

    plot_visualization(X_audio, y_labels, 
                       f"Audio Features ({VIS_METHOD.upper()}) [Downsampled]", 
                       CLASS_NAMES, method=VIS_METHOD, n_components=N_DIMS)

    # --- 步骤 4: 创建平衡的合并特征 (逻辑不变) ---
    print("\n--- 正在运行 步骤 4: 平衡的合并特征图表 ---")
    BALANCED_DIMS = 16 
    print(f"正在将 Mel 和 MUQ 压缩到 {BALANCED_DIMS} 维...")

    scaler_mel = StandardScaler(); X_mel_scaled = scaler_mel.fit_transform(X_mel)
    pca_mel = PCA(n_components=BALANCED_DIMS, random_state=42); X_mel_balanced = pca_mel.fit_transform(X_mel_scaled)
    
    scaler_muq = StandardScaler(); X_muq_scaled = scaler_muq.fit_transform(X_muq)
    pca_muq = PCA(n_components=BALANCED_DIMS, random_state=42); X_muq_balanced = pca_muq.fit_transform(X_muq_scaled)

    scaler_audio = StandardScaler(); X_audio_scaled = scaler_audio.fit_transform(X_audio)
    X_audio_balanced = X_audio_scaled # 直接使用 8 维

    X_combined_balanced = np.concatenate((X_mel_balanced, X_muq_balanced, X_audio_balanced), axis=1)

    final_balanced_shape = X_combined_balanced.shape[1]
    print(f"平衡合并后的特征维度: {X_combined_balanced.shape}")
    print(f"  (Mel={X_mel_balanced.shape[1]}维, MUQ={X_muq_balanced.shape[1]}维, Audio={X_audio_balanced.shape[1]}维)")

    plot_visualization(X_combined_balanced, y_labels, 
                       f"Balanced Combined Features (Total {final_balanced_shape}D) ({VIS_METHOD.upper()}) [Downsampled]", 
                       CLASS_NAMES, method=VIS_METHOD, n_components=N_DIMS)

    print("\n--- 实验完成 ---")