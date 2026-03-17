import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA # PCA
from tqdm import tqdm

# ═══════════＝ 1. 你的配置信息 (保持不变) ＝═══════════
MEL_FEAT_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\mel"
MUQ_EMB_DIR = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\muq"
AUDIO_FEATURES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\audio_features_normalized.csv"
GENRES_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\processed_genres.csv"
LABELS_CSV = "C:\\Users\\YAO\\Desktop\\genre ml\\paper\\quadrants_from_dynamic.csv"

# ═══════════＝ 2. 降维和绘图函数 (保持不变) ＝═══════════
def plot_visualization(X, y, title, class_names, method='tsne', n_components=2):
    """
    一个辅助函数，用于对特征 X 进行降维和可视化。
    """
    print(f"正在处理 {title}...")
    if X.shape[0] == 0:
        print(f"跳过 {title}: 没有数据。")
        return
        
    # 步骤 1: 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 步骤 2: 降维
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, 
                       perplexity=30, n_iter=1000, init='pca', 
                       learning_rate='auto')
    elif method == 'pca':
        # ⚠️ 已修改：现在会使用 PCA
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
        
    X_reduced = reducer.fit_transform(X_scaled)
    
    # 步骤 3: 准备绘图数据
    plot_df = pd.DataFrame()
    plot_df['dim_1'] = X_reduced[:, 0]
    plot_df['dim_2'] = X_reduced[:, 1]
    plot_df['emotion'] = y # y 应该是情感的字符串名称
    
    # 步骤 4: 绘制图表 (2D)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='dim_1', y='dim_2',
        hue='emotion',
        hue_order=class_names, # 确保颜色和图例顺序一致
        palette=sns.color_palette("deep", n_colors=len(class_names)),
        data=plot_df,
        legend="full",
        alpha=0.7
    )
    plt.title(title, fontsize=16)
    
    # 解释方差 (仅 PCA)
    if method == 'pca':
        explained_variance = reducer.explained_variance_ratio_
        plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.2f}%)")
        plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.2f}%)")
    
    plt.show()

# ═══════════＝ 3. 特征提取函数 (保持不变) ＝═══════════
def extract_features_from_df(df, feature_cols, mel_dir, muq_dir):
    """
    从合并的 DataFrame 中迭代并加载所有特征。
    """
    all_mels, all_muqs, all_audios, all_labels, all_label_names = [], [], [], [], []
    
    print(f"正在从 {len(df)} 条记录中提取特征...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="加载特征文件"):
        song_id = str(row.song_id)
        label_idx = row.label_idx
        # ⚠️ 这里会读取到新的名称 (e.g., 'Happy')
        label_name = row.Quadrant
        
        try:
            mel_path = os.path.join(mel_dir, f"{song_id}.npy")
            mel_spec = np.load(mel_path)
            mel_feat = np.mean(mel_spec, axis=1) 
            
            muq_path = os.path.join(muq_dir, f"{song_id}.npy")
            muq_feat = np.load(muq_path)
            if muq_feat.ndim > 1: muq_feat = muq_feat.squeeze() 

            audio_feat = row[feature_cols].values.astype(np.float32)
            
            all_mels.append(mel_feat)
            all_muqs.append(muq_feat)
            all_audios.append(audio_feat)
            all_labels.append(label_idx)
            all_label_names.append(label_name)
            
        except Exception as e:
            print(f"处理 {song_id} 时出错 (已跳过): {e}")

    print(f"\n成功加载 {len(all_label_names)} 个数据点。")
    
    X_mel = np.array(all_mels)
    X_muq = np.array(all_muqs)
    X_audio = np.array(all_audios)
    y_labels = np.array(all_label_names)
    
    return X_mel, X_muq, X_audio, y_labels

# ═══════════＝ 4. 主执行函数 (已修改) ＝═══════════
if __name__ == '__main__':
    print("--- 启动特征可视化实验 (PCA 版本) ---")
    
    try:
        df_labels = pd.read_csv(LABELS_CSV)
        df_genres = pd.read_csv(GENRES_CSV)
        df_audio_features = pd.read_csv(AUDIO_FEATURES_CSV)
    except FileNotFoundError as e:
        print(f"❌ 致命错误: 找不到 CSV 文件 {e.filename}。"); sys.exit()
    
    print("CSV 文件加载成功。")
    df = pd.merge(df_labels, df_genres, on="song_id")
    df = pd.merge(df, df_audio_features, on="song_id")
    
    df['song_id'] = pd.to_numeric(df['song_id'], errors='coerce')
    df.dropna(subset=['song_id'], inplace=True)
    df['song_id'] = df['song_id'].astype(int).astype(str)
    
    feature_cols = [col for col in df_audio_features.columns if col != 'song_id']

    # --- ⬇️ 第 1 处修改：标签重命名 ⬇️ ---
    label_mapping = {
        'Q1': 'Happy',
        'Q2': 'Tense',
        'Q3': 'Sad',
        'Q4': 'Relaxed'
    }
    # 这是图例中你想要的顺序
    NEW_CLASS_NAMES_ORDER = ['Happy', 'Tense', 'Sad', 'Relaxed']
    
    print(f"原始 Quadrant 标签: {df['Quadrant'].unique()}")
    
    # 应用映射，把 'Q1' 替换为 'Happy' 等
    df['Quadrant'] = df['Quadrant'].map(label_mapping)
    
    # 移除任何不在 Q1-Q4 中的行 (它们现在会是 NaT)
    df.dropna(subset=['Quadrant'], inplace=True)
    
    print(f"新的 'Quadrant' 标签: {df['Quadrant'].unique()}")
    # --- ⬆️ 结束修改 ⬆️ ---

    # 编码标签 (现在将对 'Happy', 'Tense' 等进行编码)
    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["Quadrant"])
    
    # 我们使用自定义的顺序，而不是 le.classes_
    CLASS_NAMES = NEW_CLASS_NAMES_ORDER
    print(f"使用 {len(CLASS_NAMES)} 个情感类别: {CLASS_NAMES}")
    
    # 检查缺失文件 (逻辑不变)
    missing_mels = {sid for sid in df['song_id'] if not os.path.exists(os.path.join(MEL_FEAT_DIR, f"{sid}.npy"))}
    missing_muqs = {sid for sid in df['song_id'] if not os.path.exists(os.path.join(MUQ_EMB_DIR, f"{sid}.npy"))}
    all_missing = missing_mels.union(missing_muqs)
    
    if all_missing:
        print(f"⚠️ 警告: 正在移除 {len(all_missing)} 条缺少特征文件的记录。")
        df = df[~df['song_id'].isin(all_missing)]
        
    if df.empty:
        print("\n❌ 致命错误: 没有可用的有效数据。"); sys.exit()

    print(f"数据准备完成。最终使用 {len(df)} 条有效记录。")
    
    # 步骤 2: 提取所有特征
    X_mel, X_muq, X_audio, y_labels = extract_features_from_df(
        df=df,
        feature_cols=feature_cols,
        mel_dir=MEL_FEAT_DIR,
        muq_dir=MUQ_EMB_DIR
    )
    
    if len(y_labels) == 0:
        print("没有数据被加载。退出。"); sys.exit()
        
    print("\n特征维度检查:")
    print(f"  Mel (X_mel):    {X_mel.shape}")
    print(f"  MUQ (X_muq):    {X_muq.shape}")
    print(f"  Audio (X_audio): {X_audio.shape}")
    print(f"  Labels (y_labels): {y_labels.shape}")

    # --- 教授要求的实验 ---
    
    # --- ⬇️ 第 2 处修改：方法切换 ⬇️ ---
    VIS_METHOD = 'pca' # 从 'tsne' 改为 'pca'
    # --- ⬆️ 结束修改 ⬆️ ---
    
    N_DIMS = 2
    
    # 步骤 1-3: 分别可视化三种特征
    print("\n--- 正在运行 步骤 1-3: 单独特征图表 ---")
    
    plot_visualization(X_mel, y_labels, 
                       f"Mel Features ({VIS_METHOD.upper()})", 
                       CLASS_NAMES, method=VIS_METHOD, n_components=N_DIMS)

    plot_visualization(X_muq, y_labels, 
                       f"MUQ Features ({VIS_METHOD.upper()})", 
                       CLASS_NAMES, method=VIS_METHOD, n_components=N_DIMS)

    plot_visualization(X_audio, y_labels, 
                       f"Audio Features ({VIS_METHOD.upper()})", 
                       CLASS_NAMES, method=VIS_METHOD, n_components=N_DIMS)

    # 步骤 4: 可视化合并后的特征
    print("\n--- 正在运行 步骤 4: 合并特征图表 ---")
    
    X_combined = np.concatenate((X_mel, X_muq, X_audio), axis=1)

    print(f"合并后特征维度: {X_combined.shape}")

    plot_visualization(X_combined, y_labels, 
                       f"Combined (Mel+MUQ+Audio) Features ({VIS_METHOD.upper()})", 
                       CLASS_NAMES, method=VIS_METHOD, n_components=N_DIMS)

    print("\n--- 实验完成 ---")