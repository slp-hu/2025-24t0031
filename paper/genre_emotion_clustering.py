import pandas as pd
import numpy as np
import ast
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# -----------------------------------------------------------------------------
# 1. 定义您的核心参数 (不变)
# -----------------------------------------------------------------------------
FILE_PATH = r'C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon2.csv'
GENRE_COLUMN = 'major_genre_name_en'
EMOTION_COLUMN = 'tag'

EMOTION_COLUMNS = [
    'Romantic', 'Happiness', 'Excitement', 'Healing', 'Sadness', 'Quiet',
    'Nostalgia', 'Loneliness', 'Touching', 'Missing', 'Fresh', 'Relaxation'
]

MIN_GENRE_SIZE = 150
MIN_SONGS_TO_CLUSTER = 50
MAX_K_TO_TRY = 5
OUTPUT_FILE = r'C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon2_with_clusters.csv'

# -----------------------------------------------------------------------------
# 2. 辅助函数 (不变)
# -----------------------------------------------------------------------------

def find_primary_emotion(tag_string):
    try:
        tag_dict = ast.literal_eval(tag_string)
        if not tag_dict: return 'No_Emotion'
    except (ValueError, SyntaxError):
        return 'No_Emotion'
    
    max_score = 0
    try: max_score = max(tag_dict.values())
    except ValueError: return 'No_Emotion'
    
    if max_score == 0: return 'No_Emotion'
    top_emotions = [key for key, value in tag_dict.items() if value == max_score]
    for emotion in EMOTION_COLUMNS:
        if emotion in top_emotions:
            return emotion
    return 'No_Emotion'

def parse_emotion_vector(tag_string):
    try: tag_dict = ast.literal_eval(tag_string)
    except (ValueError, SyntaxError): return [0] * len(EMOTION_COLUMNS)
    return [tag_dict.get(emotion, 0) for emotion in EMOTION_COLUMNS]

def find_optimal_k(X, max_k=5):
    best_k = -1
    best_score = -1.0
    for k in range(2, max_k + 1):
        if len(X) <= k: break
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            if len(np.unique(labels)) < 2: continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except ValueError: continue
    if best_k == -1:
        return 2 if len(X) >= MIN_SONGS_TO_CLUSTER else 1
    return best_k

# -----------------------------------------------------------------------------
# 3. 主执行流程 (不变)
# -----------------------------------------------------------------------------
print("--- 启动层级聚类脚本 (已修正) ---")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")

print(f"Loading data from {FILE_PATH}...")
df = pd.read_csv(FILE_PATH)
print(f"Original data shape: {df.shape}")
genre_counts = df[GENRE_COLUMN].value_counts()
major_genres = genre_counts[genre_counts >= MIN_GENRE_SIZE].index
df = df[df[GENRE_COLUMN].isin(major_genres)].copy()
print(f"Data shape after filtering small genres: {df.shape}")

print("Parsing emotion tags and finding primary emotion...")
df['emotion_vector'] = df[EMOTION_COLUMN].apply(parse_emotion_vector)
df['primary_emotion'] = df[EMOTION_COLUMN].apply(find_primary_emotion)

print("\nStarting clustering process (Corrected Logic)...")
cluster_results = []

for genre_name in major_genres:
    for emotion_name in EMOTION_COLUMNS:
        combo_name = f"{emotion_name}_{genre_name}"
        emo_genre_df = df[
            (df[GENRE_COLUMN] == genre_name) & 
            (df['primary_emotion'] == emotion_name)
        ]
        song_count = len(emo_genre_df)
        if song_count == 0: continue

        if song_count < MIN_SONGS_TO_CLUSTER:
            cluster_id = f"{combo_name}_Cluster_0"
            if song_count > 0:
                print(f"  > '{combo_name}' (Songs: {song_count}) -> Too small. Treating as ONE cluster: {cluster_id}")
            for song_index in emo_genre_df.index:
                cluster_results.append({'song_index': song_index, 'cluster_id': cluster_id, 'k_used': 1})
        else:
            print(f"  > '{combo_name}' (Songs: {song_count}) -> Finding optimal k...")
            X = np.array(emo_genre_df['emotion_vector'].tolist())
            optimal_k = find_optimal_k(X, max_k=MAX_K_TO_TRY)
            print(f"  > Optimal k for '{combo_name}' is {optimal_k}.")
            
            cluster_id_base = f"{combo_name}_Cluster_"
            if optimal_k < 2:
                for song_index in emo_genre_df.index:
                    cluster_results.append({'song_index': song_index, 'cluster_id': f"{cluster_id_base}0", 'k_used': 1})
            else:
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                for i, song_index in enumerate(emo_genre_df.index):
                    cluster_results.append({'song_index': song_index, 'cluster_id': f"{cluster_id_base}{labels[i]}", 'k_used': optimal_k})

# -----------------------------------------------------------------------------
# 4. 整理结果 (已修改)
# -----------------------------------------------------------------------------
print("\nClustering finished. Merging results...")

cluster_df = pd.DataFrame(cluster_results)
final_df = df.merge(cluster_df, left_index=True, right_on='song_index', how='left')

# --- 步骤 5: 保存和总结 (已修改) ---
final_df['cluster_id'] = final_df['cluster_id'].fillna('No_Cluster')
final_df['k_used'] = final_df['k_used'].fillna(0)

# !! (新) 添加独一无二的数字ID (您的建议) !!
print("Assigning unique numeric IDs to clusters...")
# .astype('category').cat.codes 是 pandas 的一个标准功能
# 它会为每个“唯一”的字符串分配一个从 0 开始的整数 ID
final_df['cluster_numeric_id'] = final_df['cluster_id'].astype('category').cat.codes

# 打印新 ID 和旧 ID 的对应关系 (前 10 个)
print("\n--- 字符串 ID 与 数字 ID 映射示例 ---")
print(
    final_df[['cluster_id', 'cluster_numeric_id']]
    .drop_duplicates()
    .sort_values(by='cluster_numeric_id')
    .head(10)
)
print("...")

# 清理不再需要的中间列
# (已修正，删除 'primary_emotion' 而不是 'tag_dict')
final_df = final_df.drop(columns=['emotion_vector', 'song_index'])

print(f"\n--- 最终 DataFrame (带数字 ID) ---")
print(f"Final shape: {final_df.shape}")
# 打印包含新 ID 的列
print(final_df[[GENRE_COLUMN, EMOTION_COLUMN, 'cluster_id', 'cluster_numeric_id']].head(10))

print("\n--- 簇分布情况 (Top 20) ---")
print(final_df['cluster_id'].value_counts().head(20))

final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n--- 成功！ ---")
print(f"已将带有 'cluster_id' 和 'cluster_numeric_id' 的结果保存到: {OUTPUT_FILE}")
print("您现在可以使用 'cluster_numeric_id' 列来构建您的 V¹ 矩阵。")