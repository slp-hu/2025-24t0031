import pandas as pd
import numpy as np
import os

# -----------------------------------------------------------------------------
# 1. 定义文件和列名
# -----------------------------------------------------------------------------
# (请确保此路径与您上一个脚本的输出路径一致)
INPUT_CSV = r'C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon2_with_clusters.csv'
OUTPUT_DIR = r'C:\Users\YAO\Desktop\genre ml\paper\matrices' # 用于存储矩阵的文件夹

# 确保输出文件夹存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 列名 (!! 现在 'primary_emotion' 必须在 CSV 中 !!)
GENRE_COLUMN = 'major_genre_name_en'
PRIMARY_EMOTION_COLUMN = 'primary_emotion' # (已在 CSV 中)
CLUSTER_NUMERIC_ID_COLUMN = 'cluster_numeric_id' 

# -----------------------------------------------------------------------------
# 2. 加载数据
# -----------------------------------------------------------------------------
print(f"Loading data from {INPUT_CSV}...")
try:
    df = pd.read_csv(INPUT_CSV)
except FileNotFoundError:
    print(f"错误: 找不到文件 {INPUT_CSV}。请检查路径是否正确。")
    exit()
except KeyError as e:
    print(f"错误: 您的 CSV 文件中缺少必需的列: {e}")
    print("请确保您已重新运行第一个脚本，并保留了 'primary_emotion' 列。")
    exit()

print(f"Data loaded. Shape: {df.shape}")

# -----------------------------------------------------------------------------
# 3. 创建所有“层级”的 映射 (Mappings)
# -----------------------------------------------------------------------------
print("Step 1: Creating Mappings for all levels...")

# --- 歌曲 (Songs) ---
# 歌曲 ID 就是它们在 DataFrame 中的行索引 (0, 1, 2, ...)
n_songs = len(df)
song_ids = np.arange(n_songs) 

# --- 簇 (Clusters) ---
# 维度 = 最大的ID + 1
n_clusters = df[CLUSTER_NUMERIC_ID_COLUMN].max() + 1

# --- 情感子流派 (EmoGenres) ---
# 我们使用 '|' 作为安全的分隔符
# (!! 这就是为什么我们必须保留 'primary_emotion' !!)
df['emogenre_name'] = df[PRIMARY_EMOTION_COLUMN] + '|' + df[GENRE_COLUMN]
unique_emogenres = df['emogenre_name'].unique()
n_emogenres = len(unique_emogenres)
# 创建 map: {'Romantic|POP': 0, 'Healing|POP': 1, ...}
emogenre_id_map = {name: i for i, name in enumerate(unique_emogenres)}

# --- 流派 (Genres) ---
unique_genres = df[GENRE_COLUMN].unique()
n_genres = len(unique_genres)
# 创建 map: {'POP': 0, 'Rock/Metal': 1, ...}
genre_id_map = {name: i for i, name in enumerate(unique_genres)}

print(f"  > Songs (n_songs):    {n_songs}")
print(f"  > Clusters (n_clusters): {n_clusters}")
print(f"  > EmoGenres (n_emogenres): {n_emogenres}")
print(f"  > Genres (n_genres):    {n_genres}")

# -----------------------------------------------------------------------------
# 4. 初始化 (全零) 矩阵
# -----------------------------------------------------------------------------
print("\nStep 2: Initializing Zero Matrices...")

# V¹ (簇 x 歌曲)
V1 = np.zeros((n_clusters, n_songs), dtype=np.int8)
# V² (情感子流派 x 簇)
V2 = np.zeros((n_emogenres, n_clusters), dtype=np.int8)
# V³ (流派 x 情感子流派)
V3 = np.zeros((n_genres, n_emogenres), dtype=np.int8)

print(f"  > V1 shape (簇 x 歌曲):        {V1.shape}")
print(f"  > V2 shape (情感子流派 x 簇):  {V2.shape}")
print(f"  > V3 shape (流派 x 情感子流派): {V3.shape}")

# -----------------------------------------------------------------------------
# 5. 填充矩阵 (核心)
# -----------------------------------------------------------------------------

# --- 填充 V¹ (簇 x 歌曲) ---
print("\nStep 3: Populating V1 (簇 x 歌曲)...")
# V1[i, j] = 1  如果 歌曲 j 属于 簇 i
cluster_ids = df[CLUSTER_NUMERIC_ID_COLUMN].values
V1[cluster_ids, song_ids] = 1
print(f"  > V1 populated. Total '1's: {np.sum(V1)}")

# --- 填充 V² (情感子流派 x 簇) ---
print("Step 4: Populating V2 (情感子流派 x 簇)...")
# V2[k, i] = 1  如果 簇 i 属于 情感子流派 k
# 我们需要 簇ID 和 情感子流派名称 之间的“唯一”关系
unique_cluster_df = df.drop_duplicates(subset=[CLUSTER_NUMERIC_ID_COLUMN])

for _, row in unique_cluster_df.iterrows():
    i = row[CLUSTER_NUMERIC_ID_COLUMN] # 簇 ID (数字)
    emogenre_name = row['emogenre_name'] # 'Romantic|POP' (字符串)
    
    if emogenre_name in emogenre_id_map:
        k = emogenre_id_map[emogenre_name] # 情感子流派 ID (数字)
        V2[k, i] = 1
print(f"  > V2 populated. Total '1's: {np.sum(V2)}")


# --- 填充 V³ (流派 x 情感子流派) ---
print("Step 5: Populating V3 (流派 x 情感子流派)...")
# V3[l, k] = 1  如果 情感子流派 k 属于 流派 l
for emogenre_name, k in emogenre_id_map.items():
    parts = emogenre_name.split('|')
    if len(parts) == 2:
        genre_name = parts[1]
        if genre_name in genre_id_map:
            l = genre_id_map[genre_name] # 流派 ID (数字)
            V3[l, k] = 1
print(f"  > V3 populated. Total '1's: {np.sum(V3)}")

# -----------------------------------------------------------------------------
# 6. 保存矩阵
# -----------------------------------------------------------------------------
print("\nStep 6: Saving Matrices to .npy files...")

np.save(os.path.join(OUTPUT_DIR, 'V1.npy'), V1)
np.save(os.path.join(OUTPUT_DIR, 'V2.npy'), V2)
np.save(os.path.join(OUTPUT_DIR, 'V3.npy'), V3)

# 同时保存“映射字典”，我们下一步训练 V⁴ 时会需要它们
np.save(os.path.join(OUTPUT_DIR, 'genre_id_map.npy'), genre_id_map)
np.save(os.path.join(OUTPUT_DIR, 'emogenre_id_map.npy'), emogenre_id_map)

print(f"\n--- 成功！ ---")
print(f"V1, V2, V3 矩阵已保存到: {OUTPUT_DIR}")
print("下一步：加载这些矩阵，并训练 V⁴。")