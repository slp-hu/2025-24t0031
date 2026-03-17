import pandas as pd
import numpy as np
import ast
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# --- 1. 配置参数 ---
# 输入文件的路径
INPUT_FILE_PATH = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_filtered_audioexist.csv"

# 【重要】定义你想要保存的新文件名
OUTPUT_FILE_PATH = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_final_labeled_dataset.csv"

# 聚类参数 (使用我们最终确定的值)
GLOBAL_EPS = 2.5
GLOBAL_MIN_SAMPLES = 24
SUB_EPS = 1.38
SUB_MIN_SAMPLES = 24

# --- 2. 加载和预处理数据 ---
print("--- 步骤 1: 正在加载和预处理数据... ---")
try:
    df = pd.read_csv(INPUT_FILE_PATH)
    print(f"成功加载 {len(df)} 条记录。")
except FileNotFoundError:
    print(f"错误：输入文件未在以下路径找到 {INPUT_FILE_PATH}")
    exit()

def parse_tag(tag_str):
    try:
        return ast.literal_eval(tag_str)
    except (ValueError, SyntaxError):
        return {}

df['tag_dict'] = df['tag'].apply(parse_tag)
tags_df = pd.json_normalize(df['tag_dict'])
emotion_columns = ['Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 
                   'Quiet', 'Happiness', 'Loneliness', 'Touching', 'Missing', 
                   'Fresh', 'Relaxation']
for col in emotion_columns:
    if col not in tags_df.columns:
        tags_df[col] = 0
tags_df = tags_df[emotion_columns].fillna(0)
X = tags_df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("数据预处理完成。")

# --- 3. 执行两步聚类流程 ---
print("\n--- 步骤 2: 正在执行两步聚类... ---")
# 第一次全局聚类
dbscan_global = DBSCAN(eps=GLOBAL_EPS, min_samples=GLOBAL_MIN_SAMPLES)
df['cluster'] = dbscan_global.fit_predict(X_scaled)
# 第二次分层聚类
cluster_0_indices = df.index[df['cluster'] == 0]
X_cluster_0_scaled = X_scaled[cluster_0_indices]
dbscan_sub = DBSCAN(eps=SUB_EPS, min_samples=SUB_MIN_SAMPLES)
sub_clusters = dbscan_sub.fit_predict(X_cluster_0_scaled)
print("两步聚类完成。")

# --- 4. 整合聚类结果并进行语义合并 ---
print("\n--- 步骤 3: 正在合并标签... ---")
# 整合两次聚类的结果到 'final_cluster' 列
df['final_cluster'] = df['cluster'].astype(str)
# 注意：df.loc需要对齐索引，所以这里的操作是安全的
df.loc[cluster_0_indices, 'final_cluster'] = ['0_' + str(sc) for sc in sub_clusters]

# 定义最终的映射字典
cluster_mapping = {
    '1': 'Lonely', '2': 'Lonely_Missing', '3': 'Sad', '4': 'Sad_Lonely',
    '5': 'Sad_Missing', '6': 'Missing', '7': 'Lonely', '8': 'Sad',
    '-1': 'Ambiguous', '0_-1': 'Ambiguous',
    '0_0': 'Exciting', '0_2': 'Joyful', '0_3': 'Joyful', '0_7': 'Joyful',
    '0_14': 'Joyful', '0_27': 'Joyful', '0_29': 'Joyful', '0_30': 'Joyful',
    '0_32': 'Joyful', '0_34': 'Joyful',
    '0_6': 'Romantic', '0_8': 'Romantic', '0_18': 'Romantic', '0_19': 'Romantic',
    '0_21': 'Romantic', '0_22': 'Romantic', '0_25': 'Romantic', '0_31': 'Romantic',
    '0_9': 'Romantic',
    '0_4': 'Nostalgic', '0_13': 'Nostalgic', '0_16': 'Nostalgic', '0_33': 'Nostalgic',
    '0_10': 'Uplifting_Fresh', '0_11': 'Uplifting_Fresh', '0_15': 'Uplifting_Fresh',
    '0_28': 'Uplifting_Fresh',
    '0_1': 'Peaceful', '0_17': 'Peaceful', '0_23': 'Peaceful', '0_24': 'Peaceful',
    '0_26': 'Peaceful',
    '0_5': 'Touching', '0_12': 'Touching', '0_20': 'Touching'
}

# 应用映射，创建最终的 'final_label' 列
df['final_label'] = df['final_cluster'].map(cluster_mapping)
print("语义合并完成，已生成 'final_label' 列。")

# --- 5. 清理并保存最终数据集 ---
print("\n--- 步骤 4: 正在保存最终的带标签数据集... ---")

# 为了让最终文件更整洁，我们可以删除中间过程产生的列
# 如果你想保留它们用于追溯，可以注释掉下面这行
df_to_save = df.drop(columns=['tag_dict', 'cluster', 'final_cluster'])

try:
    # index=False 避免将DataFrame的索引写入文件
    # encoding='utf-8-sig' 确保中文在Excel中正常显示
    df_to_save.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
    print("\n" + "="*50)
    print("🎉 成功！")
    print(f"已将带有 'final_label' 标签的新数据集保存至:")
    print(OUTPUT_FILE_PATH)
    print("="*50)
except Exception as e:
    print(f"\n错误：无法保存文件，原因: {e}")