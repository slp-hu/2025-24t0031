import pandas as pd

# 1. 读取推理结果 CSV
df = pd.read_csv("/mnt/data/inference_results_78.csv")

# 2. 定义一个函数，根据多标签流派（genres 列）计算目标流派（如 "rock"）的权重
def compute_genre_weight(genre_str: str, target_genre: str) -> float:
    normalized = genre_str.replace(";", ",")
    labels = [g.strip().lower() for g in normalized.split(",") if g.strip()]
    try:
        idx = labels.index(target_genre.lower())
        return 1.0 / (idx + 1)
    except ValueError:
        return 0.0

# 3. 在 DataFrame 中添加一列 'genre_weight'
df["genre_weight"] = df["genres"].apply(lambda s: compute_genre_weight(s, target_genre="rock"))

# 4. 定义一个函数，根据行数据和目标情感列计算综合得分
def compute_emotion_score(row: pd.Series, emotion_col: str) -> float:
    w_genre = row["genre_weight"]
    p_target = row[emotion_col]
    p_fuzzy = row["p_Ambiguous"]
    return w_genre * p_target * (1.0 - p_fuzzy)

# 5. 为每个基本情感列计算得分，并把得分存入新的列
emotion_cols = ["p_Happy", "p_Sad", "p_Angry", "p_Relaxed"]
for emo in emotion_cols:
    score_col = emo.replace("p_", "score_")
    df[score_col] = df.apply(lambda r: compute_emotion_score(r, emotion_col=emo), axis=1)

# 6. 根据得分对每种情感分别排序，生成歌单（取 top_k 首）并计算 precision
top_k = 20
precision_results = {}

for emo in emotion_cols:
    score_col = emo.replace("p_", "score_")
    sorted_df = df.sort_values(by=score_col, ascending=False)
    top_songs = sorted_df.head(top_k)[
        ["song_id", "genres", "label", emo, "p_Ambiguous", "genre_weight", score_col]
    ]
    # 计算 precision，即 top_songs 中 label 等于该情感的数量 / top_k
    target_label = emo.replace("p_", "")  # 如 "Happy"
    correct_count = (top_songs["label"] == target_label).sum()
    precision = correct_count / top_k
    precision_results[target_label] = precision
    
    # 保存歌单到 CSV
    emotion_name = target_label.lower()
    out_fname = f"/mnt/data/playlist_{emotion_name}.csv"
    top_songs.to_csv(out_fname, index=False)

# 显示 precision 结果
precision_df = pd.DataFrame([
    {"Emotion": emo, "Precision@20": precision_results[emo]}
    for emo in precision_results
])



