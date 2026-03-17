import pandas as pd

# 1. 读取推理结果 CSV
# 如果你的文件确实是制表符分隔，请加上 sep="\t"
# 例如：df = pd.read_csv("inference_results.csv", sep="\t")
df = pd.read_csv("inference_results_78.csv")

# 2. 定义一个函数，根据多标签流派（genres 列）计算目标流派（如 "rock"）的权重
def compute_genre_weight(genre_str: str, target_genre: str) -> float:
    """
    genre_str: 例："Soulrb;Blues;Rock"（分号分隔，且大小写混合）
    target_genre: 例如 "rock"（统一用小写）
    
    处理步骤：
    1) 把分号 ';' 全部替换为逗号 ','，得到类似 "Soulrb,Blues,Rock"。
    2) 用 ',' 拆分成列表，并逐一 strip()、lower()，即 ['soulrb','blues','rock']。
    3) 找到目标流派在列表中的索引 idx，返回权重 1/(idx+1)，否则返回 0。
    """
    # 先把所有分号换成逗号
    normalized = genre_str.replace(";", ",")
    # 拆分、去掉空白、全改小写
    labels = [g.strip().lower() for g in normalized.split(",") if g.strip()]
    try:
        idx = labels.index(target_genre.lower())
        return 1.0 / (idx + 1)
    except ValueError:
        return 0.0

# 3. 在 DataFrame 中添加一列 'genre_weight'
#    假设用户当前想听的流派是 "rock"，这里统一传小写 "rock"
df["genre_weight"] = df["genres"].apply(lambda s: compute_genre_weight(s, target_genre="rock"))

# 4. 定义一个函数，根据行数据和目标情感列计算综合得分
def compute_emotion_score(row: pd.Series, emotion_col: str) -> float:
    """
    row: DataFrame 中的一行
    emotion_col: 例如 "p_Happy"、"p_Sad"、"p_Angry"、"p_Relaxed"
    得分公式： score = genre_weight * p_target * (1 - p_Ambiguous)
    """
    w_genre = row["genre_weight"]
    p_target = row[emotion_col]
    p_fuzzy = row["p_Ambiguous"]
    return w_genre * p_target * (1.0 - p_fuzzy)

# 5. 为每个基本情感列计算得分，并把得分存入新的列
emotion_cols = ["p_Happy", "p_Sad", "p_Angry", "p_Relaxed"]
for emo in emotion_cols:
    score_col = emo.replace("p_", "score_")  # "p_Happy" -> "score_Happy"
    df[score_col] = df.apply(lambda r: compute_emotion_score(r, emotion_col=emo), axis=1)

# 6. 根据得分对每种情感分别排序，生成歌单（取 top_k 首）
top_k = 20
playlists = {}

for emo in emotion_cols:
    score_col = emo.replace("p_", "score_")
    sorted_df = df.sort_values(by=score_col, ascending=False)
    # 只保留下列几列到歌单：
    # song_id, genres, 对应情感概率, p_Ambiguous, genre_weight, score_*
    top_songs = sorted_df.head(top_k)[
        ["song_id", "genres", emo, "p_Ambiguous", "genre_weight", score_col]
    ]
    playlists[emo] = top_songs

# 7. 将每个情感歌单保存为单独的 CSV 文件
for emo, playlist_df in playlists.items():
    emotion_name = emo.replace("p_", "").lower()  # "p_Happy" -> "happy"
    out_fname = f"playlist_{emotion_name}.csv"     # 例如 "playlist_happy.csv"
    playlist_df.to_csv(out_fname, index=False)
    print(f"已生成 {emo} 歌单，保存在：{out_fname}")

# 8. （可选）打印各歌单前几行做快速检查
for emo, playlist_df in playlists.items():
    score_col = emo.replace("p_", "score_")
    print(f"\n=== Top {top_k} songs for emotion {emo} ===")
    print(playlist_df.head()[["song_id", "genres", emo, "p_Ambiguous", "genre_weight", score_col]])
