# -*- coding: utf-8 -*-
import os, json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ===================== 配置 =====================
DATA_DIR = r"C:\Users\YAO\Desktop\genre ml\newdata\split"
INFERENCE_RESULTS_CSV = os.path.join(DATA_DIR, "emotion_library.csv")   # 推理输出
DEV_SET_CSV = os.path.join(DATA_DIR, "dev_val_plus_test.csv")           # 若推理CSV无GT，则用它来merge取GT

K = 10
QUERY_SET_SIZE = 100
RANDOM_STATE = 42

# “中等口径”相关性阈值
RELEVANCE_CFG = dict(
    min_overlap=2,           # 至少重合2个标签
    jaccard_tau=0.30,        # 或者 Jaccard >= 0.30
    require_same_genre=False,# 是否要求同流派（中等口径默认 False）
    genre_bonus=0.0          # nDCG分级增益中的流派加分，默认0
)

# ===================== 工具函数 =====================
def parse_vector_cell(cell):
    """优先按 JSON 数组解析；失败再尝试逗号分隔。保留长度==12 的向量。"""
    if pd.isna(cell): return None
    s = str(cell).strip()
    if not s: return None
    # try JSON
    try:
        arr = json.loads(s)
        v = np.array(arr, dtype=float)
        return v if v.size == 12 else None
    except Exception:
        pass
    # try CSV-like
    try:
        if s.startswith('[') and s.endswith(']'):  # 容错去括号
            s = s[1:-1]
        v = np.fromstring(s, sep=',', dtype=float)
        return v if v.size == 12 else None
    except Exception:
        return None

def to_emotion_set(cell):
    if pd.isna(cell) or str(cell).strip() == "": return set()
    return set(tok.strip() for tok in str(cell).split(',') if tok.strip())

def jaccard(a: set, b: set):
    if not a and not b: return 0.0
    inter = len(a & b); uni = len(a | b)
    return (inter / uni) if uni > 0 else 0.0

# 评估指标
def precision_at_k(rec_list, rel_set, k):
    return len(set(rec_list[:k]) & rel_set) / max(k,1)

def average_precision_at_k(rec_list, rel_set, k):
    score, hits = 0.0, 0.0
    for i, item in enumerate(rec_list[:k]):
        if item in rel_set:
            hits += 1.0
            score += hits / (i + 1.0)
    denom = max(min(len(rel_set), k), 1)
    return score / denom

def recall_at_k(rec_list, rel_set, k):
    return (len(set(rec_list[:k]) & rel_set) / len(rel_set)) if len(rel_set)>0 else 0.0

def hitrate_at_k(rec_list, rel_set, k):
    return 1.0 if any(x in rel_set for x in rec_list[:k]) else 0.0

def mrr_at_k(rec_list, rel_set, k):
    for i, x in enumerate(rec_list[:k]):
        if x in rel_set: return 1.0/(i+1)
    return 0.0

def ndcg_at_k_from_gains(gains_vec, rec_indices, k):
    dcg = 0.0
    for rank, idx in enumerate(rec_indices[:k]):
        dcg += gains_vec[idx] / np.log2(rank + 2)
    ideal_idx = np.argsort(gains_vec)[::-1][:k]
    idcg = 0.0
    for rank, idx in enumerate(ideal_idx):
        idcg += gains_vec[idx] / np.log2(rank + 2)
    return (dcg / idcg) if idcg > 0 else 0.0

# 可配置的“二元相关”判定与“分级增益”
def is_relevant(Gq, Gi, genre_q=None, genre_i=None, cfg=RELEVANCE_CFG):
    if cfg.get("require_same_genre", False):
        if (genre_q is not None) and (genre_i is not None) and (genre_q != genre_i):
            return False
    inter = len(Gq & Gi)
    if inter >= cfg.get("min_overlap", 1):
        return True
    # 退一步看 Jaccard
    jac = jaccard(Gq, Gi)
    return jac >= cfg.get("jaccard_tau", 0.0)

def graded_gain(Gq, Gi, genre_q=None, genre_i=None, cfg=RELEVANCE_CFG):
    jac = jaccard(Gq, Gi)
    bonus = cfg.get("genre_bonus", 0.0)
    if bonus > 0 and (genre_q is not None) and (genre_i is not None) and (genre_q == genre_i):
        jac = min(1.0, jac + bonus)
    return jac

# ===================== 主流程 =====================
if __name__ == "__main__":
    # 1) 读取推理结果
    if not os.path.exists(INFERENCE_RESULTS_CSV):
        raise FileNotFoundError(f"找不到推理文件: {INFERENCE_RESULTS_CSV}")
    df = pd.read_csv(INFERENCE_RESULTS_CSV)
    df['id'] = df['id'].astype(str)

    # 2) 解析向量 & 过滤非法
    df['emotion_vector'] = df['emotion_vector'].apply(parse_vector_cell)
    vec_ok = df['emotion_vector'].apply(lambda v: isinstance(v, np.ndarray) and v.size==12)
    invalid_vec_cnt = int((~vec_ok).sum())
    df = df[vec_ok].copy()

    # 3) 取真实标签：优先使用推理CSV中的 gt_emotion_sequence；若没有则与 DEV_SET_CSV 合并获取
    if 'gt_emotion_sequence' in df.columns:
        df['gt_set'] = df['gt_emotion_sequence'].apply(to_emotion_set)
    else:
        if not os.path.exists(DEV_SET_CSV):
            raise FileNotFoundError("推理CSV未包含 gt_emotion_sequence，且找不到 DEV_SET_CSV 以获取GT。")
        dev_df = pd.read_csv(DEV_SET_CSV)
        dev_df['id'] = dev_df['id'].astype(str)
        df = pd.merge(df, dev_df[['id','emotion_sequence','major_genre_name_en']], on='id', how='inner')
        df.rename(columns={'emotion_sequence':'gt_emotion_sequence'}, inplace=True)
        df['gt_set'] = df['gt_emotion_sequence'].apply(to_emotion_set)

    # 4) 流派列（若无则置空字符串），去重
    if 'major_genre_name_en' not in df.columns:
        df['major_genre_name_en'] = ""
    df = df.dropna(subset=['gt_emotion_sequence']).drop_duplicates(subset='id').reset_index(drop=True)

    # 5) 划分 查询/曲库（按 id 隔离）
    if len(df) < 2:
        raise RuntimeError("有效样本不足，无法评估。")
    n_query = min(QUERY_SET_SIZE, max(1, len(df)-1))
    query_df = df.sample(n=n_query, random_state=RANDOM_STATE)
    query_ids = set(query_df['id'])
    library_df = df[~df['id'].isin(query_ids)].copy().reset_index(drop=True)

    # 6) 构建库矩阵与索引
    lib_vectors = np.vstack(library_df['emotion_vector'].values)
    lib_ids = library_df['id'].tolist()
    lib_gt_sets = library_df['gt_set'].tolist()
    lib_genres = library_df['major_genre_name_en'].astype(str).tolist()
    lib_id2idx = {sid:i for i, sid in enumerate(lib_ids)}

    # 7) 评估
    precs, maps, recs, hits, mrrs, ndcgs = [], [], [], [], [], []
    avg_rel_counts = []  # 每个查询在库中的“相关项”数量
    skipped = 0

    for _, q in tqdm(query_df.iterrows(), total=len(query_df), desc="评估中"):
        qid = q['id']
        qvec = q['emotion_vector'].reshape(1,-1)
        q_gt = q['gt_set']
        q_genre = str(q['major_genre_name_en'])

        # 找 Top-K 邻居（过滤自身）
        sims = cosine_similarity(qvec, lib_vectors)[0]
        order = np.argsort(sims)[::-1]
        rec_indices = []
        for idx in order:
            if lib_ids[idx] == qid:
                continue
            rec_indices.append(idx)
            if len(rec_indices) >= K:
                break
        rec_ids = [lib_ids[i] for i in rec_indices]

        # 构建二元相关集合（中等口径）
        rel_set = set()
        for i, Gi in enumerate(lib_gt_sets):
            if is_relevant(q_gt, Gi, genre_q=q_genre, genre_i=lib_genres[i], cfg=RELEVANCE_CFG):
                rel_set.add(lib_ids[i])
        avg_rel_counts.append(len(rel_set))

        if len(rel_set) == 0:
            skipped += 1
            continue

        precs.append(precision_at_k(rec_ids, rel_set, K))
        maps.append(average_precision_at_k(rec_ids, rel_set, K))
        recs.append(recall_at_k(rec_ids, rel_set, K))
        hits.append(hitrate_at_k(rec_ids, rel_set, K))
        mrrs.append(mrr_at_k(rec_ids, rel_set, K))

        # nDCG 的分级增益（与口径一致，使用 Jaccard + 可选流派加分）
        gains = np.array([
            graded_gain(q_gt, Gi, genre_q=q_genre, genre_i=lib_genres[i], cfg=RELEVANCE_CFG)
            for i, Gi in enumerate(lib_gt_sets)
        ], dtype=float)
        ndcgs.append(ndcg_at_k_from_gains(gains, rec_indices, K))

    # 8) 汇总
    def avg(x): return float(np.mean(x)) if x else 0.0
    print("\n" + "="*48)
    print("离线推荐评估（中等口径）")
    print(f"K={K} | 查询数={len(query_df)} | 被跳过（无相关项）={skipped} | 库大小={len(library_df)}")
    print(f"无效向量丢弃: {invalid_vec_cnt}")
    print(f"平均每查询的“相关项”数量: {avg(avg_rel_counts):.1f}")
    print("-"*48)
    print(f"Precision@{K}: {avg(precs):.4f}")
    print(f"MAP@{K}:       {avg(maps):.4f}")
    print(f"Recall@{K}:    {avg(recs):.4f}")
    print(f"HitRate@{K}:   {avg(hits):.4f}")
    print(f"MRR@{K}:       {avg(mrrs):.4f}")
    print(f"nDCG@{K}:      {avg(ndcgs):.4f}")
    print("="*48)
