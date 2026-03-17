# -*- coding: utf-8 -*-
import os, json, math, random
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ===================== 配置 =====================
DATA_DIR = r"C:\Users\YAO\Desktop\genre ml\newdata\split"
INFERENCE_RESULTS_CSV = os.path.join(DATA_DIR, "emotion_library.csv")   # 推理输出（含 emotion_vector）
DEV_SET_CSV = os.path.join(DATA_DIR, "dev_val_plus_test.csv")           # 若推理CSV无 GT，则从这里合并 emotion_sequence

K = 10
QUERY_SET_SIZE = 100
RANDOM_STATE = 42                 # 复现实验
RANDOM_BASELINE_SAMPLES = 200     # 随机基线每个查询的采样次数（越大越稳，时间也更久）
USE_BOOTSTRAP_CI = True           # 是否计算 bootstrap 置信区间
BOOTSTRAP_B = 2000                # bootstrap 次数
ALPHA = 0.05                      # 95% CI

# “中等口径”相关性阈值
RELEVANCE_CFG = dict(
    min_overlap=2,           # 至少重合2个标签
    jaccard_tau=0.30,        # 或 Jaccard >= 0.30
    require_same_genre=False,# 是否要求同流派（中等口径默认 False）
    genre_bonus=0.0          # nDCG分级增益中的流派加分，默认0
)

# ===================== 工具函数 =====================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)

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

# 二元相关判定 & 分级增益（与口径一致）
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

def bootstrap_ci(values, B=2000, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(12345)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0: return (0.0, 0.0)
    means = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        means.append(values[idx].mean())
    lo = float(np.quantile(means, alpha/2))
    hi = float(np.quantile(means, 1-alpha/2))
    return lo, hi

# ===================== 主流程 =====================
if __name__ == "__main__":
    seed_all(RANDOM_STATE)

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

    # 3) 取真实标签：优先推理CSV中的 gt_emotion_sequence；若无则与 DEV_SET_CSV 合并获取
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
    lib_size = len(library_df)

    # 7) 评估（模型 & 随机基线）
    model_prec, model_map, model_rec, model_hit, model_mrr, model_ndcg = [], [], [], [], [], []
    rand_prec,  rand_map,  rand_rec,  rand_hit,  rand_mrr,  rand_ndcg  = [], [], [], [], [], []
    np_at_k_list, lift_list = [], []
    avg_rel_counts = []
    skipped = 0

    rng = np.random.default_rng(RANDOM_STATE)

    for _, q in tqdm(query_df.iterrows(), total=len(query_df), desc="评估中"):
        qid = q['id']
        qvec = q['emotion_vector'].reshape(1,-1)
        q_gt = q['gt_set']
        q_genre = str(q['major_genre_name_en'])

        # 生成模型 Top-K 推荐（过滤自身）
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

        # 分级增益（与口径一致，用 Jaccard + 可选流派加分）
        gains = np.array([
            graded_gain(q_gt, Gi, genre_q=q_genre, genre_i=lib_genres[i], cfg=RELEVANCE_CFG)
            for i, Gi in enumerate(lib_gt_sets)
        ], dtype=float)

        # ===== 模型指标 =====
        p_at_k  = precision_at_k(rec_ids, rel_set, K)
        ap_at_k = average_precision_at_k(rec_ids, rel_set, K)
        r_at_k  = recall_at_k(rec_ids, rel_set, K)
        h_at_k  = hitrate_at_k(rec_ids, rel_set, K)
        mrr_k   = mrr_at_k(rec_ids, rel_set, K)
        ndcg_k  = ndcg_at_k_from_gains(gains, rec_indices, K)

        model_prec.append(p_at_k); model_map.append(ap_at_k)
        model_rec.append(r_at_k);  model_hit.append(h_at_k)
        model_mrr.append(mrr_k);   model_ndcg.append(ndcg_k)

        # ===== 随机基线（对每个查询重复采样取均值）=====
        # 候选索引全集（库中全部）
        all_idx = np.arange(lib_size, dtype=int)
        # 由于库与查询已按 id 隔离，这里无需特意剔除自身；若保险：
        # all_idx = np.array([i for i in range(lib_size) if lib_ids[i] != qid], dtype=int)

        rp, rmap, rrec, rhit, rmrr, rndcg = [], [], [], [], [], []
        for _ in range(RANDOM_BASELINE_SAMPLES):
            sampled = rng.choice(all_idx, size=min(K, lib_size), replace=False)
            rand_ids = [lib_ids[i] for i in sampled]

            rp.append( precision_at_k(rand_ids, rel_set, K) )
            rmap.append( average_precision_at_k(rand_ids, rel_set, K) )
            rrec.append( recall_at_k(rand_ids, rel_set, K) )
            rhit.append( hitrate_at_k(rand_ids, rel_set, K) )
            rmrr.append( mrr_at_k(rand_ids, rel_set, K) )
            rndcg.append( ndcg_at_k_from_gains(gains, sampled.tolist(), K) )

        rand_prec.append( float(np.mean(rp)) )
        rand_map.append(  float(np.mean(rmap)) )
        rand_rec.append(  float(np.mean(rrec)) )
        rand_hit.append(  float(np.mean(rhit)) )
        rand_mrr.append(  float(np.mean(rmrr)) )
        rand_ndcg.append( float(np.mean(rndcg)) )

        # ===== NP@K & Lift@K（基于 Precision）=====
        R = len(rel_set)
        p_random = R / lib_size
        p_oracle = min(1.0, R / K)
        denom = max(p_oracle - p_random, 1e-12)
        np_at_k = (p_at_k - p_random) / denom
        lift = p_at_k / max(p_random, 1e-12)
        np_at_k_list.append(np_at_k)
        lift_list.append(lift)

    # 8) 汇总
    def avg(x): return float(np.mean(x)) if x else 0.0

    def maybe_ci(values):
        if not USE_BOOTSTRAP_CI: return "", ""
        lo, hi = bootstrap_ci(values, B=BOOTSTRAP_B, alpha=ALPHA, rng=np.random.default_rng(RANDOM_STATE))
        return f"[{lo:.4f}, {hi:.4f}]", (lo, hi)

    print("\n" + "="*60)
    print("离线推荐评估：模型 vs 随机基线（中等口径）")
    print(f"K={K} | 查询数={len(query_df)} | 被跳过（无相关项）={skipped} | 库大小={lib_size}")
    print(f"无效向量丢弃: {invalid_vec_cnt}")
    print(f"平均每查询的“相关项”数量: {avg(avg_rel_counts):.1f}")
    print("-"*60)

    metrics = [
        ("Precision@K", model_prec, rand_prec),
        ("MAP@K",       model_map,  rand_map),
        ("Recall@K",    model_rec,  rand_rec),
        ("HitRate@K",   model_hit,  rand_hit),
        ("MRR@K",       model_mrr,  rand_mrr),
        ("nDCG@K",      model_ndcg, rand_ndcg),
    ]

    for name, mvals, rvals in metrics:
        m_mean = avg(mvals); r_mean = avg(rvals); delta = m_mean - r_mean
        ci_text, _ = maybe_ci(mvals)
        print(f"{name:12s}  模型={m_mean:.4f} {('95% CI ' + ci_text) if USE_BOOTSTRAP_CI else ''}  |  随机={r_mean:.4f}  |  Δ={delta:.4f}")

    npk_mean = avg(np_at_k_list)
    lift_mean = avg(lift_list)
    ci_text_npk, _  = maybe_ci(np_at_k_list)
    ci_text_lift, _ = maybe_ci(lift_list)
    print("-"*60)
    print(f"NP@{K}:  {npk_mean:.4f} {('95% CI ' + ci_text_npk) if USE_BOOTSTRAP_CI else ''}")
    print(f"Lift@{K}: {lift_mean:.2f}x {('95% CI ' + ci_text_lift) if USE_BOOTSTRAP_CI else ''}")
    print("="*60)
