import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# =================== 路径参数 ===================
MEL_DIR   = Path("mel_features")   # 你的 *.npy 所在目录
OUTPUT_CSV = Path("angry_feats.csv")  # 输出文件

# =================== 核心函数 ===================
def load_mel_linear(npy_path: Path):
    """
    读取 log10 Mel 并还原到线性功率域:
    mel_lin = (10**y - 1) / 10000
    """
    y = np.load(npy_path)        # (n_mels, T)     log10 值
    mel_lin = (np.power(10.0, y) - 1.0) / 10000.0
    return mel_lin.astype(np.float32)

def calc_median_dEdt(mel_lin):
    """瞬时能量上升: 先求每帧能量 E(t)，对 E 做一阶差分，再取中位数"""
    E = mel_lin.sum(axis=0)          # (T,)
    dE = np.diff(E)                  # (T-1,)
    return float(np.median(dE))

def calc_mean_roughness_proxy(mel_lin):
    """
    粗糙度近似: 先把每帧 Mel 归一化后，做邻带幅度乘积再对频带求和。
    最后对时间取平均。
    """
    # 每帧总能量，防止除零加 eps
    E = mel_lin.sum(axis=0, keepdims=True) + 1e-8
    mel_norm = mel_lin / E           # (n_mels, T)
    rough_frame = (mel_norm[:-1] * mel_norm[1:]).sum(axis=0)  # (T,)
    return float(rough_frame.mean())

# =================== 批量处理 ===================
records = []
for npy_path in tqdm(sorted(MEL_DIR.glob("*.npy")), desc="Extracting features"):
    mel_lin = load_mel_linear(npy_path)
    median_dE_dt  = calc_median_dEdt(mel_lin)
    mean_roughness = calc_mean_roughness_proxy(mel_lin)

    song_id = npy_path.stem
    records.append({
        "song_id": song_id,
        "median_dE_dt": median_dE_dt,
        "mean_roughness_proxy": mean_roughness
    })

pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
print(f"✅  Features saved to {OUTPUT_CSV}")
