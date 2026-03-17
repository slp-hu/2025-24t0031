"""
batch_extract_muq_fp32.py  ─ 批量抽取 MuQ embeddings（FP32）

• 按照 'id' 列批量读取音频 (CSV 可含 [id] 或 [id1,id2])
• 输出 G:\13kmid30s_muq\<id>.npy      1024 维向量
"""

import os
import glob, ast, torch, librosa, numpy as np, pandas as pd
from tqdm import tqdm
from muq import MuQ

# ─────── 用户可配置 ───────
CSV_PATH  = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_with_clusters_filtered.csv"
AUDIO_DIR = r"G:\13kmid30s"
OUT_DIR   = r"G:\13kmid30s_muq"

CKPT_NAME = "OpenMuQ/MuQ-large-msd-iter"
USE_1024  = True
BATCH_SR  = 24_000
# ─────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading MuQ checkpoint ...")
muq = MuQ.from_pretrained(CKPT_NAME).to(DEVICE).eval()

# 1) 读取并清洗 id 🔧
def clean_id(raw):
    raw = str(raw).strip()
    try:                           # 尝试把 "[123,456]" 解析成列表
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return str(val[0])
    except (ValueError, SyntaxError):
        pass
    # 去除两侧方括号、空格，再取逗号/空格之前的部分
    raw = raw.strip("[]").split(",")[0].split()[0]
    return raw

df = pd.read_csv(CSV_PATH)
song_ids = [clean_id(x) for x in df["id"]]
print(f"Total files to process: {len(song_ids)}")

# 2) 批量抽取
for sid in tqdm(song_ids, desc="Extracting (FP32)"):
    out_path = os.path.join(OUT_DIR, f"{sid}.npy")
    if os.path.exists(out_path):
        continue

    # 找音频文件
    files = glob.glob(os.path.join(AUDIO_DIR, f"{sid}.*"))
    if not files:
        print(f"[WARN] {sid} audio not found, skip.")
        continue
    audio_path = files[0]

    # 解码 & 重采样
    wav, _ = librosa.load(audio_path, sr=BATCH_SR, mono=True)
    wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = muq(wav_t)
        vec = out.last_hidden_state.mean(1)

        if not USE_1024:
            if hasattr(muq, "projector"):
                vec = muq.projector(vec)
            else:
                raise RuntimeError("MuQ 版本过旧，请 pip install -U muq 升级。")

    np.save(out_path, vec.squeeze(0).cpu().numpy())

print("✅ 任务完成！向量已保存至:", OUT_DIR)
