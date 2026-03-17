import os
import numpy as np
import pandas as pd
import soundfile as sf
import openl3
import tensorflow as tf
from tqdm import tqdm

# -------------------- 配置 --------------------
AUDIO_DIR  = r"C:\Users\YAO\Desktop\MEMD_audio"
CSV_PATH   = os.path.join(AUDIO_DIR, "processed_annotations.csv")
OUT_DIR    = os.path.join(AUDIO_DIR, "openl3_emb")
AUDIO_EXTS = [".wav", ".mp3", ".flac", ".ogg"]
BATCH_SIZE = 32          # 每批处理的窗口数，GPU 可适当增大
# ---------------------------------------------

print("🖥  Detected GPU:", tf.config.list_physical_devices('GPU'))
os.makedirs(OUT_DIR, exist_ok=True)

# 1) 读取待处理 song_id
df = pd.read_csv(CSV_PATH)
song_ids = df["song_id"].astype(str).tolist()

# 2) 预加载 Kapre 前端模型
model = openl3.models.load_audio_embedding_model(
    input_repr="mel128", content_type="music", embedding_size=512, frontend='kapre'
)

def find_audio(song_id: str):
    for ext in AUDIO_EXTS:
        p = os.path.join(AUDIO_DIR, f"{song_id}{ext}")
        if os.path.exists(p):
            return p
    return None

# 3) 主循环
for sid in tqdm(song_ids, desc="Extracting (GPU Kapre)"):
    out_path = os.path.join(OUT_DIR, f"{sid}.npy")
    if os.path.exists(out_path):
        continue

    audio_path = find_audio(sid)
    if audio_path is None:
        print(f"[WARN] 找不到音频: {sid}")
        continue

    try:
        audio, sr = sf.read(audio_path)
    except RuntimeError as e:
        print(f"[ERROR] 读取失败 {audio_path}: {e}")
        continue

    # --- 核心：Kapre 前端 + GPU ---
    emb_seq, _ = openl3.get_audio_embedding(
        audio, sr,
        model=model,       # 复用已加载模型（Kapre，GPU）
        center=False,
        batch_size=BATCH_SIZE,
        verbose=0
    )
    # ------------------------------

    emb_mean = emb_seq.mean(axis=0, keepdims=True).astype(np.float32)
    np.save(out_path, emb_mean)

print(f"✅ 全部完成！输出在 {OUT_DIR}")
