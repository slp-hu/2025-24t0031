import os, glob, gc, torch, librosa, numpy as np
from muq import MuQ
from tqdm import tqdm

IN_DIR  = r"C:\Users\YAO\Desktop\MEMD_audio"
OUT_DIR = r"C:\Users\YAO\Desktop\MEMD_audio\muq"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()

TARGET_SEC = 45          # ★ 裁剪长度
SR         = 24_000
audio_fs   = glob.glob(os.path.join(IN_DIR, "*.mp3"))

for fp in tqdm(audio_fs, desc="MuQ FP32"):
    fid = os.path.splitext(os.path.basename(fp))[0]
    out_fp = os.path.join(OUT_DIR, f"{fid}.npy")
    if os.path.exists(out_fp):
        continue

    wav, _ = librosa.load(fp, sr=SR, mono=True)
    if len(wav) > SR * TARGET_SEC:           # ★ 裁剪到中段 30 s
        mid   = len(wav) // 2
        half  = SR * TARGET_SEC // 2
        wav   = wav[mid-half : mid+half]

    wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.inference_mode():             # ★ 比 no_grad 更省显存
        emb = muq(wav_t).last_hidden_state.mean(1)  # ★ 汇聚
        np.save(out_fp, emb.squeeze(0).cpu().numpy().astype("float32"))

    del wav_t, emb
    torch.cuda.empty_cache(); gc.collect()

print("✅ 全部完成，向量已保存到:", OUT_DIR)
