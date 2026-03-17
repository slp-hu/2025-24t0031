import torch, librosa, numpy as np
from muq import MuQ

# ---------- 1. 读取 MP3 ----------
wav, _ = librosa.load(r"C:\Users\YAO\Desktop\MEMD_audio\2.mp3",
                      sr=24_000, mono=True)          # 24 kHz 单声道
wavs = torch.tensor(wav).unsqueeze(0).to("cuda")     # (1, T)

# ---------- 2. 载入 MuQ ----------
muq = (MuQ
       .from_pretrained("OpenMuQ/MuQ-large-msd-iter")
       .to("cuda")
       .eval())

# ---------- 3. 推理并取均值嵌入 ----------
with torch.no_grad():                    # 先不用 autocast，保证稳定
    out = muq(wavs)                      # 不传输出选项
    emb = out.last_hidden_state.mean(1)  # (1, 1024)
    emb_np = emb.squeeze(0).cpu().numpy()

# ---------- 4. 保存为 npy ----------
np.save("2_emb.npy", emb_np)
print("Saved 2_emb.npy, shape:", emb_np.shape)
