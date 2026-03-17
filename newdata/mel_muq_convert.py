import os, glob, argparse, numpy as np, torch
import torch.nn.functional as F
from tqdm import tqdm
from muq import MuQ

# ---------- 参数 ----------
FREQ_IN   = 48
FREQ_OUT  = 128
SCALE_F   = FREQ_OUT / FREQ_IN               # 2.666…
SCALE_T   = (24000 * 1024) / (44100 * 320)   # ≈ 1.741
CHUNK_FRAMES = 375                           # ≈ 5 s 原 Mel 帧数 / hop 23.2 ms

# ---------- Mel48 → Mel128 ----------
def mel48_to_muq128(mel48: np.ndarray) -> torch.Tensor:
    """
    48×T numpy.float32, log10(1 + mel*1e4)  →  128×T' torch.float32, log10(mel+1e-6)
    """
    x = torch.from_numpy(mel48).unsqueeze(0).unsqueeze(0)   # [1,1,48,T]
    x_lin = (10 ** x - 1.0) / 1e4                           # 逆量纲
    x_log = torch.log10(torch.clamp(x_lin, min=1e-6))       # 对齐 MuQ 量纲

    _, _, _, T = x_log.shape
    # 频率轴插值
    x_log = F.interpolate(x_log, size=(FREQ_OUT, T), mode="bilinear", align_corners=False)
    # 时间轴插值
    T_new = int(round(T * SCALE_T))
    x_log = F.interpolate(x_log, size=(FREQ_OUT, T_new), mode="bilinear", align_corners=False)

    return x_log.squeeze(0).squeeze(0).contiguous()         # 128×T'

# ---------- 主流程 ----------
def main(src_dir, dst_dir, device):
    os.makedirs(dst_dir, exist_ok=True)

    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(device).eval()

    npy_files = glob.glob(os.path.join(src_dir, '*.npy'))
    for npy_path in tqdm(npy_files, ncols=80, desc='Processing'):
        fname = os.path.basename(npy_path)
        mel48 = np.load(npy_path).astype('float32')          # 48×T
        mel128 = mel48_to_muq128(mel48)                     # 128×T'
        T_new = mel128.shape[1]

        # ------------- 切 5 s 小块跑 -------------
        embeddings = []
        for start in range(0, T_new, CHUNK_FRAMES):
            end = min(start + CHUNK_FRAMES, T_new)
            chunk = mel128[:, start:end]                    # 128×chunk
            with torch.no_grad():
                out = muq(
                    input_features=chunk.unsqueeze(0).to(device),
                    output_hidden_states=False
                )
            embeddings.append(out.last_hidden_state.squeeze(0).cpu())  # frames/4 × 1024

        emb_full = torch.cat(embeddings, dim=0).numpy()     # (T_total/4, 1024)
        np.save(os.path.join(dst_dir, fname), emb_full)
        # 可选：再保存整首平均向量
        np.save(os.path.join(dst_dir, fname[:-4] + '_mean.npy'), emb_full.mean(axis=0))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=r'E:\melon\extracted', help='原 Mel48 文件夹')
    ap.add_argument('--dst', default=r'E:\melon\emb_muqlarge', help='输出 Embeddings 文件夹')
    ap.add_argument('--cpu', action='store_true', help='强制用 CPU')
    args = ap.parse_args()

    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    main(args.src, args.dst, device)
