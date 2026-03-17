#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_ncm.py  —— 基于 manhole-cover demo 第三方 API 的网易云批量下载脚本（极高→标准，可断点续传）
====================================================================================
亮点：
1. **极高→标准** 两档音质自动降级；无浏览器、无加密，仅 requests。
2. 支持 **断点续传**：启动时先扫描 --outdir 已有 *.mp3 / *.flac，跳过已下 ID。
3. CSV 的 `id` 列可写成 `[21256424, 123]`；脚本取第一个。
4. 依赖：`pip install requests`（如需代理可 `--proxy`）。
"""

import argparse, csv, ast, os, sys, time, requests, re
from typing import List, Tuple, Set

# ---------------- CLI 参数 ----------------
parser = argparse.ArgumentParser(description="批量下载网易云音乐（支持续传）")
parser.add_argument("--csv", default=r"C:\Users\YAO\Desktop\genre ml\joyful_melon_intersection_v2_with8feats.csv", help="包含 id 列的 CSV")
parser.add_argument("--outdir", default=r"G:\\13k", help="保存目录")
parser.add_argument("--proxy", default="", help="HTTP(S) 代理，如 http://127.0.0.1:7890")
args = parser.parse_args()

CSV_FILE = args.csv
OUT_DIR  = args.outdir
PROXIES  = {"http": args.proxy, "https": args.proxy} if args.proxy else None
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- 音质顺序 ----------------
QUALITY_SEQ = ["exhigh", "standard"]
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36")

API_TMPL = (
    "https://api.kxzjoker.cn/api/163_music?"
    "url=https://y.music.163.com/m/song?id={id}&userid=8719916627&dlt=0846&"
    "level={ql}&type=json"
)

# ---------------- 工具函数 ----------------

def scan_downloaded(out_dir: str) -> Set[int]:
    """扫描目录中已存在的音频文件，返回已下载 id 集合"""
    ids: Set[int] = set()
    pattern = re.compile(r"^(\d+)")  # 文件名开头的数字即 id
    for name in os.listdir(out_dir):
        if name.lower().endswith((".mp3", ".flac", ".m4a", ".ogg", ".wav")):
            m = pattern.match(name)
            if m:
                try:
                    ids.add(int(m.group(1)))
                except ValueError:
                    continue
    return ids

def query_api(song_id: int, quality: str) -> dict:
    url = API_TMPL.format(id=song_id, ql=quality)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, proxies=PROXIES, timeout=10)
        return r.json()
    except Exception as e:
        print(f"    · API 失败: {e}")
        return {}

def fetch_best_link(song_id: int) -> Tuple[str, str]:
    for ql in QUALITY_SEQ:
        data = query_api(song_id, ql)
        if data.get("status") == 200 and data.get("url"):
            return data["url"], data.get("level", ql)
        time.sleep(0.3)
    return "", ""

def download(url: str, path: str):
    try:
        with requests.get(url, headers={"User-Agent": UA}, stream=True, proxies=PROXIES, timeout=10) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done * 100 // total
                        sys.stdout.write(f"\r      进度 {pct}%")
                        sys.stdout.flush()
            print("\r      完成 ✔")
    except Exception as e:
        print(f"      下载失败: {e}")

# ---------------- 主流程 ----------------

def main():
    downloaded_ids = scan_downloaded(OUT_DIR)
    if downloaded_ids:
        print(f"已存在 {len(downloaded_ids)} 首音频，启动时将跳过这些 ID…")

    with open(CSV_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        todo_idx = 0
        for row in reader:
            raw = row.get("id")
            if not raw:
                continue
            try:
                ids = ast.literal_eval(raw)
            except Exception:
                ids = [raw]
            if not isinstance(ids, list):
                ids = [ids]
            try:
                song_id = int(ids[0])
            except ValueError:
                continue

            if song_id in downloaded_ids:
                continue  # 跳过已下载

            todo_idx += 1
            print(f"\n[{todo_idx}] 处理 ID {song_id}")
            link, level = fetch_best_link(song_id)
            if not link:
                print("    × 获取直链失败，跳过")
                continue

            ext = ".flac" if "无损" in level else ".mp3"
            filename = f"{song_id}{ext}"
            save_path = os.path.join(OUT_DIR, filename)
            print(f"    ↓ 下载 ({level}) → {filename}")
            download(link, save_path)

    print("\n全部任务完成。")

if __name__ == "__main__":
    main()
