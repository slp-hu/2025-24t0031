#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计 Melon（song_meta.json）与 JoyfulJuneS16k（songs_169148_with_tag.csv）
之间“歌曲名 + 歌手”精确匹配的交集数量。
"""

import re
from pathlib import Path

# ────────── 1. 文件路径写在这里 ──────────
MELON_JSON = Path(r"E:\melon\melon\metadata\kakao_meta\song_meta.json")
JJ_CSV     = Path(r"E:\joyfuljuneS16k\songs_169148_with_tag.csv")

# 若想强制用 pandas，设 True；否则先尝试 Polars，再自动降级
USE_PANDAS = False
CHUNKSIZE  = 200_000        # pandas 分块读取行数

# ────────── 2. 字符串规范化 ──────────
def normalize(text: str) -> str:
    """小写、去掉尾括号、收紧空格"""
    if text is None:
        return ""
    text = re.sub(r"\s*\(.*?\)\s*$", "", text)
    return re.sub(r"\s+", " ", text.strip().lower())

# ────────── 3. Polars 版本（最快）─────────
def count_with_polars(melon_path: Path, jj_path: Path) -> tuple[int, int, int]:
    import polars as pl

    df_melon = (
        pl.read_json(melon_path)                                  # ← ⬅ 没有 encoding 参数
        .with_columns([
            pl.col("song_name").map_elements(normalize).alias("n_song"),
            pl.col("artist_name_basket").list.first().map_elements(normalize).alias("n_artist")
        ])
        .select(pl.concat_str(["n_song", "n_artist"], separator=" | ").alias("key"))
    )

    df_jj = (
        pl.read_csv(jj_path, encoding="utf8", columns=["name", "artist"])
        .with_columns([
            pl.col("name").map_elements(normalize).alias("n_song"),
            pl.col("artist").map_elements(normalize).alias("n_artist")
        ])
        .select(pl.concat_str(["n_song", "n_artist"], separator=" | ").alias("key"))
    )

    mel_keys = set(df_melon["key"].to_list())
    jj_keys  = set(df_jj["key"].to_list())
    return len(mel_keys), len(jj_keys), len(mel_keys & jj_keys)

# ────────── 4. pandas 分块版本（低内存）─────────
def count_with_pandas(melon_path: Path, jj_path: Path) -> tuple[int, int, int]:
    import pandas as pd
    from tqdm import tqdm

    melon_df = pd.read_json(melon_path)
    melon_df["key"] = (
        melon_df["song_name"].map(normalize) + " | " +
        melon_df["artist_name_basket"].str[0].map(normalize)
    )
    mel_keys = set(melon_df["key"].values)

    jj_keys, total = set(), 0
    for chunk in tqdm(pd.read_csv(jj_path, chunksize=CHUNKSIZE,
                                  usecols=["name", "artist"], encoding="utf8")):
        chunk["key"] = (
            chunk["name"].map(normalize) + " | " +
            chunk["artist"].map(normalize)
        )
        jj_keys.update(chunk["key"].values)
        total += len(chunk)

    return len(mel_keys), total, len(mel_keys & jj_keys)

# ────────── 5. 主流程 ──────────
def main():
    if not MELON_JSON.exists() or not JJ_CSV.exists():
        raise FileNotFoundError("❌ 路径不对，请检查 MELON_JSON / JJ_CSV")

    if not USE_PANDAS:
        try:
            mel_cnt, jj_cnt, inter_cnt = count_with_polars(MELON_JSON, JJ_CSV)
        except ModuleNotFoundError:
            print("⚠️  未安装 Polars，改用 pandas 分块模式……")
            mel_cnt, jj_cnt, inter_cnt = count_with_pandas(MELON_JSON, JJ_CSV)
    else:
        mel_cnt, jj_cnt, inter_cnt = count_with_pandas(MELON_JSON, JJ_CSV)

    print("\n" + "-" * 60)
    print(f"Melon 曲目总数          : {mel_cnt:,}")
    print(f"JoyfulJuneS16k 曲目总数 : {jj_cnt:,}")
    print(f"交集（精确匹配）曲目数 : {inter_cnt:,}")
    print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
