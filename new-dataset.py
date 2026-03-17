#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JoyfulJune × Melon 交集 CSV（含一级/二级流派；逐步生成列，避免 struct/unnest 问题）
"""

import re
import json
from pathlib import Path
import polars as pl

# -------- 路径：按需修改 --------
JOY_CSV    = Path(r"E:\joyfuljuneS16k\songs_169148_with_tag.csv")
MEL_JSON   = Path(r"E:\melon\melon\metadata\kakao_meta\song_meta.json")
GENRE_JSON = Path(r"C:\Users\YAO\Downloads\genre_gn_all_en.json")
OUT_CSV    = Path(r"E:\joyfuljuneS16k\joyful_melon_intersection_v2.csv")
# --------------------------------

def norm(t: str) -> str:
    t = re.sub(r"\s*\(.*?\)\s*$", "", t or "")
    return re.sub(r"\s+", " ", t.strip().lower())

# 1) 读取 Melon，构建映射 ----------------------------------------
df_melon = (
    pl.read_json(MEL_JSON)
      .with_columns([
          pl.col("song_name").map_elements(norm, return_dtype=pl.Utf8).alias("n_song"),
          pl.col("artist_name_basket").list.first()
            .map_elements(norm, return_dtype=pl.Utf8).alias("n_artist"),
          pl.col("song_gn_gnr_basket").list.first()
            .cast(pl.Utf8)
            .alias("sub_genre_code")
      ])
      .select([
          pl.concat_str(["n_song","n_artist"], separator=" | ")
            .alias("key"),
          pl.col("id").alias("melon_id"),
          pl.col("sub_genre_code")
      ])
)
# build dicts
melon_id_map    = {r["key"]: r["melon_id"]     for r in df_melon.rows(named=True)}
melon_genre_map = {r["key"]: r["sub_genre_code"] for r in df_melon.rows(named=True)}

# 2) 读取 genre 英文映射 ----------------------------------------
code2en = json.load(GENRE_JSON.open(encoding="utf-8"))

def major_of(code: str) -> str:
    if not code or len(code) < 6:
        return ""
    return f"GN{code[2:4]}00"

# 3) 读取 JoyfulJune 并过滤交集 ----------------------------------
df = (
    pl.read_csv(JOY_CSV, try_parse_dates=False)
      .with_columns([
          pl.col("name").map_elements(norm, return_dtype=pl.Utf8).alias("n_song"),
          pl.col("artist").map_elements(norm, return_dtype=pl.Utf8).alias("n_artist")
      ])
      .with_columns(
          pl.concat_str(["n_song","n_artist"], separator=" | ")
            .alias("key")
      )
      .filter(pl.col("key").is_in(list(melon_id_map.keys())))
)

# 4) 依 key 添加 melon_id & sub_genre_code --------------------
df = df.with_columns([
    pl.col("key").map_elements(lambda k: melon_id_map[k], return_dtype=pl.Int64)
      .alias("melon_id"),
    pl.col("key").map_elements(lambda k: melon_genre_map[k], return_dtype=pl.Utf8)
      .alias("sub_genre_code")
])

# 5) 生成一级大类 code ------------------------------------------
df = df.with_columns([
    pl.col("sub_genre_code")
      .map_elements(major_of, return_dtype=pl.Utf8)
      .alias("major_genre_code")
])

# 6) 根据 code2en 填英文名 ---------------------------------------
df = df.with_columns([
    pl.col("major_genre_code")
      .map_elements(lambda c: code2en.get(c, c), return_dtype=pl.Utf8)
      .alias("major_genre_name_en"),
    pl.col("sub_genre_code")
      .map_elements(lambda c: code2en.get(c, c), return_dtype=pl.Utf8)
      .alias("sub_genre_name_en")
])

# 7) 清理临时列并输出 ------------------------------------------
df = df.drop(["n_song","n_artist","key"])

print(f"交集条数：{df.height:,}  →  写入 {OUT_CSV}")
df.write_csv(OUT_CSV, null_value="")

print("✅ Finished!")
