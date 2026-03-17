import pandas as pd

# ---------- 参数配置 （请根据实际路径调整） ------------
input_csv_path  = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_intersection_v2_processed.csv"
output_csv_path = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_intersection_v2_cleaned.csv"
column_name     = "melon_id"  # 或者 "melon_id"（根据你的 CSV 列名调整）
# ---------------------------------------------------------

# 1. 缺失的 melon_id 列表
missing_ids = [
    189724, 563019, 563457, 563988, 37836, 189354, 37285, 563840, 563420,
    189518, 189172, 37212, 37254, 563620, 563691, 189118, 563863, 563010,
    189274, 189589, 37113, 563369, 563726, 563540, 189998, 189547, 189032,
    189280, 189587, 563373, 189753, 189894, 189389, 189096, 189525, 563366,
    189444, 189794, 189833, 563786, 563680, 563955, 563117, 563137, 37055,
    563379, 563160, 563791, 37813, 189803, 563670, 189340, 563635, 37027,
    37327
]

# 2. 加载 CSV（尝试 utf-8，失败再 latin1）
try:
    df = pd.read_csv(input_csv_path, encoding="utf-8", low_memory=False)
except Exception:
    df = pd.read_csv(input_csv_path, encoding="latin1", low_memory=False)

# 3. 转数值 & 删除缺失 ID 对应的行
df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
df_cleaned = df[~df[column_name].isin(missing_ids)]

# 4. 保存新 CSV
df_cleaned.to_csv(output_csv_path, index=False)
print(f"✅ 清理完成，输出文件：{output_csv_path}")
