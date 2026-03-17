import os
import shutil
import pandas as pd

# -------------------------------
# 参数配置（请根据实际路径调整）
# -------------------------------
csv_path    = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_intersection_v2.csv"
root_dir    = r"E:\melon\melon\arena_mel"
output_dir  = r"E:\melon\extracted"
chunk_size  = 1000          # 每个文件夹包含的 .npy 数量区间
column_name = "melon_id"    # CSV 中第85列的列名
# -------------------------------

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

def load_ids(path, col):
    """
    加载 CSV 中指定列，尝试 utf-8，如果失败再尝试 latin1，返回唯一的整数 ID 列表。
    """
    try:
        df = pd.read_csv(path, usecols=[col], encoding="utf-8", low_memory=False)
    except Exception:
        print("⚠️ utf-8 解码失败，尝试 latin1 重新读取…")
        df = pd.read_csv(path, usecols=[col], encoding="latin1", low_memory=False)
    return pd.to_numeric(df[col], errors="coerce").dropna().astype(int).unique().tolist()

melon_ids = load_ids(csv_path, column_name)

if not os.path.isdir(root_dir):
    print(f"❌ 目录未找到: {root_dir}，请检查路径是否正确。")
else:
    for mid in melon_ids:
        block = mid // chunk_size
        src = os.path.join(root_dir, str(block), f"{mid}.npy")
        dst = os.path.join(output_dir, f"{mid}.npy")
        if os.path.isfile(src):
            shutil.copy(src, dst)
        else:
            print(f"⚠️ 未找到: {src}")

print("🎉 提取完成！所有存在的 .npy 文件已复制到输出目录。")
