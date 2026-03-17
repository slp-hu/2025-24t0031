import os
import pandas as pd

# --- 请在这里配置 ---
# 【重要】请确保这两个路径和你的主脚本完全一致
EMB_DIR  = r"G:\13kmid30s_muq"
CSV_PATH = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_final_labeled_dataset.csv"
ID_COL   = "melon_id"
# --------------------


print("="*60)
print("开始进行文件系统访问诊断...")
print("="*60)

# 1. 首先，检查Embedding目录本身是否存在
print(f"步骤 1: 检查目录 '{EMB_DIR}'...")
if os.path.isdir(EMB_DIR):
    print(f"✅ 成功: 目录 '{EMB_DIR}' 已找到。")
else:
    print(f"❌ 失败: 目录 '{EMB_DIR}' 未找到或不是一个目录。")
    print("请立即停止并修正 Config 中的 EMB_DIR 路径。")
    exit() # 如果目录都找不到，后续检查无意义

# 2. 如果目录存在，尝试列出其中的前5个文件
print("\n步骤 2: 尝试列出目录中的内容...")
try:
    file_list = os.listdir(EMB_DIR)
    if not file_list:
        print("⚠️ 警告: 目录 '{EMB_DIR}' 是空的，里面没有任何文件。")
    else:
        print(f"✅ 成功: 在目录中找到了 {len(file_list)} 个项目。")
        print("目录中前5个文件的示例:")
        for fname in file_list[:5]:
            print(f"  - {fname}")
except Exception as e:
    print(f"❌ 失败: 尝试列出目录内容时出错: {e}")
    print("这通常意味着存在权限问题。请检查你的Python脚本是否有权读取G盘。")
    exit()

# 3. 从CSV中读取几个ID，然后尝试构建路径并检查
print("\n步骤 3: 从CSV加载ID并检查对应的.npy文件...")
try:
    df = pd.read_csv(CSV_PATH)
    # 取前5个ID作为样本进行测试
    sample_ids = df[ID_COL].dropna().head(5).tolist()
    print(f"将使用以下示例ID进行测试: {sample_ids}")

    for melon_id in sample_ids:
        # 使用和你主脚本完全相同的逻辑来构建路径
        # 强制转换为整数，再转为字符串
        melon_id_str = str(int(melon_id))
        expected_path = os.path.join(EMB_DIR, f"{melon_id_str}.npy")
        
        print("-" * 20)
        print(f"正在检查 ID: {melon_id_str}")
        print(f"  -> 代码构建的路径是: {expected_path}")
        
        if os.path.exists(expected_path):
            print(f"  -> ✅ 成功: 文件已找到！")
        else:
            print(f"  -> ❌ 失败: 文件未找到！")

except Exception as e:
    print(f"\n❌ 失败: 读取CSV或处理ID时出错: {e}")


print("\n" + "="*60)
print("诊断结束。")
print("="*60)