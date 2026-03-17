import pandas as pd

# 读取第一个 CSV 文件
df1 = pd.read_csv('all_features2.csv')

# 读取第二个 CSV 文件
df2 = pd.read_csv('updated_audio_features_with_id_first.csv')

# 合并两个 DataFrame
merged_df = pd.concat([df1, df2], ignore_index=True)

# 如果需要去重，可以使用 drop_duplicates
# merged_df = merged_df.drop_duplicates()

# 将合并后的 DataFrame 保存为新的 CSV 文件
merged_df.to_csv('merged_file.csv', index=False, encoding='utf-8-sig')

print("CSV 文件已成功合并为 'merged_file.csv'")
