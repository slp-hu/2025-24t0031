import json

def filter_jsonl(input_file_path, output_file_path, genre_to_exclude="Niche"):
    """
    过滤 JSONL 文件中 genres 包含指定类别的行。

    :param input_file_path: 输入 JSONL 文件路径
    :param output_file_path: 输出过滤后的 JSONL 文件路径
    :param genre_to_exclude: 要排除的 genres 类别
    """
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        
        for line_number, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                # 跳过空行
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"行 {line_number} 解析错误: {e}")
                continue  # 跳过无法解析的行

            genres = data.get("genres", [])
            if genre_to_exclude not in genres:
                # 将不包含要排除类别的行写入输出文件
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"过滤完成。过滤后的文件保存在: {output_file_path}")

# 使用示例
input_jsonl = 'merged_genres.jsonl'    # 替换为你的输入文件路径
output_jsonl = 'merged_genres_exclude_niche.jsonl'  # 替换为你希望输出的文件路径

filter_jsonl(input_jsonl, output_jsonl, genre_to_exclude="Niche")
