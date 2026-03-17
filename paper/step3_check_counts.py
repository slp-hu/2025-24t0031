import numpy as np
import pandas as pd
import os

# ══════════ 配置 ══════════
DATA_FILE = "music_data.npz"
GENRE_THRESHOLD = 150  # 筛选标准：该流派的总曲数必须 > 150
OUTPUT_CSV = "big_genre_analysis_result.csv" # 结果保存文件名

def analyze_big_genres():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    try:
        print("Loading data...")
        data = np.load(DATA_FILE)
        genres = data['genres']
        emotions = data['emotions']
        ids = data['ids']
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1. 创建总表
    df = pd.DataFrame({'Genre': genres, 'Emotion': emotions, 'ID': ids})
    
    # 2. 计算每个流派的总曲数
    genre_counts = df['Genre'].value_counts()
    
    # 3. 筛选出“大流派” (Big Genres)
    big_genres = genre_counts[genre_counts > GENRE_THRESHOLD].index.tolist()
    
    print("\n" + "="*60)
    print(f"🎯 ANALYSIS REPORT: Genres with > {GENRE_THRESHOLD} Total Songs")
    print("="*60)
    
    if not big_genres:
        print(f"No genres found with > {GENRE_THRESHOLD} songs.")
        return

    # 用于收集所有结果的列表
    all_results = []

    # 4. 遍历每个大流派，查看内部情感分布
    for g in big_genres:
        total_in_genre = genre_counts[g]
        print(f"\n🎵 GENRE: {g} (Total: {total_in_genre} songs)")
        print("-" * 50)
        
        # 获取该流派下的所有数据
        sub_df = df[df['Genre'] == g]
        
        # 统计情感组合，并按数量从多到少排序
        emo_counts = sub_df['Emotion'].value_counts().reset_index()
        emo_counts.columns = ['Emotion', 'Count']
        
        # 打印详情并收集数据
        print(f"{'Emotion':<20} | {'Count':<10} | {'Status'}")
        
        for _, row in emo_counts.iterrows():
            emo = row['Emotion']
            count = row['Count']
            
            # 标记状态
            status_label = ""
            if count > 50:
                status_label = "High (Success Candidate)"
                print_status = "✅ High"
            elif count < 10:
                status_label = "Low (Failure Candidate)"
                print_status = "❌ Low"
            else:
                status_label = "Mid"
                print_status = "⚠️ Mid"
                
            print(f"{emo:<20} | {count:<10} | {print_status}")

            # 添加到结果列表
            all_results.append({
                'Genre': g,
                'Total_Genre_Songs': total_in_genre,
                'Emotion': emo,
                'Count': count,
                'Status': status_label
            })

    # 5. 保存到 CSV
    if all_results:
        result_df = pd.DataFrame(all_results)
        result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig') # utf-8-sig 防止Excel打开乱码
        
        print("\n" + "="*60)
        print(f"✅ Analysis saved to '{OUTPUT_CSV}'")
        print("You can open this file in Excel to filter/sort.")
    else:
        print("\nNo results to save.")

if __name__ == "__main__":
    analyze_big_genres()