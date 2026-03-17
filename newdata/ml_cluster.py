import pandas as pd
import numpy as np
import ast
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. Configuration & Helper Function ---

# Define all parameters
FILE_PATH = r"C:\Users\YAO\Desktop\genre ml\newdata\joyful_melon_filtered_audioexist.csv"
GLOBAL_EPS = 2.5
GLOBAL_MIN_SAMPLES = 24
SUB_EPS = 1.38
SUB_MIN_SAMPLES = 24
HIGH_INTENSITY_THRESHOLD = 0.7
MIXED_CLUSTER_THRESHOLD = 0.4

def print_and_plot_report(title, summary_df, counts_series, figure_size=(18, 8)):
    """
    A helper function to print a terminal report and generate a heatmap.
    :param title: The title for the report and the plot.
    :param summary_df: DataFrame containing the mean emotion values.
    :param counts_series: Series containing the sample counts for each cluster.
    :param figure_size: The size of the plot figure.
    """
    print(f"\n--- {title} ---")
    
    # Print dominant emotion profile
    print("\nDominant Emotion Profile:")
    for cluster_id, emotion_values in summary_df.iterrows():
        dominant_emotions = {}
        high_emotions = emotion_values[emotion_values >= HIGH_INTENSITY_THRESHOLD]
        
        if not high_emotions.empty:
            dominant_emotions = high_emotions.round(2).to_dict()
        else:
            mixed_emotions = emotion_values[emotion_values >= MIXED_CLUSTER_THRESHOLD]
            if not mixed_emotions.empty:
                dominant_emotions = mixed_emotions.round(2).to_dict()
            else:
                # If no emotions meet the thresholds, show the top two
                dominant_emotions = emotion_values.nlargest(2).round(2).to_dict()
                
        print(f"Cluster '{cluster_id}' (Count: {counts_series.get(cluster_id, 0)}): {dominant_emotions}")

    # Create and display the heatmap
    plt.figure(figsize=figure_size)
    # Use transpose .T to have emotions on the y-axis for better readability
    sns.heatmap(summary_df.T, annot=True, cmap='viridis', fmt='.2f')
    plt.title(title)
    plt.xlabel('Cluster ID')
    plt.ylabel('Emotion')
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    print("\nPlot has been generated. Please view the pop-up window. The program will continue after you close it.")
    plt.show()


# --- 1. Load and Preprocess Data ---
print("--- Step 1: Loading and Preprocessing Data ---")
df = pd.read_csv(FILE_PATH)
df['tag_dict'] = df['tag'].apply(lambda x: ast.literal_eval(x))
tags_df = pd.json_normalize(df['tag_dict'])
emotion_columns = ['Healing', 'Nostalgia', 'Excitement', 'Sadness', 'Romantic', 
                   'Quiet', 'Happiness', 'Loneliness', 'Touching', 'Missing', 
                   'Fresh', 'Relaxation']
for col in emotion_columns:
    if col not in tags_df.columns:
        tags_df[col] = 0
tags_df = tags_df[emotion_columns].fillna(0)
X = tags_df.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data preprocessing complete.")


# --- PART 1: First Global Clustering & Report ---
print("\n" + "="*50)
print("PART 1: Global Clustering Analysis")
print("="*50)
dbscan_global = DBSCAN(eps=GLOBAL_EPS, min_samples=GLOBAL_MIN_SAMPLES)
df['cluster'] = dbscan_global.fit_predict(X_scaled)
global_summary = df.groupby('cluster')[emotion_columns].mean()
global_counts = df['cluster'].value_counts()
print_and_plot_report("Results of First Global Clustering", global_summary, global_counts)


# --- PART 2: Second-Level Sub-Clustering & Report ---
print("\n" + "="*50)
print("PART 2: Second-Level Sub-Clustering Analysis on Cluster 0")
print("="*50)
# Isolate data and perform sub-clustering
cluster_0_indices = df.index[df['cluster'] == 0]
X_cluster_0_scaled = X_scaled[cluster_0_indices]
dbscan_sub = DBSCAN(eps=SUB_EPS, min_samples=SUB_MIN_SAMPLES)
sub_clusters = dbscan_sub.fit_predict(X_cluster_0_scaled)
# Create temporary sub-cluster labels and add them to the subset of the DataFrame
df.loc[cluster_0_indices, 'sub_cluster'] = ['0_' + str(sc) for sc in sub_clusters]
# Analyze the results of the sub-clustering only
sub_summary = df[df['cluster'] == 0].groupby('sub_cluster')[emotion_columns].mean()
sub_counts = df['sub_cluster'].value_counts()
print_and_plot_report("Results of Second-Level Sub-Clustering on Cluster 0", sub_summary, sub_counts, figure_size=(20, 10)) # Make figure larger for more clusters


# --- PART 3: Semantic Merging & Final Report ---
print("\n" + "="*50)
print("PART 3: Semantic Merging and Final Results Analysis")
print("="*50)
# Integrate results into a single 'final_cluster' column
df['final_cluster'] = df['cluster'].astype(str)
df.loc[cluster_0_indices, 'final_cluster'] = df.loc[cluster_0_indices, 'sub_cluster']
# Define the mapping dictionary
cluster_mapping = {
    '1': 'Lonely', '2': 'Lonely_Missing', '3': 'Sad', '4': 'Sad_Lonely',
    '5': 'Sad_Missing', '6': 'Missing', '7': 'Lonely', '8': 'Sad',
    '-1': 'Ambiguous', '0_-1': 'Ambiguous',
    '0_0': 'Exciting', '0_2': 'Joyful', '0_3': 'Joyful', '0_7': 'Joyful',
    '0_14': 'Joyful', '0_27': 'Joyful', '0_29': 'Joyful', '0_30': 'Joyful',
    '0_32': 'Joyful', '0_34': 'Joyful',
    '0_6': 'Romantic', '0_8': 'Romantic', '0_18': 'Romantic', '0_19': 'Romantic',
    '0_21': 'Romantic', '0_22': 'Romantic', '0_25': 'Romantic', '0_31': 'Romantic',
    '0_9': 'Romantic',
    '0_4': 'Nostalgic', '0_13': 'Nostalgic', '0_16': 'Nostalgic', '0_33': 'Nostalgic',
    '0_10': 'Uplifting_Fresh', '0_11': 'Uplifting_Fresh', '0_15': 'Uplifting_Fresh',
    '0_28': 'Uplifting_Fresh',
    '0_1': 'Peaceful', '0_17': 'Peaceful', '0_23': 'Peaceful', '0_24': 'Peaceful',
    '0_26': 'Peaceful',
    '0_5': 'Touching', '0_12': 'Touching', '0_20': 'Touching'
}
df['final_label'] = df['final_cluster'].map(cluster_mapping)
# Check for any unmapped clusters
unmapped_clusters = df[df['final_label'].isnull()]['final_cluster'].unique()
if len(unmapped_clusters) > 0:
    print(f"\nWARNING: The following detailed clusters were not found in the mapping dictionary: {unmapped_clusters}")
# Analyze the final merged results
final_summary = df.dropna(subset=['final_label']).groupby('final_label')[emotion_columns].mean()
final_counts = df['final_label'].value_counts()
print("\nTo address granularity overload, we performed semantic merging on the detailed clusters.")
print_and_plot_report("Results of Final Merged High-Level Emotion Labels", final_summary, final_counts)

print("\n\nReport finished!")