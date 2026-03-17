import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import ttk, messagebox
import pygame
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ════════════════════ 配置 ════════════════════
INPUT_FILE = "full_eval_results.npz"
# 更新你的特征文件路径
FEATURE_FILE = r"C:\Users\YAO\Desktop\genre ml\paper\216\audio_features.csv" 
AUDIO_DIR = r"G:\13kmid30s" 

# ════════════════════ 轴解释逻辑 ════════════════════
AXIS_MAPPING = {
    'energy': ('Intensity', 'Soft', 'Powerful'),
    'rms': ('Loudness', 'Quiet', 'Loud'),
    'energy_std': ('Dynamic Range', 'Flat', 'Dynamic'),
    'bpm': ('Tempo', 'Slow', 'Fast'),
    'danceability': ('Groove', 'Still', 'Danceable'),
    'beats_loudness': ('Beat Strength', 'Weak', 'Strong'),
    'onset_rate': ('Note Density', 'Sparse', 'Dense'),
    'spectral_centroid':('Brightness', 'Dark', 'Bright'),
    'hfc': ('High-Frequency', 'Muffled', 'Crisp'),
    'spectral_rolloff': ('Sharpness', 'Warm', 'Sharp'),
    'spectral_flux': ('Timbre Variation', 'Stable', 'Varying'),
    'spectral_entropy': ('Instrumentation', 'Simple', 'Complex'),
    'spectral_flatness_db': ('Sound Texture', 'Pure', 'Noisy'),
    'dissonance': ('Tension', 'Consonant', 'Dissonant'),
    'key_strength': ('Tonal Clarity', 'Ambiguous', 'Clear')
}

class AxisInterpreter:
    def __init__(self, feature_csv_path):
        self.feature_df = pd.read_csv(feature_csv_path)
        self.feature_df['id'] = self.feature_df['id'].astype(str)
        self.feature_cols = [col for col in self.feature_df.columns if col not in ['id', 'error']]
    
    def analyze_axis(self, pc_values, song_ids):
        song_ids_str = [str(sid) for sid in song_ids]
        subset_df = self.feature_df[self.feature_df['id'].isin(song_ids_str)].copy()
        id_to_idx = {str(sid): i for i, sid in enumerate(song_ids)}
        subset_df['_order'] = subset_df['id'].map(id_to_idx)
        subset_df.sort_values('_order', inplace=True)
        valid_indices = subset_df['_order'].astype(int).values
        pc_aligned = pc_values[valid_indices]
        
        correlations = []
        for feat in self.feature_cols:
            vals = subset_df[feat].values
            if len(vals) < 2: continue
            r, _ = pearsonr(pc_aligned, vals)
            mapping = AXIS_MAPPING.get(feat, (feat, 'Low', 'High'))
            correlations.append({'feature': feat, 'label': mapping[0], 'low': mapping[1], 'high': mapping[2], 'r': r, 'abs_r': abs(r)})
        
        correlations.sort(key=lambda x: x['abs_r'], reverse=True)
        return correlations

    def get_best_label(self, correlations):
        if not correlations: return "Unknown", 0, "Low", "High"
        best = correlations[0]
        return best['label'], best['r'], best['low'], best['high']

# ════════════════════ Explorer UI ════════════════════
class ExplorerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eval Set Explorer - Find the Best Demo Case")
        self.root.geometry("1300x850")
        
        pygame.mixer.init()
        
        # 1. Load Full Data
        if not os.path.exists(INPUT_FILE):
            messagebox.showerror("Error", f"Run step1_export_full_eval.py first!")
            root.destroy(); return
            
        print("Loading data...")
        data = np.load(INPUT_FILE, allow_pickle=True)
        self.all_ids = data['ids']
        self.all_embeddings = data['embeddings']
        self.all_scores = data['scores'] # (N, 12)
        self.all_genres = data['genres']
        self.emotion_names = list(data['emotion_names'])
        self.unique_genres = sorted(np.unique(self.all_genres))
        
        # Load Interpreter
        self.interpreter = AxisInterpreter(FEATURE_FILE)
        
        # Layout
        left_panel = ttk.Frame(root, padding=10, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Controls
        ttk.Label(left_panel, text="1. Select Genre:").pack(anchor=tk.W)
        self.genre_combo = ttk.Combobox(left_panel, values=self.unique_genres, state="readonly")
        self.genre_combo.pack(fill=tk.X, pady=5)
        self.genre_combo.set("EDM")
        
        ttk.Label(left_panel, text="2. Select Emotion:").pack(anchor=tk.W, pady=(10,0))
        self.emo_combo = ttk.Combobox(left_panel, values=self.emotion_names, state="readonly")
        self.emo_combo.pack(fill=tk.X, pady=5)
        self.emo_combo.set("Excitement")
        
        ttk.Button(left_panel, text="Run Analysis", command=self.run_analysis).pack(fill=tk.X, pady=20)
        
        self.lbl_stats = ttk.Label(left_panel, text="")
        self.lbl_stats.pack(anchor=tk.W)
        
        # Player
        ttk.Separator(left_panel, orient='horizontal').pack(fill='x', pady=20)
        self.lbl_info = ttk.Label(left_panel, text="Click point to play", wraplength=200)
        self.lbl_info.pack(anchor=tk.W)
        ttk.Button(left_panel, text="Stop", command=self.stop_music).pack(fill=tk.X, pady=5)

        # Plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # State
        self.current_ids = None
        self.current_coords = None
        self.current_ranks = None
        
        self.run_analysis() # Run default

    def run_analysis(self):
        target_genre = self.genre_combo.get()
        target_emo = self.emo_combo.get()
        target_emo_idx = self.emotion_names.index(target_emo)
        
        # 1. Filter by Genre
        indices = np.where(self.all_genres == target_genre)[0]
        if len(indices) < 10:
            messagebox.showwarning("Warning", f"Not enough songs for {target_genre} (Found {len(indices)})")
            return
            
        # 2. Get Scores for target emotion
        genre_scores = self.all_scores[indices, target_emo_idx]
        genre_ids = self.all_ids[indices]
        genre_embs = self.all_embeddings[indices]
        
        # 3. Sort Top 40
        top_indices = np.argsort(genre_scores)[::-1][:40] # Top 40 indices local to genre subset
        
        self.current_ids = genre_ids[top_indices]
        subset_embs = genre_embs[top_indices]
        self.current_ranks = np.arange(1, len(top_indices)+1)
        
        # 4. Draw
        self.draw_plot(subset_embs, target_genre, target_emo)
        
    def draw_plot(self, embeddings, genre, emotion):
        self.ax.clear()
        
        # PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        self.current_coords = coords
        
        # Axis Interpretation
        x_corr = self.interpreter.analyze_axis(coords[:, 0], self.current_ids)
        y_corr = self.interpreter.analyze_axis(coords[:, 1], self.current_ids)
        xl, _, xlo, xhi = self.interpreter.get_best_label(x_corr)
        yl, _, ylo, yhi = self.interpreter.get_best_label(y_corr)
        
        # Plot
        top10 = self.current_ranks <= 10
        rest = self.current_ranks > 10
        
        self.ax.scatter(coords[rest, 0], coords[rest, 1], c='#9b59b6', alpha=0.6, s=80, label='Top 11-40')
        self.ax.scatter(coords[top10, 0], coords[top10, 1], c='#f1c40f', alpha=1.0, s=200, edgecolors='k', label='Top 1-10')
        
        for i in range(len(self.current_ids)):
            if self.current_ranks[i] <= 10:
                self.ax.text(coords[i, 0], coords[i, 1], str(self.current_ranks[i]), 
                             ha='center', va='center', fontweight='bold')

        self.ax.set_title(f"Top-40: {genre} + {emotion} (Eval Set)", fontsize=14)
        # Use Dynamic Labels but add (Primary PC)
        self.ax.set_xlabel(f"{xl} ({xlo} <-> {xhi})\n(Primary PC Correlation)", fontsize=10, fontweight='bold')
        self.ax.set_ylabel(f"{yl} ({ylo} <-> {yhi})\n(Secondary PC Correlation)", fontsize=10, fontweight='bold')
        
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.legend()
        self.canvas.draw()
        
    def on_click(self, event):
        if event.inaxes != self.ax or self.current_coords is None: return
        click_pos = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(self.current_coords - click_pos, axis=1)
        idx = np.argmin(dists)
        if dists[idx] < 1.0:
            self.play_music(self.current_ids[idx], self.current_ranks[idx])

    def play_music(self, song_id, rank):
        path = os.path.join(AUDIO_DIR, f"{song_id}.mp3")
        if not os.path.exists(path): path = os.path.join(AUDIO_DIR, f"{song_id}.wav")
        
        if os.path.exists(path):
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            self.lbl_info.config(text=f"Playing Rank #{rank}\nID: {song_id}", foreground="green")
        else:
            self.lbl_info.config(text=f"Missing File: {song_id}", foreground="red")

    def stop_music(self):
        pygame.mixer.music.stop()

if __name__ == "__main__":
    root = tk.Tk()
    app = ExplorerApp(root)
    root.mainloop()