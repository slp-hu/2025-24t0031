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
# 必须与 Step 1 生成的文件名一致
INPUT_FILE = "top40_for_ui_eval.npz"
# 更新为你提供的路径
FEATURE_FILE = r"C:\Users\YAO\Desktop\genre ml\paper\216\audio_features.csv" 
AUDIO_DIR = r"G:\13kmid30s"         # 音频目录

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
        # Ensure order matches
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

# ════════════════════ Top 40 UI ════════════════════
class Top40App:
    def __init__(self, root):
        self.root = root
        self.root.title("Top-40 Recommendation Explorer (Eval Set Mode)")
        self.root.geometry("1200x800")
        
        # Audio
        pygame.mixer.init()
        self.is_playing = False
        
        # Load Data
        if not os.path.exists(INPUT_FILE):
            messagebox.showerror("Error", f"Run step1_export_top40_eval.py first!")
            root.destroy()
            return
            
        data = np.load(INPUT_FILE)
        self.ids = data['ids']
        self.embeddings = data['embeddings']
        self.ranks = data['ranks'] # 1 to 40
        self.genre = str(data['genre'])
        self.emotion = str(data['emotion'])
        
        # Load Interpreter
        if not os.path.exists(FEATURE_FILE):
             messagebox.showerror("Error", f"Feature file not found: {FEATURE_FILE}")
             root.destroy()
             return
        self.interpreter = AxisInterpreter(FEATURE_FILE)
        
        # Layout
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Plot Area
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Control Area
        ctrl_frame = ttk.Frame(main_frame, padding=10)
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.lbl_info = ttk.Label(ctrl_frame, text="Click a point to play...", font=('Arial', 12))
        self.lbl_info.pack(side=tk.LEFT)
        
        ttk.Button(ctrl_frame, text="Stop Music", command=self.stop_music).pack(side=tk.RIGHT)
        
        # Draw immediately
        self.draw_plot()
        
    def draw_plot(self):
        self.ax.clear()
        
        # 1. PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(self.embeddings)
        self.coords = coords
        
        # 2. Axis Interpretation
        x_corr = self.interpreter.analyze_axis(coords[:, 0], self.ids)
        y_corr = self.interpreter.analyze_axis(coords[:, 1], self.ids)
        
        xl, xr, xlo, xhi = self.interpreter.get_best_label(x_corr)
        yl, yr, ylo, yhi = self.interpreter.get_best_label(y_corr)
        
        # 3. Plot Points
        # Split into Top 10 and Rest
        top10_mask = self.ranks <= 10
        rest_mask = self.ranks > 10
        
        # Plot Rest (Rank 11-40) - Smaller, Purple
        self.ax.scatter(coords[rest_mask, 0], coords[rest_mask, 1], 
                        c='#9b59b6', alpha=0.6, s=80, label='Top 11-40 (Hidden in List)')
        
        # Plot Top 10 (Rank 1-10) - Larger, Gold/Orange
        self.ax.scatter(coords[top10_mask, 0], coords[top10_mask, 1], 
                        c='#f1c40f', alpha=1.0, s=200, edgecolors='black', label='Top 1-10 (Visible in List)')
        
        # 4. Add Rank Numbers to Top 10
        for i in range(len(self.ids)):
            rank = self.ranks[i]
            if rank <= 10:
                self.ax.text(coords[i, 0], coords[i, 1], str(rank), 
                             fontsize=10, ha='center', va='center', fontweight='bold', color='black')
        
        # 5. Labels and Titles
        self.ax.set_title(f"Top-40 Recommendations for '{self.genre} + {self.emotion}' (Eval Set)\nList View limits you to Top-10 (Yellow). Map reveals alternatives (Purple).", fontsize=12)
        
        # X Axis Label
        self.ax.set_xlabel(f"{xl} ({xlo} <-> {xhi})", fontsize=10, fontweight='bold')
        # Y Axis Label
        self.ax.set_ylabel(f"{yl} ({ylo} <-> {yhi})", fontsize=10, fontweight='bold')
        
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.legend()
        self.canvas.draw()
        
    def on_click(self, event):
        if event.inaxes != self.ax: return
        
        # Find nearest point
        click_pos = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(self.coords - click_pos, axis=1)
        nearest_idx = np.argmin(dists)
        
        if dists[nearest_idx] < 1.0: # Tolerance
            song_id = self.ids[nearest_idx]
            rank = self.ranks[nearest_idx]
            self.play_music(song_id, rank)
            
    def play_music(self, song_id, rank):
        # Find file
        path = os.path.join(AUDIO_DIR, f"{song_id}.mp3") # Try mp3
        if not os.path.exists(path):
            path = os.path.join(AUDIO_DIR, f"{song_id}.wav") # Try wav
        
        if os.path.exists(path):
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            self.lbl_info.config(text=f"Playing Rank #{rank} (ID: {song_id})", foreground="green")
        else:
            self.lbl_info.config(text=f"File not found: {song_id}", foreground="red")

    def stop_music(self):
        pygame.mixer.music.stop()
        self.lbl_info.config(text="Stopped", foreground="black")

if __name__ == "__main__":
    root = tk.Tk()
    app = Top40App(root)
    root.mainloop()