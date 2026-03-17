import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox
import pygame  # 用于音频播放
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ═══════════ 配置 ═══════════
DATA_FILE = "music_data.npz"

# 【请确认】你的音频文件存放目录
AUDIO_DIR = r"G:\13kmid30s" 

class ClusterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Cluster Explorer & Player (Interactive)")
        self.root.geometry("1200x850") 

        # 0. 初始化音频引擎
        pygame.mixer.init()
        self.is_playing = False
        self.is_paused = False
        self.current_song_length = 30.0 
        self.update_job = None 

        # 1. 加载数据
        try:
            print("Loading data...")
            data = np.load(DATA_FILE)
            self.X = data['embeddings']
            self.genres = data['genres']
            self.emotions = data['emotions']
            self.ids = data['ids']
            self.unique_genres = sorted(np.unique(self.genres))
            self.unique_emotions = sorted(np.unique(self.emotions))
            print(f"Loaded {len(self.X)} songs.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load {DATA_FILE}\nPlease run step1_export_data.py first.\n{e}")
            root.destroy()
            return

        # ─── 布局 ───
        left_panel = ttk.Frame(root, padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ─── 左侧控制 ───
        ttk.Label(left_panel, text="1. Select Genre:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.genre_combo = ttk.Combobox(left_panel, values=self.unique_genres, state="readonly")
        self.genre_combo.pack(fill=tk.X, pady=(0, 15))
        self.genre_combo.current(0)

        ttk.Label(left_panel, text="2. Select Emotion:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.emotion_combo = ttk.Combobox(left_panel, values=self.unique_emotions, state="readonly")
        self.emotion_combo.pack(fill=tk.X, pady=(0, 15))
        self.emotion_combo.current(0)

        ttk.Label(left_panel, text="3. Clustering Strategy:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.auto_k_var = tk.BooleanVar(value=True)
        self.chk_auto = ttk.Checkbutton(left_panel, text="Auto-detect Best K", variable=self.auto_k_var, command=self.toggle_spinbox)
        self.chk_auto.pack(anchor=tk.W)

        self.k_frame = ttk.Frame(left_panel)
        self.k_frame.pack(anchor=tk.W, pady=(5, 15))
        ttk.Label(self.k_frame, text="Manual K:").pack(side=tk.LEFT)
        self.k_spin = ttk.Spinbox(self.k_frame, from_=2, to=8, width=5)
        self.k_spin.set(3)
        self.k_spin.pack(side=tk.LEFT, padx=5)
        self.toggle_spinbox()

        self.btn_run = ttk.Button(left_panel, text="Run Analysis", command=self.run_analysis)
        self.btn_run.pack(fill=tk.X, pady=(10, 10))

        ttk.Label(left_panel, text="Analysis Results:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.info_text = tk.Text(left_panel, height=20, width=35, wrap=tk.WORD, bg="#f0f0f0")
        self.info_text.pack(fill=tk.BOTH, expand=True)

        # ─── 底部播放器 ───
        player_frame = ttk.LabelFrame(left_panel, text="Music Player", padding="10")
        player_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

        self.lbl_song_info = ttk.Label(player_frame, text="Click a point to play", font=('Arial', 9, 'bold'), foreground="#333", wraplength=200)
        self.lbl_song_info.pack(fill=tk.X, pady=(0, 5))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(player_frame, variable=self.progress_var, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_seek)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        btn_frame = ttk.Frame(player_frame)
        btn_frame.pack(fill=tk.X)
        self.btn_play = ttk.Button(btn_frame, text="▶ Play", command=self.toggle_play, state="disabled")
        self.btn_play.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        self.btn_stop = ttk.Button(btn_frame, text="■ Stop", command=self.stop_play, state="disabled")
        self.btn_stop.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=2)

        # ─── 图表初始化 ───
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        self.current_coords = None
        self.current_ids = None

    def toggle_spinbox(self):
        if self.auto_k_var.get():
            self.k_spin.state(['disabled'])
        else:
            self.k_spin.state(['!disabled'])

    def run_analysis(self):
        selected_genre = self.genre_combo.get()
        selected_emotion = self.emotion_combo.get()
        
        mask = (self.genres == selected_genre) & (self.emotions == selected_emotion)
        subset_X = self.X[mask]
        subset_ids = self.ids[mask]

        self.info_text.delete(1.0, tk.END)
        self.ax.clear()

        if len(subset_X) < 3:
            self.info_text.insert(tk.END, f"Too few songs ({len(subset_X)}). Need at least 3.")
            self.canvas.draw()
            return

        self.info_text.insert(tk.END, f"Dataset: {selected_genre} + {selected_emotion}\nTotal: {len(subset_X)}\n{'-'*30}\n")

        # ─── K-Means 逻辑 (核心修改处) ───
        final_k = 3
        best_labels = None
        best_kmeans = None

        if self.auto_k_var.get():
            self.info_text.insert(tk.END, "Auto-detecting K (Silhouette):\n")
            best_score = -1
            max_possible_k = min(7, len(subset_X))
            
            # 循环计算每一个 K
            for k in range(2, max_possible_k):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(subset_X)
                score = silhouette_score(subset_X, labels)
                
                # 【修改点】打印每一行的分数
                self.info_text.insert(tk.END, f"K={k}: Score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    final_k = k
                    best_kmeans = kmeans
                    best_labels = labels
                    # 【修改点】如果是当前最佳，加个标记
                    self.info_text.insert(tk.END, " [*]\n") 
                else:
                    self.info_text.insert(tk.END, "\n")
            
            self.info_text.insert(tk.END, f"\nSelected Optimal K = {final_k}\n")
        else:
            final_k = int(self.k_spin.get())
            if final_k >= len(subset_X): final_k = len(subset_X) - 1
            best_kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
            best_labels = best_kmeans.fit_predict(subset_X)
            score = silhouette_score(subset_X, best_labels) # 手动模式也算一下分
            self.info_text.insert(tk.END, f"Manual K={final_k}\nScore={score:.4f}\n")

        # ─── PCA & 绘图 ───
        pca = PCA(n_components=2)
        coords = pca.fit_transform(subset_X)
        
        self.current_coords = coords
        self.current_ids = subset_ids

        self.ax.scatter(coords[:, 0], coords[:, 1], c=best_labels, cmap='viridis', s=100, alpha=0.7, edgecolors='w', picker=True)
        self.ax.set_title(f"{selected_genre} - {selected_emotion}\n(Click any point to Play)", fontsize=10)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.ax.set_xlabel("PCA Dim 1")
        self.ax.set_ylabel("PCA Dim 2")

        # 标注重心
        centers = best_kmeans.cluster_centers_
        from sklearn.metrics import pairwise_distances_argmin_min
        closest, _ = pairwise_distances_argmin_min(centers, subset_X)
        
        self.info_text.insert(tk.END, f"\nRepresentatives:\n")
        for i, idx in enumerate(closest):
            song_id = subset_ids[idx]
            self.info_text.insert(tk.END, f"C{i+1}: {song_id}\n")
            self.ax.scatter(coords[idx, 0], coords[idx, 1], c='red', marker='*', s=300, edgecolors='white', linewidth=1.5, zorder=10)
            self.ax.text(coords[idx, 0], coords[idx, 1], f"C{i+1}", fontsize=9, ha='center', va='bottom', fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        self.canvas.draw()

    # ─── 播放逻辑 ───
    def on_plot_click(self, event):
        if event.inaxes != self.ax or self.current_coords is None: return
        click_point = np.array([event.xdata, event.ydata])
        distances = np.linalg.norm(self.current_coords - click_point, axis=1)
        nearest_idx = np.argmin(distances)
        if distances[nearest_idx] < 0.8: 
            song_id = self.current_ids[nearest_idx]
            self.play_music(song_id)

    def play_music(self, song_id):
        file_path = None
        for ext in ['.mp3', '.wav']:
            path = os.path.join(AUDIO_DIR, f"{song_id}{ext}")
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            self.lbl_song_info.config(text=f"File not found: {song_id}", foreground="red")
            return

        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            self.is_playing = True
            self.is_paused = False
            self.btn_play.config(text="⏸ Pause", state="normal")
            self.btn_stop.config(state="normal")
            self.lbl_song_info.config(text=f"Playing: {song_id}", foreground="green")
            self.update_progress()
        except Exception as e:
            messagebox.showerror("Playback Error", str(e))
            self.stop_play()

    def toggle_play(self):
        if not self.is_playing: return
        if self.is_paused:
            pygame.mixer.music.unpause()
            self.btn_play.config(text="⏸ Pause")
            self.is_paused = False
        else:
            pygame.mixer.music.pause()
            self.btn_play.config(text="▶ Play")
            self.is_paused = True

    def stop_play(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.is_paused = False
        self.btn_play.config(text="▶ Play", state="disabled")
        self.btn_stop.config(state="disabled")
        self.lbl_song_info.config(text="Stopped", foreground="black")
        self.progress_var.set(0)

    def on_seek(self, value):
        if self.is_playing:
            try:
                target_sec = float(value) / 100 * self.current_song_length
                pygame.mixer.music.set_pos(target_sec) 
            except: pass 

    def update_progress(self):
        if self.is_playing and not self.is_paused:
            current_ms = pygame.mixer.music.get_pos()
            if current_ms == -1: 
                if not pygame.mixer.music.get_busy():
                    self.stop_play()
                    return
            current_sec = current_ms / 1000.0
            percent = (current_sec / self.current_song_length) * 100
            if percent > 100: percent = 100
            self.progress_var.set(percent)
        if self.is_playing:
            self.root.after(500, self.update_progress)

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusterApp(root)
    root.mainloop()