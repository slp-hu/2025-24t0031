import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import warnings

# -----------------------------------------------------------------------------
# 1. 定义参数
# -----------------------------------------------------------------------------
D_DIM = 64
LEARNING_RATE = 0.001   # (!! 修正 !!) 调回一个“安全”的初始值
EPOCHS = 500            
LAMBDA_REG = 0.0001

# (!! 关键修正：平衡损失权重 !!)
ALPHA_HIERARCHICAL = 100.0  # (新) 提高 HIRE 权重 (100 * 0.0004 ≈ 0.04)
ALPHA_TRIPLET = 1.0       # (新) 降低 Triplet 权重 (1 * 0.04 ≈ 0.04)

VAL_EPOCH_STEP = 10
VAL_MARGIN = 0.2        # (!! 修正 !!) 调回 0.2
TRAIN_TRIPLET_SIZE = 50000 
VAL_TRIPLET_SIZE = 10000   

MATRIX_DIR = r'C:\Users\YAO\Desktop\genre ml\paper\matrices'

# -----------------------------------------------------------------------------
# 2. 辅助函数 (不变)
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._kmeans")
os.environ['OMP_NUM_THREADS'] = '1'

def load_matrix(filename):
    path = os.path.join(MATRIX_DIR, filename)
    m = np.load(path)
    return torch.tensor(m, dtype=torch.float32)

def normalize_T(T_matrix):
    col_sum = T_matrix.sum(dim=0, keepdim=True)
    col_sum[col_sum == 0] = 1.0
    return T_matrix / col_sum

def create_triplets(H1_matrix, num_triplets, set_name="Validation"):
    print(f"\n--- 步骤：创建 {num_triplets} 个 {set_name} 三元组 ---")
    n_clusters, n_songs = H1_matrix.shape
    song_to_cluster_map = H1_matrix.argmax(dim=0)
    cluster_to_songs_map = {}
    for song_id, cluster_id in enumerate(song_to_cluster_map):
        cluster_id = cluster_id.item()
        if cluster_id not in cluster_to_songs_map:
            cluster_to_songs_map[cluster_id] = []
        cluster_to_songs_map[cluster_id].append(song_id)
    print(f"  > 找到了 {n_songs} 首歌 和 {n_clusters} 个簇。")
    triplets = []
    attempts = 0
    max_attempts = num_triplets * 10
    while len(triplets) < num_triplets and attempts < max_attempts:
        attempts += 1
        anchor_song_id = random.randint(0, n_songs - 1)
        anchor_cluster_id = song_to_cluster_map[anchor_song_id].item()
        if len(cluster_to_songs_map.get(anchor_cluster_id, [])) < 2: continue
        positive_song_id = anchor_song_id
        while positive_song_id == anchor_song_id:
            positive_song_id = random.choice(cluster_to_songs_map[anchor_cluster_id])
        negative_cluster_id = anchor_cluster_id
        while negative_cluster_id == anchor_cluster_id:
            negative_cluster_id = random.randint(0, n_clusters - 1)
        if not cluster_to_songs_map.get(negative_cluster_id): continue
        negative_song_id = random.choice(cluster_to_songs_map[negative_cluster_id])
        triplets.append((anchor_song_id, positive_song_id, negative_song_id))
    print(f"  > 成功创建 {len(triplets)} 个 {set_name} 三元组。")
    return torch.tensor(triplets, dtype=torch.long)

def calculate_triplet_loss(V_final_tensor, triplets, margin):
    V_norm = V_final_tensor / (V_final_tensor.norm(dim=0, keepdim=True) + 1e-9)
    A_vecs = V_norm[:, triplets[:, 0]]
    P_vecs = V_norm[:, triplets[:, 1]]
    N_vecs = V_norm[:, triplets[:, 2]]
    cos_sim_AP = (A_vecs * P_vecs).sum(dim=0)
    cos_sim_AN = (A_vecs * N_vecs).sum(dim=0)
    dist_AP = 1.0 - cos_sim_AP
    dist_AN = 1.0 - cos_sim_AN
    loss = torch.relu(dist_AP - dist_AN + margin)
    return loss.mean()

# -----------------------------------------------------------------------------
# 3. 加载、创建(固定)三元组集、初始化 V⁴
# -----------------------------------------------------------------------------
print("--- 步骤一：加载所有“已知”矩阵 ---")
H1 = load_matrix('V1.npy') 
H2 = load_matrix('V2.npy') 
H3 = load_matrix('V3.npy') 
T1, T2, T3 = H1.t(), H2.t(), H3.t()
Q1, Q2, Q3 = normalize_T(T1), normalize_T(T2), normalize_T(T3)
H3_H2, H3_H2_H1 = H3 @ H2, (H3 @ H2) @ H1

val_triplets = create_triplets(H1, VAL_TRIPLET_SIZE, set_name="Validation")
train_triplets = create_triplets(H1, TRAIN_TRIPLET_SIZE, set_name="Training")

print("\n--- 步骤二：初始化 V⁴ (H4) 作为训练参数 ---")
n_genres = H3.shape[0]
H4 = nn.Parameter(torch.randn(D_DIM, n_genres)) 
nn.init.xavier_uniform_(H4)
print(f"H4 (V⁴) shape (d x n_genres): {H4.shape}")

# -----------------------------------------------------------------------------
# 4. (重大修改) 开始训练 (已平衡权重)
# -----------------------------------------------------------------------------
print(f"\n--- 步骤三：开始训练 V⁴ (共 {EPOCHS} 轮) ---")
print(f"Loss = {ALPHA_HIERARCHICAL} * Hierarchical + {ALPHA_TRIPLET} * Triplet")
optimizer = optim.Adam([H4], lr=LEARNING_RATE)
loss_fn_mse = nn.MSELoss()

# (修正) 我们使用 `patience=5` (即 5 * 10 = 50 轮)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

best_val_loss = float('inf')
best_epoch = 0
train_loss_history = []
val_loss_history = []
val_epochs_plotted = []
last_lr = LEARNING_RATE # (新) 用于监控 LR 变化

for epoch in range(EPOCHS):
    H4.requires_grad_(True)
    optimizer.zero_grad()
    
    # --- 损失 1: HIRE 层级损失 ---
    Parent_4, ChildAgg_4 = H4, H4 @ H3 @ Q3
    loss_4 = loss_fn_mse(Parent_4, ChildAgg_4)
    Parent_3, ChildAgg_3 = H4 @ H3, H4 @ H3_H2 @ Q2
    loss_3 = loss_fn_mse(Parent_3, ChildAgg_3)
    Parent_2, ChildAgg_2 = H4 @ H3_H2, H4 @ H3_H2_H1 @ Q1
    loss_2 = loss_fn_mse(Parent_2, ChildAgg_2)
    loss_hierarchical = loss_2 + loss_3 + loss_4

    # --- 损失 2: Triplet 损失 ---
    V_current = H4 @ H3_H2_H1
    loss_triplet = calculate_triplet_loss(V_current, train_triplets, VAL_MARGIN)
    reg_loss = LAMBDA_REG * torch.norm(H4, 2)

    # --- (!! 关键修正：平衡的 总损失 !!) ---
    total_train_loss = (ALPHA_HIERARCHICAL * loss_hierarchical) + \
                       (ALPHA_TRIPLET * loss_triplet) + \
                       reg_loss
    
    total_train_loss.backward()
    optimizer.step()
    
    train_loss_history.append(total_train_loss.item())
    
    # --- 验证步骤 (Validation Step) ---
    if (epoch + 1) % VAL_EPOCH_STEP == 0:
        with torch.no_grad():
            V_current_val = H4 @ H3_H2_H1
            current_val_loss = calculate_triplet_loss(V_current_val, val_triplets, VAL_MARGIN)
        
        val_loss_item = current_val_loss.item()
        val_loss_history.append(val_loss_item)
        val_epochs_plotted.append(epoch + 1)
        
        # --- (!! 关键：调度器“喂食” !!) ---
        scheduler.step(val_loss_item) 
        
        # (新) 手动打印 LR 变化
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f"  > (!!) 学习率已降低: {last_lr} -> {current_lr}")
            last_lr = current_lr
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_train_loss.item():.6f}, Val Loss (Triplet): {val_loss_item:.6f}")
        
        if val_loss_item < best_val_loss:
            best_val_loss = val_loss_item
            best_epoch = epoch + 1
            print(f"  > (新) 最佳模型！已保存到 V4_best.npy。")
            np.save(os.path.join(MATRIX_DIR, 'V4_best.npy'), H4.detach().numpy())

print(f"--- 训练完成！ ---")
print(f"最佳验证损失 {best_val_loss:.6f} 出现在第 {best_epoch} 轮。")

# -----------------------------------------------------------------------------
# 5. 绘制训练/验证曲线 (不变)
# -----------------------------------------------------------------------------
print("\n--- 步骤四：正在绘制损失曲线... ---")
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label='Total Train Loss (Hier + Trip)', color='blue', alpha=0.7)
plt.plot(val_epochs_plotted, val_loss_history, label='Val Loss (Triplet)', color='red', marker='o', linestyle='--')
if best_epoch > 0:
    plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.title('Train vs. Validation Loss (Balanced Weights)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
print("图表窗口已弹出。关闭图表窗口以结束脚本。")
plt.show()

# -----------------------------------------------------------------------------
# 6. 保存最终产物 (不变)
# -----------------------------------------------------------------------------
print("\n--- 步骤五：加载最佳模型并保存最终产物 ---")
try:
    H4_best_np = np.load(os.path.join(MATRIX_DIR, 'V4_best.npy'))
    H4_best = torch.tensor(H4_best_np, dtype=torch.float32)
    print("  > 已成功加载 V4_best.npy。")
except FileNotFoundError:
    print("  > 未找到 V4_best.npy，将使用最后一轮的 H4。")
    H4_best = H4.detach()
    H4_best_np = H4_best.numpy()

print("  > G:\.py.\(V_final = H4_best @ H3 @ H2 @ H1)...")
H3_H2_H1_np = H3_H2_H1.detach().numpy() 
V_final = H4_best_np @ H3_H2_H1_np

np.save(os.path.join(MATRIX_DIR, 'V_final.npy'), V_final)
print(f"  > 已保存 V_final (歌曲特征矩阵) 到: V_final.npy (Shape: {V_final.shape})")
print("\n--- S U C C E S S ! ---")