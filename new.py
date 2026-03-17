import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# -------------------------------
# 1. 自定义 Dataset
# -------------------------------
class MusicEmotionDataset(Dataset):
    def __init__(self, csv_file):
        # 读取 CSV 数据
        self.data = pd.read_csv(csv_file)
        # 目标：arousal, valence （CSV 中第二、第三列）
        self.targets = self.data[['arousal', 'valence']].values.astype(np.float32)
        # 输入特征：从 'avg_pitch' 开始至最后，共17个特征
        self.features = self.data.loc[:, 'avg_pitch':].values.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.targets[idx])
        return x, y

# -------------------------------
# 2. 模型定义
# -------------------------------
# 单个输出的子网络，隐藏层节点数采用公式：hidden_dim = int(sqrt(N_input + 1)) + tau
class emoDNN_SingleOutput(nn.Module):
    def __init__(self, input_dim, tau=4, num_hidden_layers=5, dropout_rate=0.3):
        super(emoDNN_SingleOutput, self).__init__()
        hidden_dim = int(math.sqrt(input_dim + 1)) + tau
        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        # 构建剩余隐藏层
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        # 输出层：单个神经元
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# 两个独立网络分别预测 arousal 和 valence
class emoDNN_TwoNetworks(nn.Module):
    def __init__(self, input_dim, tau=4, num_hidden_layers=5, dropout_rate=0.3):
        super(emoDNN_TwoNetworks, self).__init__()
        self.net_arousal = emoDNN_SingleOutput(input_dim, tau=tau, num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
        self.net_valence = emoDNN_SingleOutput(input_dim, tau=tau, num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    
    def forward(self, x):
        pred_arousal = self.net_arousal(x)  # 输出形状 (batch_size, 1)
        pred_valence = self.net_valence(x)    # 输出形状 (batch_size, 1)
        return pred_arousal, pred_valence

# -------------------------------
# 3. 评价指标：R²
# -------------------------------
def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# -------------------------------
# 4. 训练与验证函数
# -------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss_arousal = 0.0
    total_loss_valence = 0.0
    all_preds_arousal = []
    all_trues_arousal = []
    all_preds_valence = []
    all_trues_valence = []
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        true_arousal = targets[:, 0].unsqueeze(1)
        true_valence = targets[:, 1].unsqueeze(1)
        
        optimizer.zero_grad()
        pred_arousal, pred_valence = model(inputs)
        loss_arousal = criterion(pred_arousal, true_arousal)
        loss_valence = criterion(pred_valence, true_valence)
        loss = loss_arousal + loss_valence
        loss.backward()
        optimizer.step()
        
        total_loss_arousal += loss_arousal.item() * inputs.size(0)
        total_loss_valence += loss_valence.item() * inputs.size(0)
        
        all_preds_arousal.append(pred_arousal.detach())
        all_trues_arousal.append(true_arousal.detach())
        all_preds_valence.append(pred_valence.detach())
        all_trues_valence.append(true_valence.detach())
    
    n = len(dataloader.dataset)
    avg_loss_arousal = total_loss_arousal / n
    avg_loss_valence = total_loss_valence / n
    all_preds_arousal = torch.cat(all_preds_arousal, dim=0)
    all_trues_arousal = torch.cat(all_trues_arousal, dim=0)
    all_preds_valence = torch.cat(all_preds_valence, dim=0)
    all_trues_valence = torch.cat(all_trues_valence, dim=0)
    
    r2_arousal = r2_score(all_trues_arousal, all_preds_arousal).item()
    r2_valence = r2_score(all_trues_valence, all_preds_valence).item()
    
    return avg_loss_arousal, r2_arousal, avg_loss_valence, r2_valence

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss_arousal = 0.0
    total_loss_valence = 0.0
    all_preds_arousal = []
    all_trues_arousal = []
    all_preds_valence = []
    all_trues_valence = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            true_arousal = targets[:, 0].unsqueeze(1)
            true_valence = targets[:, 1].unsqueeze(1)
            
            pred_arousal, pred_valence = model(inputs)
            loss_arousal = criterion(pred_arousal, true_arousal)
            loss_valence = criterion(pred_valence, true_valence)
            total_loss_arousal += loss_arousal.item() * inputs.size(0)
            total_loss_valence += loss_valence.item() * inputs.size(0)
            
            all_preds_arousal.append(pred_arousal)
            all_trues_arousal.append(true_arousal)
            all_preds_valence.append(pred_valence)
            all_trues_valence.append(true_valence)
    
    n = len(dataloader.dataset)
    avg_loss_arousal = total_loss_arousal / n
    avg_loss_valence = total_loss_valence / n
    all_preds_arousal = torch.cat(all_preds_arousal, dim=0)
    all_trues_arousal = torch.cat(all_trues_arousal, dim=0)
    all_preds_valence = torch.cat(all_preds_valence, dim=0)
    all_trues_valence = torch.cat(all_trues_valence, dim=0)
    
    r2_arousal = r2_score(all_trues_arousal, all_preds_arousal).item()
    r2_valence = r2_score(all_trues_valence, all_preds_valence).item()
    
    return avg_loss_arousal, r2_arousal, avg_loss_valence, r2_valence

# -------------------------------
# 5. 主训练流程
# -------------------------------
def main():
    # 超参数设置
    csv_file = "emo_merged.csv"  # 请将此处文件名替换为您的 CSV 文件路径
    num_epochs = 300
    batch_size = 20
    learning_rate = 0.05
    tau = 1
    num_hidden_layers = 5
    dropout_rate = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集
    dataset = MusicEmotionDataset(csv_file)
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = dataset.features.shape[1]
    model = emoDNN_TwoNetworks(input_dim, tau=tau, num_hidden_layers=num_hidden_layers, dropout_rate=dropout_rate)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录每个 epoch 的指标
    history = {
        "train_loss_arousal": [],
        "train_r2_arousal": [],
        "train_loss_valence": [],
        "train_r2_valence": [],
        "val_loss_arousal": [],
        "val_r2_arousal": [],
        "val_loss_valence": [],
        "val_r2_valence": [],
    }
    
    # 训练过程：每个 epoch 在命令行打印结果
    for epoch in range(1, num_epochs + 1):
        train_loss_a, train_r2_a, train_loss_v, train_r2_v = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss_a, val_r2_a, val_loss_v, val_r2_v = eval_epoch(model, val_loader, criterion, device)
        
        history["train_loss_arousal"].append(train_loss_a)
        history["train_r2_arousal"].append(train_r2_a)
        history["train_loss_valence"].append(train_loss_v)
        history["train_r2_valence"].append(train_r2_v)
        history["val_loss_arousal"].append(val_loss_a)
        history["val_r2_arousal"].append(val_r2_a)
        history["val_loss_valence"].append(val_loss_v)
        history["val_r2_valence"].append(val_r2_v)
        
        print(f"Epoch {epoch}/{num_epochs}:")
        print(f"  Arousal - Train Loss: {train_loss_a:.4f}, Train R2: {train_r2_a:.4f} | Val Loss: {val_loss_a:.4f}, Val R2: {val_r2_a:.4f}")
        print(f"  Valence - Train Loss: {train_loss_v:.4f}, Train R2: {train_r2_v:.4f} | Val Loss: {val_loss_v:.4f}, Val R2: {val_r2_v:.4f}")
    
    # 训练结束后，绘制整个训练过程的变化图
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    
    axs[0, 0].plot(history["train_loss_arousal"], label="Train Loss Arousal")
    axs[0, 0].plot(history["val_loss_arousal"], label="Val Loss Arousal")
    axs[0, 0].set_title("Arousal Loss")
    axs[0, 0].legend()
    
    axs[0, 1].plot(history["train_r2_arousal"], label="Train R2 Arousal")
    axs[0, 1].plot(history["val_r2_arousal"], label="Val R2 Arousal")
    axs[0, 1].set_title("Arousal R2")
    axs[0, 1].legend()
    
    axs[1, 0].plot(history["train_loss_valence"], label="Train Loss Valence")
    axs[1, 0].plot(history["val_loss_valence"], label="Val Loss Valence")
    axs[1, 0].set_title("Valence Loss")
    axs[1, 0].legend()
    
    axs[1, 1].plot(history["train_r2_valence"], label="Train R2 Valence")
    axs[1, 1].plot(history["val_r2_valence"], label="Val R2 Valence")
    axs[1, 1].set_title("Valence R2")
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
