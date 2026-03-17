import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# -------------------------------
# 数据集定义（目标为 valence）
# -------------------------------
class MusicEmotionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.loc[:, 'avg_pitch':].values.astype(np.float32)
        self.targets = self.data['valence'].values.astype(np.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.targets[idx])
        return x, y

# -------------------------------
# 模型结构
# -------------------------------
class emoDNN_SingleOutput(nn.Module):
    def __init__(self, input_dim, tau=4, num_hidden_layers=5, dropout_rate=0.3):
        super().__init__()
        hidden_dim = int(math.sqrt(input_dim + 1)) + tau
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# -------------------------------
# EarlyStopping 类
# -------------------------------
class EarlyStopping:
    def __init__(self, patience=10, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# -------------------------------
# 评价指标
# -------------------------------
def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# -------------------------------
# 训练函数（增加 early_stop_enabled 参数）
# -------------------------------
def train(model, train_loader, val_loader, device, num_epochs=200, early_stop_enabled=True):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if early_stop_enabled:
        early_stopping = EarlyStopping(patience=15)
    
    history = {"train_loss": [], "train_r2": [], "val_loss": [], "val_r2": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, preds, trues = 0, [], []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            preds.append(out.detach())
            trues.append(y.detach())
        train_loss = total_loss / len(train_loader.dataset)
        train_r2 = r2_score(torch.cat(trues), torch.cat(preds)).item()

        model.eval()
        total_val_loss, val_preds, val_trues = 0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_val_loss += loss.item() * x.size(0)
                val_preds.append(out)
                val_trues.append(y)
        val_loss = total_val_loss / len(val_loader.dataset)
        val_r2 = r2_score(torch.cat(val_trues), torch.cat(val_preds)).item()

        history["train_loss"].append(train_loss)
        history["train_r2"].append(train_r2)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)

        print(f"Epoch {epoch}: Train Loss {train_loss:.4f} R2 {train_r2:.4f} | Val Loss {val_loss:.4f} R2 {val_r2:.4f}")

        if early_stop_enabled:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("早停触发，停止训练！")
                break

    return history

# -------------------------------
# 主流程
# -------------------------------
def main():
    csv_file = "emo_merged.csv"  # 请替换为你的数据文件路径
    batch_size = 20
    num_epochs = 300
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数设置
    tau = 7
    num_hidden_layers = 4
    dropout_rate = 0

    # 早停开关（True：启用早停，False：关闭早停）
    use_early_stopping = True

    # 加载数据集
    dataset = MusicEmotionDataset(csv_file)
    n_train = int(0.7 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    input_dim = dataset.features.shape[1]
    model = emoDNN_SingleOutput(
        input_dim=input_dim,
        tau=tau,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate
    ).to(device)

    history = train(model, train_loader, val_loader, device, num_epochs=num_epochs, early_stop_enabled=use_early_stopping)

    # 可视化：Loss 和 R²
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Valence Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_r2"], label="Train R²")
    plt.plot(history["val_r2"], label="Val R²")
    plt.title("Valence R² Score")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.ylim(-0.1, 0.6)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
