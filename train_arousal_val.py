import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# -------------------------------
# 数据集定义（目标：arousal）
# -------------------------------
class MusicEmotionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # 特征：从 avg_pitch 开始到最后
        self.features = self.data.loc[:, 'avg_pitch':].values.astype(np.float32)
        # 目标：arousal
        self.targets = self.data['arousal'].values.astype(np.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.targets[idx])
        return x, y

# -------------------------------
# 模型结构（单输出）
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
# 评价指标：R²
# -------------------------------
def r2_score(y_true, y_pred):
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

# -------------------------------
# 训练函数
# -------------------------------
def train(model, train_loader, val_loader, device, num_epochs=200, early_stop_enabled=True):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if early_stop_enabled:
        early_stopping = EarlyStopping(patience=15)
    
    history = {"train_loss": [], "train_r2": [], "val_loss": [], "val_r2": []}

    for epoch in range(1, num_epochs + 1):
        # -------- 训练阶段 --------
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

        # -------- 验证阶段 --------
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

        # 记录
        history["train_loss"].append(train_loss)
        history["train_r2"].append(train_r2)
        history["val_loss"].append(val_loss)
        history["val_r2"].append(val_r2)

        print(f"Epoch {epoch}: "
              f"Train Loss {train_loss:.4f} R2 {train_r2:.4f} | "
              f"Val Loss {val_loss:.4f} R2 {val_r2:.4f}")

        # 早停检查
        if early_stop_enabled:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("早停触发，停止训练！")
                break

    return history

# -------------------------------
# (新增) 验证集上推理并导出最优/最差案例
# -------------------------------
def inference_and_export(model, val_ds, val_loader, device, out_csv="val_best_worst.csv"):
    """
    在验证集上跑推理，记录每首曲子的预测值和真实值、误差；
    选出最好的 5 首和最差的 5 首，导出到 CSV。
    """
    model.eval()
    
    results = []
    val_index = 0  # 记录在验证集里的顺序索引
    criterion = nn.MSELoss(reduction='none')  # 用于计算逐样本 MSE
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            # preds, y_batch 均为 [batch_size, 1]
            # 逐条记录
            batch_mse = criterion(preds, y_batch).cpu().numpy().flatten()  # shape=[batch_size]
            
            for i in range(x_batch.size(0)):
                pred_val = preds[i].item()
                true_val = y_batch[i].item()
                error = batch_mse[i]  # 即 (pred - true)^2
                results.append({
                    "val_index": val_index,
                    "pred": pred_val,
                    "true": true_val,
                    "error": error
                })
                val_index += 1
    
    df = pd.DataFrame(results)
    # 根据 error 从小到大排序
    df_sorted = df.sort_values("error", ascending=True).reset_index(drop=True)
    best_5 = df_sorted.head(5).copy()
    worst_5 = df_sorted.tail(5).copy()

    # 添加一个列标注
    best_5["rank_type"] = "best"
    worst_5["rank_type"] = "worst"

    # 合并并导出
    df_out = pd.concat([best_5, worst_5], ignore_index=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[Info] 已导出验证集中预测最好的 5 条和最差的 5 条样本到: {out_csv}")

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
    num_hidden_layers = 5
    dropout_rate = 0

    # 早停开关
    use_early_stopping = False

    # 1) 加载数据并划分训练 / 验证
    dataset = MusicEmotionDataset(csv_file)
    n_train = int(0.7 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2) 构建模型并训练
    input_dim = dataset.features.shape[1]
    model = emoDNN_SingleOutput(
        input_dim=input_dim,
        tau=tau,
        num_hidden_layers=num_hidden_layers,
        dropout_rate=dropout_rate
    ).to(device)

    history = train(model, train_loader, val_loader, device,
                    num_epochs=num_epochs, early_stop_enabled=use_early_stopping)

    # 3) 可视化训练过程
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Arousal Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_r2"], label="Train R²")
    plt.plot(history["val_r2"], label="Val R²")
    plt.title("Arousal R² Score")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 4) 预测验证集并导出最优/最差的 5 条记录
    inference_and_export(model, val_ds, val_loader, device, out_csv="arousal_val_best_worst.csv")


if __name__ == "__main__":
    main()
