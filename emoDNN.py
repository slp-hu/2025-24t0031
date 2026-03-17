import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 自定义数据集，加载 CSV 数据
class MusicEmotionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # CSV 中：第一列 song_id，第二、三列为目标（valence_mean, arousal_mean），后面为音频特征
        # 假设在预处理时已将目标值映射到 [-1,1]
        self.features = self.data.iloc[:, 3:].values.astype(np.float32)
        self.targets = self.data.iloc[:, 1:3].values.astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return torch.tensor(x), torch.tensor(y)

# 分别计算 valence 和 arousal 的准确率（按设定阈值判断每个维度是否正确）
def separate_accuracy_batch(predictions, targets, threshold=0.1):
    # predictions, targets: shape (batch_size, 2)
    correct_valence = (torch.abs(predictions[:, 0] - targets[:, 0]) < threshold)
    correct_arousal = (torch.abs(predictions[:, 1] - targets[:, 1]) < threshold)
    return correct_valence.sum().item(), correct_arousal.sum().item()

# 构建 emoDNN 模型：两层全连接网络
class EmoDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, output_dim=2):
        super(EmoDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

# 训练模型，记录每个 epoch 的训练和验证指标，并打印每个 epoch 的结果
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.01, thresh=0.1):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loss_history = []
    val_loss_history = []
    train_valence_acc_history = []
    train_arousal_acc_history = []
    val_valence_acc_history = []
    val_arousal_acc_history = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        valence_correct = 0
        arousal_correct = 0
        total_samples = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            c_valence, c_arousal = separate_accuracy_batch(outputs, targets, threshold=thresh)
            valence_correct += c_valence
            arousal_correct += c_arousal
            total_samples += batch_size
        
        train_loss = running_loss / total_samples
        train_valence_acc = valence_correct / total_samples
        train_arousal_acc = arousal_correct / total_samples
        
        train_loss_history.append(train_loss)
        train_valence_acc_history.append(train_valence_acc)
        train_arousal_acc_history.append(train_arousal_acc)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        valence_correct = 0
        arousal_correct = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                batch_size = inputs.size(0)
                val_running_loss += loss.item() * batch_size
                c_valence, c_arousal = separate_accuracy_batch(outputs, targets, threshold=thresh)
                valence_correct += c_valence
                arousal_correct += c_arousal
                total_val += batch_size
        
        val_loss = val_running_loss / total_val
        val_valence_acc = valence_correct / total_val
        val_arousal_acc = arousal_correct / total_val
        
        val_loss_history.append(val_loss)
        val_valence_acc_history.append(val_valence_acc)
        val_arousal_acc_history.append(val_arousal_acc)
        
        # 每个 epoch 都打印结果
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Valence Acc: {train_valence_acc:.4f}, Train Arousal Acc: {train_arousal_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Valence Acc: {val_valence_acc:.4f}, Val Arousal Acc: {val_arousal_acc:.4f}")
    
    return (train_loss_history, val_loss_history, 
            train_valence_acc_history, train_arousal_acc_history,
            val_valence_acc_history, val_arousal_acc_history)

def main():
    csv_file = r"C:\Users\YAO\Desktop\genre ml\final_merged3.csv"  # 请确保文件路径正确
    dataset = MusicEmotionDataset(csv_file)
    
    # 按 80% 训练、20% 验证划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    input_dim = dataset.features.shape[1]
    model = EmoDNN(input_dim)
    
    num_epochs = 100
    learning_rate = 0.01
    thresh = 0.1  # 根据目标归一化后的范围[-1,1]选择合适的阈值
    (train_loss_history, val_loss_history, 
     train_valence_acc_history, train_arousal_acc_history,
     val_valence_acc_history, val_arousal_acc_history) = train_model(
        model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate, thresh=thresh)
    
    epochs = range(1, num_epochs+1)
    
    # 使用 subplot 将三个图绘制在同一个窗口里
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 子图1：Loss 曲线（训练与验证）
    axs[0].plot(epochs, train_loss_history, label="Train Loss")
    axs[0].plot(epochs, val_loss_history, label="Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("MSE Loss")
    axs[0].set_title("Loss Curve")
    axs[0].legend()
    
    # 子图2：Valence 准确率曲线
    axs[1].plot(epochs, train_valence_acc_history, label="Train Valence Acc")
    axs[1].plot(epochs, val_valence_acc_history, label="Validation Valence Acc")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_title("Valence Accuracy Curve")
    axs[1].legend()
    
    # 子图3：Arousal 准确率曲线
    axs[2].plot(epochs, train_arousal_acc_history, label="Train Arousal Acc")
    axs[2].plot(epochs, val_arousal_acc_history, label="Validation Arousal Acc")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_title("Arousal Accuracy Curve")
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
