import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard
import librosa  # 如果未使用，可移除

from sklearn.metrics import classification_report, confusion_matrix  # 确保导入这些模块

class AudioFeatureWithGenreDataset(Dataset):
    def __init__(self, dataframe, feature_columns, label_column, scaler=None, augment=False):
        """
        初始化 Dataset。

        参数:
            dataframe (pd.DataFrame): 数据框，包含特征和标签。
            feature_columns (list): 特征列的名称列表。
            label_column (str): 标签列的名称。
            scaler (StandardScaler, optional): 用于标准化的 Scaler 对象。如果提供，则应用于特征。
            augment (bool): 是否应用数据增强。
        """
        self.augment = augment
        self.features = dataframe[feature_columns].values.astype("float32")
        self.labels = dataframe[label_column].values.astype("int64")
        self.scaler = scaler
        if scaler:
            self.features = scaler.transform(self.features).astype("float32")  # 确保转换回 float32
        else:
            self.features = self.features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature, label = self.features[idx], self.labels[idx]

        if self.augment:
            feature = self.apply_augmentation(feature)

        # 确保特征仍为 float32
        feature = feature.astype("float32")

        # 转换为 torch.float32 张量
        feature = torch.from_numpy(feature).float()
        label = torch.tensor(label, dtype=torch.long)

        return feature, label

    def apply_augmentation(self, feature):
        """
        应用数据增强到特征向量。

        参数:
            feature (np.ndarray): 原始特征向量。

        返回:
            np.ndarray: 增强后的特征向量。
        """
        try:
            # 添加随机噪声
            feature = self.add_noise(feature, noise_factor=0.01)

            # 随机缩放
            feature = self.random_scaling(feature, scale_range=(0.9, 1.1))

            # 随机遮盖部分特征
            feature = self.random_feature_masking(feature, mask_ratio=0.1)
        except Exception as e:
            print(f"在应用数据增强时出错: {e}")
        return feature

    def add_noise(self, feature, noise_factor=0.01):
        """
        向特征向量中添加随机噪声。

        参数:
            feature (np.ndarray): 原始特征向量。
            noise_factor (float): 噪声因子。

        返回:
            np.ndarray: 添加噪声后的特征向量。
        """
        noise = np.random.normal(0, noise_factor, feature.shape).astype("float32")
        return feature + noise

    def random_scaling(self, feature, scale_range=(0.9, 1.1)):
        """
        随机缩放特征向量。

        参数:
            feature (np.ndarray): 原始特征向量。
            scale_range (tuple): 缩放比例范围。

        返回:
            np.ndarray: 缩放后的特征向量。
        """
        scale = np.float32(np.random.uniform(*scale_range))  # 使用 numpy.float32 转换标量
        return feature * scale

    def random_feature_masking(self, feature, mask_ratio=0.1):
        """
        随机遮盖部分特征值。

        参数:
            feature (np.ndarray): 原始特征向量。
            mask_ratio (float): 遮盖比例。

        返回:
            np.ndarray: 遮盖后的特征向量。
        """
        num_features = len(feature)
        num_mask = int(num_features * mask_ratio)
        if num_mask > 0:
            mask_indices = np.random.choice(num_features, num_mask, replace=False)
            feature[mask_indices] = 0.0
        return feature

class AudioGenreClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train(model, train_loader, val_loader, label_encoder, num_epochs=10, lr=5e-4, device="cpu"):
    """
    训练模型。

    参数:
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        val_loader (DataLoader): 验证数据加载器。
        label_encoder (LabelEncoder): 用于解码标签的 LabelEncoder 对象。
        num_epochs (int): 训练的轮数。
        lr (float): 学习率。
        device (str): 设备类型（'cpu' 或 'cuda'）。
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 初始化 TensorBoard SummaryWriter
    writer = SummaryWriter('runs/genre_classification')

    best_val_acc = 0.0

    # 初始化列表以记录指标
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            # 确保 features 的 dtype 是 float32
            assert features.dtype == torch.float32, f"Expected features dtype torch.float32 but got {features.dtype}"

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算训练准确率
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_features = val_features.to(device)
                val_labels   = val_labels.to(device)

                # 确保 val_features 的 dtype 是 float32
                assert val_features.dtype == torch.float32, f"Expected val_features dtype torch.float32 but got {val_features.dtype}"

                val_logits = model(val_features)
                preds = torch.argmax(val_logits, dim=1)
                correct_val += (preds == val_labels).sum().item()
                total_val   += val_labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_labels.cpu().numpy())

        val_acc = correct_val / total_val if total_val > 0 else 0.0

        # 记录指标
        train_losses.append(epoch_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # 写入 TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        # # 生成分类报告和混淆矩阵（每10个epoch或最后一个epoch）
        # if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
        #     print("Classification Report:")
        #     print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

        #     print("Confusion Matrix:")
        #     print(confusion_matrix(all_labels, all_preds))

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_audio_genre_classifier.pth')
            print(f"--> 保存最佳模型，验证准确率: {best_val_acc:.4f}")

    writer.close()

    # 绘制训练过程
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(fontsize=20)

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, 'g-', label='Training Accuracy')
    plt.plot(epochs_range, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(fontsize=20)

    plt.tight_layout()
    plt.show()

def main():


        # 加载 CSV 和 JSONL 文件
        csv_file_path = 'merged_file.csv'
        jsonl_file_path = 'merged_genres_exclude_niche.jsonl'

        # 加载 CSV 文件
        try:
            df = pd.read_csv(csv_file_path)
            print(f"CSV 文件加载成功，包含 {len(df)} 行。")
        except FileNotFoundError:
            print(f"未找到 CSV 文件: {csv_file_path}")
            return
        except Exception as e:
            print(f"加载 CSV 文件时出错: {e}")
            return

        # 加载 JSONL 文件
        try:
            with open(jsonl_file_path, 'r') as f:
                genre_data = [json.loads(line) for line in f]
            genre_df = pd.DataFrame(genre_data)
            print(f"JSONL 文件加载成功，包含 {len(genre_df)} 行。")
        except FileNotFoundError:
            print(f"未找到 JSONL 文件: {jsonl_file_path}")
            return
        except json.JSONDecodeError as e:
            print(f"解析 JSONL 文件时出错: {e}")
            return
        except Exception as e:
            print(f"加载 JSONL 文件时出错: {e}")
            return

        # 确保 'id' 列类型一致
        df['id'] = df['id'].astype(str)
        genre_df['id'] = genre_df['id'].astype(str)

        # 合并 DataFrame
        merged_df = pd.merge(df, genre_df, on='id', how='inner')
        print(f"Merged DataFrame 加载成功，包含 {len(merged_df)} 行。")
        print("Merged DataFrame Columns:", merged_df.columns.tolist())  # 调试信息

        # 检查 'genres' 列是否存在
        if 'genres' not in merged_df.columns:
            raise ValueError("Merged DataFrame does not contain 'genres' column.")

        # 处理缺失值（可选）
        merged_df = merged_df.dropna(subset=['genres'])
        print("处理缺失值后的 DataFrame 大小:", merged_df.shape)

        # 确保 'genres' 列中的每个条目都是列表，并选择第一个流派作为标签
        if merged_df['genres'].apply(lambda x: isinstance(x, list)).all():
            # 选择第一个流派作为单标签
            merged_df['genres'] = merged_df['genres'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
            print("将多标签转换为单标签，选择第一个流派。")
        else:
            # 如果 'genres' 不是列表，确保每个条目是字符串
            merged_df['genres'] = merged_df['genres'].apply(lambda x: x if isinstance(x, str) else 'Unknown')
            print("确保 'genres' 列中的每个条目都是字符串。")

        # 使用 LabelEncoder 将 'genres' 转换为整数索引
        label_encoder = LabelEncoder()
        merged_df['genre_index'] = label_encoder.fit_transform(merged_df['genres'])

        # 确认 'genre_index' 已添加
        print("Genres and their corresponding indices:")
        for genre, index in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
            print(f"{genre}: {index}")

        # 定义特征列和标签列
        feature_columns = [c for c in merged_df.columns if c not in ['id','genres','genre_index']]
        label_column = 'genre_index'

        # 检查是否有特征列
        if not feature_columns:
            raise ValueError("No feature columns found. Please check your DataFrame.")
        else:
            print(f"特征列数量: {len(feature_columns)}")

        # 创建完整的 Dataset，不进行标准化
        full_dataset = AudioFeatureWithGenreDataset(merged_df, feature_columns, label_column)
        print("创建完整的 Dataset 成功。")

        # 拆分为训练集和验证集
        val_size = 150
        train_size = len(full_dataset) - val_size
        if train_size <= 0:
            raise ValueError(f"Dataset size ({len(full_dataset)}) 太小，无法拆分出 {val_size} 个验证样本。")
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        print(f"拆分数据集成功，训练集大小: {len(train_ds)}，验证集大小: {len(val_ds)}")

        # 获取训练和验证的 DataFrame
        train_indices = train_ds.indices
        val_indices = val_ds.indices

        train_df = merged_df.iloc[train_indices].reset_index(drop=True)
        val_df = merged_df.iloc[val_indices].reset_index(drop=True)
        print("获取训练和验证的 DataFrame 成功。")

        # 初始化并拟合标准化器仅使用训练集
        scaler = StandardScaler()
        scaler.fit(train_df[feature_columns])
        print("StandardScaler 拟合训练集成功。")

        # 保存 scaler 以便未来使用（可选）
        joblib.dump(scaler, 'scaler.joblib')
        print("StandardScaler 已保存。")

        # 创建新的 Dataset 对象并应用标准化
        # 训练集启用数据增强，验证集不启用
        train_ds = AudioFeatureWithGenreDataset(train_df, feature_columns, label_column, scaler=scaler, augment=True)
        val_ds = AudioFeatureWithGenreDataset(val_df, feature_columns, label_column, scaler=scaler, augment=False)
        print("创建训练集和验证集的 Dataset 对象成功。")

        # 创建 DataLoaders
        batch_size = 128
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        print("创建 DataLoaders 成功。")

        # 构建模型
        input_dim = len(feature_columns)   # 特征数
        num_classes = merged_df['genre_index'].nunique()  # 类别数量
        print(f"Input Dimension: {input_dim}, Number of Classes: {num_classes}")

        model = AudioGenreClassifier(input_dim, num_classes)
        print("模型构建成功。")

        # 训练
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        train(model, train_loader, val_loader, label_encoder=label_encoder, num_epochs=150, lr=1e-3, device=device)
        print("训练完成。")

if __name__ == "__main__":
   main()
