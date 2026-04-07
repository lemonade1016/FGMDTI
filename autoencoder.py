import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# 为可复现性设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义目录和文件路径
# 请确保这里的路径是正确的
directory = '/media/4T2/zc/Drug-Protein/datasets/KIBA/final'
text_file = os.path.join(directory, 'sub_text.npy')
sequence_file = os.path.join(directory, 'sub_sequence.npy')
output_file = os.path.join(directory, 'drug_merge_1024.npy')

# 以 allow_pickle=True 加载 .npy 文件
raw_text_data = np.load(text_file, allow_pickle=True)
raw_sequence_data = np.load(sequence_file, allow_pickle=True)

# 验证原始形状和数据类型
print(f"Original shape of sub_text.npy: {raw_text_data.shape}")
print(f"Original shape of sub_sequence.npy: {raw_sequence_data.shape}")

# --- START OF MODIFICATION ---

# 1. 处理文本数据 (sub_text.npy)
# 跳过前4列，目标形状 (175, 1024)
print("\nProcessing text data...")
# 对数组进行切片，获取所有行，但只取第4列之后的所有列
text_data = raw_text_data[:, 4:]
# 将切片后的数组转换为 float32 类型
text_data = text_data.astype(np.float32)
print(f"Shape of processed text data: {text_data.shape}")
print(f"Dtype of processed text data: {text_data.dtype}")


# 2. 处理序列数据 (sub_sequence.npy)
# 跳过前4列，目标形状 (175, 768)
print("\nProcessing sequence data...")
# 对数组进行切片，获取所有行，但只取第4列之后的所有列
sequence_data = raw_sequence_data[:, 4:]
# 将切片后的数组转换为 float32 类型
sequence_data = sequence_data.astype(np.float32)
print(f"Shape of processed sequence data: {sequence_data.shape}")
print(f"Dtype of processed sequence data: {sequence_data.dtype}")

# --- END OF MODIFICATION ---


# 定义自编码器的输入维度
# 文本输入维度应为 1028 - 4 = 1024
# 序列输入维度应为 772 - 4 = 768
TEXT_INPUT_DIM = text_data.shape[1]
SEQUENCE_INPUT_DIM = sequence_data.shape[1]

print(f"\nText autoencoder input dim: {TEXT_INPUT_DIM}")
print(f"Sequence autoencoder input dim: {SEQUENCE_INPUT_DIM}")

# 转换为 PyTorch 张量
try:
    # 确保两个张量的数据类型都是 float32
    # text_data 和 sequence_data 已经是正确的类型和形状
    text_tensor = torch.from_numpy(text_data)
    sequence_tensor = torch.from_numpy(sequence_data)
except Exception as e:
    print(f"Error converting to tensor: {e}")
    exit(1)

# 创建 DataLoader 进行批处理
text_dataset = TensorDataset(text_tensor)
sequence_dataset = TensorDataset(sequence_tensor)
text_loader = DataLoader(text_dataset, batch_size=64, shuffle=False)
sequence_loader = DataLoader(sequence_dataset, batch_size=64, shuffle=False)


# 定义自编码器类
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# 初始化自编码器
# 使用从数据形状动态获取的维度
text_autoencoder = Autoencoder(input_dim=TEXT_INPUT_DIM, encoding_dim=512)
sequence_autoencoder = Autoencoder(input_dim=SEQUENCE_INPUT_DIM, encoding_dim=512)

# 定义损失函数和优化器
criterion = nn.MSELoss()
text_optimizer = optim.Adam(text_autoencoder.parameters(), lr=0.001)
sequence_optimizer = optim.Adam(sequence_autoencoder.parameters(), lr=0.001)


# 训练函数
def train_autoencoder(model, data_loader, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            inputs = batch[0]
            optimizer.zero_grad()
            _, decoded = model(inputs)
            loss = criterion(decoded, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader):.6f}")


# 训练文本自编码器
print("\nTraining text autoencoder...")
train_autoencoder(text_autoencoder, text_loader, text_optimizer)

# 训练序列自编码器
print("\nTraining sequence autoencoder...")
train_autoencoder(sequence_autoencoder, sequence_loader, sequence_optimizer)

# 提取编码后的特征
text_autoencoder.eval()
sequence_autoencoder.eval()

with torch.no_grad():
    # 将完整的张量传入模型以获取所有数据的编码特征
    text_encoded, _ = text_autoencoder(text_tensor)
    sequence_encoded, _ = sequence_autoencoder(sequence_tensor)

    text_encoded = text_encoded.numpy()
    sequence_encoded = sequence_encoded.numpy()

# 验证编码后特征的形状
print(f"\nShape of encoded text features: {text_encoded.shape}")
print(f"Shape of encoded sequence features: {sequence_encoded.shape}")

# 拼接编码后的特征 (目标维度: 512 + 512 = 1024)
fused_features = np.concatenate((text_encoded, sequence_encoded), axis=1)

# 验证融合后特征的形状
print(f"Shape of fused features: {fused_features.shape}")

# 保存融合后的特征
np.save(output_file, fused_features)
print(f"Saved fused features to {output_file}")