# GRU（Gated Recurrent Unit）是简化版的 LSTM，计算效率更高

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 1. 定义 GRU 模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,  # 输入特征的维度
            hidden_size=hidden_size,  # GRU 每层的输出维度
            num_layers=num_layers,  # 堆叠的 GRU 层数
            batch_first=True,  # 是否调整输入输出形状为(batch, seq, feature)，便于处理批量数据
            dropout=dropout if num_layers > 1 else 0,  # 多层时启用 dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        out, _ = self.gru(x)  # 输出形状: (batch_size, seq_len, hidden_size)
        if self.dropout:
            out = self.dropout(out[:, -1, :])  # 取最后一个时间步并应用 Dropout
        else:
            out = out[:, -1, :]
        return self.fc(out)


# 2. 数据准备
input_size = 10  # 输入特征维度
seq_len = 15  # 序列长度
n_samples = 1000  # 总样本数

# 生成模拟数据（随机数据，实际应用中需替换为真实数据）
x = torch.randn(n_samples, seq_len, input_size)
y = torch.randn(n_samples, 1)  # 假设输出为回归任务

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.1, random_state=42
)

# 转换为 DataLoader
batch_size = 32
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 3. 初始化模型与训练参数
hidden_size = 64
output_size = 1
num_layers = 2
dropout = 0.2
learning_rate = 0.001
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel(input_size, hidden_size, output_size, num_layers, dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. 训练循环
train_losses = []
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 5. 模型评估与保存
# 保存模型参数
# torch.save(model.state_dict(), "gru_model.pth")

# 绘制损失曲线
plt.plot(train_losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()

# 测试集评估
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        test_loss += criterion(outputs, targets).item()
print(f"Test Loss: {test_loss / len(test_loader):.4f}")
