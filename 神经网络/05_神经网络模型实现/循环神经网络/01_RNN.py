# RNN 的应用场景和功能包括：
# 1.自然语言处理（NLP，nature language process）。文本生成、机器翻译、情感分析等
# 2.时间序列预测。股票价格预测、天气预测、设备故障预测
# 3.语音识别。语音转文字、语音合成
# 4.视频分析。动作识别、视频描述生成、视频分类
# 5.推荐系统。用户行为预测、商品推荐
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# 超参数设置
input_size = 28  # 输入特征维度（MNIST 每行像素数）
hidden_size = 128  # 隐藏层维度
num_layers = 2  # RNN 层数
num_classes = 10  # 输出类别数
batch_size = 64
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 MNIST 数据集并转换为序列格式
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform)


# 将图像转换为序列输入 (batch_size, seq_len, input_size)
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images).squeeze(1)  # [N, 28, 28]
    return images.to(device), torch.tensor(labels).to(device)


train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,  # 输入形状为 (batch, seq, feature)
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播: output 形状 (batch, seq_len, hidden_size)
        out, _ = self.rnn(x, h0)

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # 全连接层分类
        out = self.fc(out)
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计指标
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 打印训练进度
    print(
        f"Epoch [{epoch+1}/{num_epochs}] | Loss: {total_loss/len(train_loader):.4f} | Acc: {100*correct/total:.2f}%"
    )

# 测试评估
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100*correct/total:.2f}%")

