"""
手写数字识别
运行前需要先下载 MNIST 数据集，并解压到 ./data 目录下。
下载地址：https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
MNIST 数据集是手写数字数据集，包含 60000 张训练图像和 10000 张测试图像

"""

import torch
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
from torch.utils.data import DataLoader  # 数据加载器
from torchvision import datasets, transforms  # 数据集和转换模块
import matplotlib.pyplot as plt  # 可视化模块
import numpy as np  # 数组处理模块

# GPU 可用就使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # 转换为 Tensor 格式 [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 数据集的均值和标准差
    ]
)

# 加载数据集并划分为训练集和测试集
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 定义神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # mnist 数据集是单通道的灰度图像，所以输入通道数为 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 注意全连接层的输入大小要根据卷积层和池化层的输出大小来计算
        # 输入大小为 1 * 28 * 28，经过两次卷积和池化后输出大小为 64 * 7 * 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # [B, 32, 28, 28]
        x = torch.max_pool2d(x, 2)  # [B, 32, 14, 14]
        x = torch.relu(self.conv2(x))  # [B, 64, 14, 14]
        x = torch.max_pool2d(x, 2)  # [B, 64, 7, 7]
        # view() 函数用于改变张量的形状，-1 表示自动计算该维度的大小
        x = x.view(-1, 64 * 7 * 7)  # 展平 [B, 3136]
        x = torch.relu(self.fc1(x))  # [B, 128]
        x = self.fc2(x)  # [B, 10]
        return x


# 模型实例化并移动到 GPU 或 CPU
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器


# 训练函数
def train(epoch):
    model.train()  # 设置模型为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # 清空梯度
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        if batch_idx % 100 == 0:
            print(
                f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
            )


# 测试函数
def test():
    model.eval()  # 设置模型为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # 累加损失
            pred = output.argmax(dim=1)  # 获取预测结果
            correct += pred.eq(target).sum().item()  # 统计正确预测的数量

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)"
    )


# 训练10个epoch
for epoch in range(1, 11):
    train(epoch)
    test()


# 可视化预测结果
def plot_predictions():
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)
        preds = output.argmax(dim=1)

    fig = plt.figure(figsize=(12, 4))
    for i in range(10):
        ax = fig.add_subplot(1, 10, i + 1)
        img = data[i].cpu().numpy().squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"Pred: {preds[i].item()}\nTrue: {target[i].item()}",
            color="green" if preds[i] == target[i] else "red",
        )
        ax.axis("off")
    plt.show()


plot_predictions()
