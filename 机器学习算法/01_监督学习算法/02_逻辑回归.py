# 逻辑回归用于二分类问题（如垃圾邮件检测、疾病预测等）
# 损失函数是交叉熵损失函数
# 逻辑回归的目标是最大化似然函数，最小化交叉熵损失函数
# 逻辑回归的假设函数是 sigmoid 函数，输出值在 0 到 1 之间
# 逻辑回归的输出值可以看作是正类的概率，（1-输出值）就是负类的概率
# 当输出值大于 0.5 时，预测为正类；否则预测为负类

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据生成与预处理
np.random.seed(42)  # 设置随机种子以便复现结果
X = np.random.randn(200, 2)  # 生成 200 个服从标准正态分布的二维样本

# X 第一行和第二行之和大于 0 的样本标记为 1，否则为 0
y = ((X[:, 0] + X[:, 1]) > 0).astype(np.float32)
print(X[:5])  # 打印前 5 个样本
print(y[:5])
print(f"y.shape is :{y.shape}") # y.shape 是 (200,)，表示 200 个样本的标签

# 转换为 PyTorch 张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
print(f"y_tensor.shape is :{y_tensor.shape}") # y_tensor.shape 是 (200, 1)，因为我们需要将 y 转换为列向量


# 2. 定义逻辑回归模型
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # sigmoid 函数将线性输出转换为概率


model = LogisticRegression(input_dim=2)

# 3. 定义损失函数与优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. 训练循环
losses = []  # 记录损失
accuracies = []  # 记录准确率

for epoch in range(1000):
    # 前向传播
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录训练指标
    losses.append(loss.item())

    # 将预测概率大于 0.5 的转换为 1，小于等于 0.5 的转换为 0
    predictions = (outputs > 0.5).float()

    accuracy = (predictions == y_tensor).float().mean()
    accuracies.append(accuracy)

    # 每 100 轮打印指标
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Acc: {accuracy.item():.4f}")

# 5. 可视化结果
# 损失曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 决策边界
# 根据全连接神经网络的结构，这里权重和偏置的形状是 (1, 2) 和 (1,)
# 如果输入数据是三维的，输出数据是二维的，那么权重的形状是 (2, 3)，偏置的形状是 (2,)
# 因为是全连接，前一层的每个神经元都与后一层的每个神经元相连，就有一组权重和一个偏置
print("weights:", model.linear.weight) # 输入数据形状是(1,2)，所以每一组权重形状是(1,2)
print("bias:", model.linear.bias) # 只有一个输出神经元，所以只有一个偏置
w = model.linear.weight.detach().numpy()[0]  # 获取权重的第一组元素
b = model.linear.bias.detach().numpy()[0]  # 获取偏置的第一个元素
print("w:", w)
print("b:", b)
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")  # 绘制散点图
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
x1_min, x1_max = X[:, 0].min(), X[:, 0].max()  # 获取 X 第一列的最小值和最大值
x2_min, x2_max = X[:, 1].min(), X[:, 1].max()  # 获取 X 第二列的最小值和最大值

# 生成网格点。xx1 和 xx2 是网格点的坐标，grid 是网格点的坐标矩阵
# np.meshgrid 函数生成网格点坐标矩阵，np.c_ 将两个数组按列拼接在一起
xx1, xx2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)
)
grid = torch.tensor(np.c_[xx1.ravel(), xx2.ravel()], dtype=torch.float32)
probs = model(grid).reshape(xx1.shape).detach().numpy() # 计算网格点的预测概率

plt.contour(xx1, xx2, probs, levels=[0.5], colors="green")  # 绘制决策边界
plt.title("Decision Boundary")
plt.show()
