# 线性回归用于预测连续值（如房价、销售额等），损失函数一般是均方误差

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 数据生成
X = 2 * np.random.rand(100, 1)  # numpy 生成 100 个随机数，范围[0, 2)
y = 4 + 3 * X + np.random.randn(100, 1)  # 线性关系 y = 4 + 3 * x + 噪声
X_train = torch.FloatTensor(X)  # numpy 格式转换为 PyTorch 张量
y_train = torch.FloatTensor(y)

# 模型定义。多元线性回归只需要增加输入特征数即可
model = nn.Linear(1, 1)  # 线性回归模型，输入特征数 1，输出特征数 1
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 训练
for epoch in range(1000):
    outputs = model(X_train)  # 前向传播
    loss = criterion(outputs, y_train)  # 计算损失

    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

# 预测
x_test = torch.tensor([[5.0]])
print(f"预测值：{model(x_test).item():.2f}")  # 理想输出≈19.0（4+3 * 5）
