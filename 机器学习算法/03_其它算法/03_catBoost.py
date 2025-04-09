# catBoost 是一个开源的机器学习库，专门用于处理分类特征。它在许多机器学习竞赛中表现出色，尤其是在 Kaggle 上。catBoost 的全称是 Categorical Boosting。
# 它是一个基于梯度提升算法的库，支持分类特征的自动处理。catBoost 的设计目标是提供高效、易用和准确的机器学习工具。

# catBoost 的主要特点包括：
# 1. 自动处理分类特征：catBoost 可以自动识别和处理分类特征，无需手动编码。
# 2. 高效的梯度提升算法：catBoost 使用了一种高效的梯度提升算法，可以处理大规模数据集。
# 3. 支持多种损失函数：catBoost 支持多种损失函数，包括回归、分类和排序任务。
# 4. 支持 GPU 加速：catBoost 可以使用 GPU 加速训练过程，提高训练速度。
# 5. 易于使用：catBoost 提供了简单易用的 API，可以快速上手。
# 6. 支持多种编程语言：catBoost 支持 Python、R、Java 和 C++ 等多种编程语言。
# 7. 支持多种模型评估指标：catBoost 提供了多种模型评估指标，可以帮助用户选择最佳模型。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class GradientBoostingNN:
    def __init__(self, n_estimators=100, learning_rate=0.1, embedding_dim=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.models = []
        self.embeddings = None

    def fit(self, X_cat, X_num, y, num_epochs=50):
        """
        X_cat: 类别特征 (LongTensor, shape: [n_samples, n_cat_features])
        X_num: 数值特征 (FloatTensor, shape: [n_samples, n_num_features])
        y: 目标值 (FloatTensor, shape: [n_samples])
        """
        n_samples, n_cat_features = X_cat.shape
        n_num_features = X_num.shape[1]

        # 初始化嵌入层
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(int(X_cat[:, i].max() + 1), self.embedding_dim)
                for i in range(n_cat_features)
            ]
        )

        # 初始化预测值
        current_pred = torch.zeros_like(y)

        for _ in range(self.n_estimators):
            # 计算残差
            residual = y - current_pred.detach()

            # 定义神经网络模型
            model = nn.Sequential(
                nn.Linear(n_cat_features * self.embedding_dim + n_num_features, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

            # 定义优化器和损失函数
            optimizer = optim.Adam(
                list(model.parameters()) + list(self.embeddings.parameters()), lr=0.01
            )
            criterion = nn.MSELoss()

            # 训练当前模型拟合残差
            dataset = TensorDataset(X_cat, X_num, residual)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            for epoch in range(num_epochs):
                for batch_cat, batch_num, batch_resid in loader:
                    # 嵌入类别特征
                    embedded = [
                        emb(batch_cat[:, i]) for i, emb in enumerate(self.embeddings)
                    ]
                    embedded = torch.cat(embedded, dim=1)

                    # 拼接数值特征
                    features = torch.cat([embedded, batch_num], dim=1)

                    # 前向传播
                    pred = model(features).squeeze()

                    # 计算损失
                    loss = criterion(pred, batch_resid)

                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # 保存模型并更新预测
            self.models.append(model)
            with torch.no_grad():
                embedded = [emb(X_cat[:, i]) for i, emb in enumerate(self.embeddings)]
                embedded = torch.cat(embedded, dim=1)
                features = torch.cat([embedded, X_num], dim=1)
                current_pred += self.learning_rate * model(features).squeeze()

    def predict(self, X_cat, X_num):
        with torch.no_grad():
            pred = torch.zeros(X_cat.shape[0])
            embedded = [emb(X_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            embedded = torch.cat(embedded, dim=1)
            features = torch.cat([embedded, X_num], dim=1)
            for model in self.models:
                pred += self.learning_rate * model(features).squeeze()
        return pred.numpy()


if __name__ == "__main__":
    # 示例数据
    n_samples = 1000
    n_cat_features = 2
    n_num_features = 5

    # 生成随机数据
    X_cat = torch.randint(0, 10, (n_samples, n_cat_features)).long()  # 类别特征（0-9）
    X_num = torch.randn(n_samples, n_num_features).float()  # 数值特征
    y = torch.randn(n_samples).float()  # 目标值

    # 训练模型
    gbnn = GradientBoostingNN(n_estimators=20, learning_rate=0.1)
    gbnn.fit(X_cat, X_num, y)

    # 预测
    predictions = gbnn.predict(X_cat, X_num)
    print(f"预测值形状是：{predictions.shape}")  # 打印预测值的形状
    print(predictions[:10])  # 打印前 10 个预测值
