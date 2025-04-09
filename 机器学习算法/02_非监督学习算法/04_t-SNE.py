# t-SNE 算法是一种非线性降维算法，常用于高维数据的可视化。
# 它通过将高维数据映射到低维空间（通常是二维或三维）来保留数据的局部结构。
# t-SNE 的全称是 t-Distributed Stochastic Neighbor Embedding。
# 它的主要思想是将高维空间中的相似点映射到低维空间中的相似点，同时保持它们之间的距离关系。
# t-SNE 通过计算高维空间中点对之间的相似度，并在低维空间中尽量保持这些相似度来实现降维。
# t-SNE 的优点是能够很好地处理非线性数据，并且在可视化高维数据时效果很好。
# 它常用于图像、文本和基因组数据的可视化。
# t-SNE 的缺点是计算复杂度较高，尤其是在处理大规模数据集时，可能会导致计算时间较长。
# 此外，t-SNE 的结果对参数设置（如 perplexity）敏感，不同的参数设置可能会导致不同的可视化结果。
# t-SNE 还不适合用于数据的聚类或分类任务，因为它主要用于可视化，而不是数据分析。

# t-SNE 的应用场景包括：
# 1. 图像数据的可视化：t-SNE 可以将高维图像数据映射到低维空间，以便于观察图像之间的相似性。
# 2. 文本数据的可视化：t-SNE 可以将高维文本数据（如词向量）映射到低维空间，以便于观察文本之间的相似性。
# 3. 基因组数据的可视化：t-SNE 可以将高维基因组数据映射到低维空间，以便于观察基因之间的相似性。


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


class TSNE(nn.Module):
    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        learning_rate=200.0,
        n_iter=1000,
        device=None,
    ):
        super(TSNE, self).__init__()
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def fit_transform(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_samples, n_features = X.shape

        # 初始化低维嵌入
        Y = torch.randn(
            n_samples, self.n_components, device=self.device, requires_grad=True
        )
        optimizer = optim.Adam([Y], lr=self.learning_rate)

        # 计算高维空间的条件概率分布
        P = self._compute_P(X)

        # 用于存储KL散度的列表
        kl_divergence_values = []

        for epoch in range(self.n_iter):
            # 计算低维空间的联合概率分布
            Q = self._compute_Q(Y)

            # 计算KL散度
            loss = self._kl_divergence(P, Q)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录KL散度
            kl_divergence_values.append(loss.item())

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.n_iter}, Loss: {loss.item()}")

        # 绘制KL散度变化图
        self._plot_kl_divergence(kl_divergence_values)

        return Y.detach().cpu().numpy()

    def _compute_P(self, X):
        # 计算高维空间中样本点之间的距离
        distances = torch.cdist(X, X)
        # 计算条件概率分布
        P = torch.zeros_like(distances)
        for i in range(X.shape[0]):
            # 计算第i个样本点的条件概率分布
            beta = 1.0
            # 使用二分法找到合适的beta使得困惑度等于指定值
            # ...
            # 这里省略了计算beta的代码，实际应用中需要实现
            # 计算条件概率分布
            p_i = torch.exp(-distances[i] * beta)
            p_i[i] = 0.0
            p_i_sum = torch.sum(p_i)
            if p_i_sum > 0.0:
                p_i = p_i / p_i_sum
            P[i] = p_i

        # 对称化条件概率分布
        P = (P + P.T) / (2.0 * X.shape[0])
        P = torch.clamp(P, min=1e-12)

        return P

    def _compute_Q(self, Y):
        # 计算低维空间中样本点之间的距离
        distances = torch.cdist(Y, Y)
        # 计算联合概率分布
        Q = 1.0 / (1.0 + distances**2)
        Q = Q / torch.sum(Q)
        Q = torch.clamp(Q, min=1e-12)

        return Q

    def _kl_divergence(self, P, Q):
        # 计算KL散度
        kl_div = torch.sum(P * torch.log(P / Q))
        return kl_div

    def _plot_kl_divergence(self, kl_values):
        # 绘制KL散度变化图
        plt.figure(figsize=(10, 6))
        plt.plot(kl_values)
        plt.title("KL Divergence during t-SNE optimization")
        plt.xlabel("Iteration")
        plt.ylabel("KL Divergence")
        plt.grid(True)
        plt.show()


# 示例用法
if __name__ == "__main__":
    # 加载数据
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 标准化数据
    X = StandardScaler().fit_transform(X)

    # 创建t-SNE模型
    tsne = TSNE(n_components=2, perplexity=30.0, learning_rate=1.0, n_iter=1000)

    # 执行t-SNE降维
    Y = tsne.fit_transform(X)

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.scatter(Y[:, 0], Y[:, 1], c=y)
    plt.title("t-SNE visualization of Iris dataset")
    plt.show()
