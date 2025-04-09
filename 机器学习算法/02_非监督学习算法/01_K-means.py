# K-means 聚类是一种常用的无监督学习算法，用于将数据集划分为 K 个簇。
# 它通过迭代地将数据点分配到最近的簇中心，并更新簇中心来实现聚类。

# K-means 聚类的步骤如下：
# 1. 初始化 K 个簇中心（可以随机选择数据点作为初始中心）。
# 2. 将每个数据点分配到最近的簇中心。
# 3. 更新每个簇的中心为簇内所有数据点的均值。
# 4. 重复步骤 2 和 3，直到簇中心不再变化或达到最大迭代次数。
# 5. 返回最终的簇中心和每个数据点的簇标签。

# K-means 聚类的优缺点：
# 优点：简单易懂，计算效率高，适用于大规模数据集。
# 缺点：需要预先指定 K 值，对初始簇中心敏感，容易陷入局部最优解，对噪声和异常值敏感。

# K-means 聚类的应用场景：
# 图像压缩、市场细分、社交网络分析、文档聚类等。

import torch
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-6):
        self.n_clusters = n_clusters  # 聚类数
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛容忍度

    def fit(self, X):
        # 初始化质心（K-means++）
        self.centroids = X[torch.randperm(len(X))[: self.n_clusters]]
        for _ in range(self.max_iter):
            dists = torch.cdist(X, self.centroids)
            labels = torch.argmin(dists, dim=1)
            new_centroids = torch.stack(
                [X[labels == i].mean(dim=0) for i in range(self.n_clusters)]
            )  # 新质心

            # 检查收敛条件
            if torch.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids  # 更新质心
        self.labels_ = labels # 保存标签
        return self # 返回 self 以便链式调用


# 示例使用
if __name__ == "__main__":
    # 生成数据并运行聚类
    X = torch.randn(1000, 2).cuda()  # GPU 数据
    model = KMeans(n_clusters=3).fit(X)

    # 可视化
    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=model.labels_.cpu())
    plt.scatter(
        model.centroids[:, 0].cpu(),
        model.centroids[:, 1].cpu(),
        s=200,
        marker="X",
        c="red",
    )
    plt.show()
