# 层次聚类是一种聚类算法，它通过构建一个层次结构的树状图来组织数据点。
# 它可以分为两种主要类型：凝聚型（自底向上）和分裂型（自顶向下）。
# 在这里，我们实现一个简单的凝聚型层次聚类算法。
# 该实现使用 PyTorch 进行计算，并支持不同的链接方法（如 Ward、单链接、全链接等）。
# 我们将使用 PyTorch 来处理数据和计算距离矩阵，以便利用 GPU 加速。

# 层次聚类的步骤如下：
# 1. 初始化：每个数据点作为一个簇。
# 2. 计算簇之间的距离矩阵。
# 3. 找到最近的两个簇并合并它们。
# 4. 更新距离矩阵。
# 5. 重复步骤 3 和 4，直到达到所需的簇数。
# 6. 返回最终的簇标签和树状图（可选）。

# 层次聚类的优缺点：
# 优点：不需要预先指定簇数，可以生成树状图，适用于小规模数据集。
# 缺点：计算复杂度高，内存消耗大，对噪声和异常值敏感。

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform


class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage  # 支持'single', 'complete', 'average', 'ward'

    def fit(self, X):
        # 转换为PyTorch张量并标准化
        X = torch.tensor(X, dtype=torch.float32)
        X = (X - X.mean(dim=0)) / X.std(dim=0)

        # 初始化：每个样本为一个簇
        clusters = [{"samples": [i], "centroid": X[i]} for i in range(len(X))]
        cluster_history = []  # 记录合并过程

        # 迭代合并直到目标簇数
        while len(clusters) > self.n_clusters:
            # 计算簇间距离矩阵
            dist_matrix = self._compute_distance_matrix(clusters)

            # 找到最近的两个簇
            i, j = self._find_nearest_clusters(dist_matrix)

            # 合并簇并更新列表
            merged_cluster = self._merge_clusters(clusters[i], clusters[j])
            del clusters[j], clusters[i]
            clusters.append(merged_cluster)
            cluster_history.append((i, j, merged_cluster))

        self.labels_ = self._get_labels(cluster_history, len(X))
        return self

    def _compute_distance_matrix(self, clusters):
        n = len(clusters)
        dist_matrix = torch.full((n, n), float("inf"))

        for i in range(n):
            for j in range(i + 1, n):
                if self.linkage == "ward":
                    # Ward方差法计算簇间距离
                    m_i = len(clusters[i]["samples"])
                    m_j = len(clusters[j]["samples"])
                    dist = (
                        torch.norm(clusters[i]["centroid"] - clusters[j]["centroid"])
                        ** 2
                        * (m_i * m_j)
                        / (m_i + m_j)
                    )
                else:
                    # 其他链接方式（需实现对应距离计算）
                    pass
                dist_matrix[i, j] = dist
        return dist_matrix

    def _find_nearest_clusters(self, dist_matrix):
        flat_idx = torch.argmin(dist_matrix).item()
        return (flat_idx // dist_matrix.shape[0], flat_idx % dist_matrix.shape[0])

    def _merge_clusters(self, c1, c2):
        merged_samples = c1["samples"] + c2["samples"]
        merged_centroid = (
            c1["centroid"] * len(c1["samples"]) + c2["centroid"] * len(c2["samples"])
        ) / (len(c1["samples"]) + len(c2["samples"]))
        return {"samples": merged_samples, "centroid": merged_centroid}

    def _get_labels(self, history, n_samples):
        # 生成最终标签（略，可通过树形结构回溯）
        pass


if __name__ == "__main__":
    # 生成测试数据
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=100, centers=3)

    # 执行聚类
    hc = HierarchicalClustering(n_clusters=3)
    hc.fit(X)

    # 可视化（需配合 matplotlib）
    import matplotlib.pyplot as plt

    plt.scatter(X[:, 0], X[:, 1], c=hc.labels_)
    plt.show()
