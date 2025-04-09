# DBSCAN 聚类是一种基于密度的聚类算法，能够发现任意形状的簇，并且能够识别噪声点。
# 它的基本思想是：如果一个点的邻域内有足够多的点（至少 min_samples 个），
# 则将这些点归为同一簇，否则将其视为噪声点。
# DBSCAN 算法的核心是计算每个点的邻域内的点数，并根据这些点数来判断是否形成一个簇。

import torch


def dbscan_torch(X, eps, min_samples):
    """
    PyTorch 实现 DBSCAN 聚类算法

    参数:
        X (Tensor): 输入数据，形状为(n_samples, n_features)
        eps (float): 邻域半径
        min_samples (int): 形成核心点的最小邻域样本数

    返回:
        labels (Tensor): 聚类标签，形状为(n_samples,)，-1表示噪声点
    """
    device = X.device
    n_samples = X.shape[0]

    # 计算所有点之间的欧氏距离
    dists = torch.cdist(X, X)

    # 确定核心点（邻居数 >= min_samples）
    neighbours = (dists <= eps).sum(dim=1)
    core_mask = neighbours >= min_samples

    # 初始化访问标记和聚类标签
    visited = torch.zeros(n_samples, dtype=torch.bool, device=device)
    labels = torch.full((n_samples,), -1, dtype=torch.long, device=device)

    cluster_id = 0

    for i in range(n_samples):
        if not visited[i]:
            visited[i] = True
            if core_mask[i]:
                # 创建新簇
                labels[i] = cluster_id

                # 获取当前点的邻居（排除自身）
                neighbour_mask = (dists[i] <= eps) & (
                    torch.arange(n_samples, device=device) != i
                )
                queue = torch.nonzero(neighbour_mask).view(-1).cpu().tolist()

                # 扩展簇
                while queue:
                    q = queue.pop(0)
                    if not visited[q]:
                        visited[q] = True
                        labels[q] = cluster_id
                        if core_mask[q]:
                            # 获取q的邻居（排除自身）
                            q_neighbours = (dists[q] <= eps) & (
                                torch.arange(n_samples, device=device) != q
                            )
                            new_neighbours = (
                                torch.nonzero(q_neighbours).view(-1).cpu().tolist()
                            )
                            queue += new_neighbours
                cluster_id += 1
    return labels


if __name__ == "__main__":

    # 示例数据
    X = torch.tensor(
        [[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]], dtype=torch.float32
    )

    # DBSCAN 参数
    eps = 3.0
    min_samples = 2

    # 执行 DBSCAN 聚类
    labels = dbscan_torch(X, eps, min_samples)

    print("聚类标签:", labels.numpy())
