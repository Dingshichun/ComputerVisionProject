# PCA (主成分分析) 是一种常用的降维技术，旨在通过线性变换将数据投影到一个新的坐标系中，
# 使得投影后的数据在新坐标系中的方差最大化。
# PCA 的主要步骤包括数据标准化、计算协方差矩阵、计算特征值和特征向量、选择主成分以及数据投影。

# 为什么选择方差最大的方向作为主成分？如何选择主成分？
# 方差越大说明数据在该方向的信息量越大，保留这些方向能最大程度保留原始数据信息
# 通常选择前 k 个特征值最大的特征向量作为主成分，这些主成分能够解释数据中大部分的方差。


import torch

def pca(X: torch.Tensor, num_components: int):
    """
    PCA 实现（基于 SVD 优化版）
    :param X: 输入数据，形状 (n_samples, n_features)
    :param num_components: 目标维度
    :return: 降维后数据、主成分矩阵
    """
    # 1. 数据标准化（仅中心化）
    mean = torch.mean(X, dim=0)
    X_centered = X - mean

    # 2. SVD 分解
    _, _, V = torch.linalg.svd(X_centered, full_matrices=False)

    # 3. 选择主成分
    principal_components = V[:num_components, :].T

    # 4. 数据投影
    X_reduced = torch.mm(X_centered, principal_components)
    return X_reduced, principal_components

# 示例数据（4 个样本，3 个特征）
X = torch.tensor([
    [2.5, 2.4, 3.3],
    [0.5, 0.7, 1.9],
    [2.2, 2.9, 3.1],
    [1.9, 2.2, 2.6]
])

# 降维到 2 维
X_reduced, components = pca(X, num_components=2)
print("降维数据:\n", X_reduced)
print("主成分:\n", components)