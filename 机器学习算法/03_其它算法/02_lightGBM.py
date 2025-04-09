# lightGBM 是一个高效的梯度提升树算法，适用于大规模数据集和高维特征
# 它的主要优点是速度快、内存占用少、支持并行计算和分布式计算
# 适用于分类、回归和排序等任务

import torch
import numpy as np


# 定义单个决策树
class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    class Node:
        def __init__(
            self, feature_idx=None, threshold=None, value=None, left=None, right=None
        ):
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.value = value
            self.left = left
            self.right = right

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        best_gain = 0
        best_feature, best_threshold = None, None

        # 如果是叶子节点，直接返回值
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return self.Node(value=self._leaf_value(y))

        # 寻找最佳分裂点
        for feature_idx in range(n_features):
            thresholds = torch.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idx = X[:, feature_idx] <= threshold
                right_idx = ~left_idx
                if torch.sum(left_idx) == 0 or torch.sum(right_idx) == 0:
                    continue
                gain = self._information_gain(y, left_idx, right_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # 如果无法分裂，返回叶子节点
        if best_gain == 0:
            return self.Node(value=self._leaf_value(y))

        # 递归分裂左右子树
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = ~left_idx
        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)
        return self.Node(
            feature_idx=best_feature, threshold=best_threshold, left=left, right=right
        )

    def _leaf_value(self, y):
        # 回归任务：输出均值
        return torch.mean(y)

    def _information_gain(self, y, left_idx, right_idx):
        # 信息增益（方差减少）
        var_parent = torch.var(y)
        var_left = torch.var(y[left_idx])
        var_right = torch.var(y[right_idx])
        n_left = torch.sum(left_idx)
        n_right = torch.sum(right_idx)
        n_total = n_left + n_right
        return var_parent - (
            n_left / n_total * var_left + n_right / n_total * var_right
        )

    def predict(self, X):
        return torch.tensor([self._predict(x, self.root) for x in X])

    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)


# 定义梯度提升模型
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # 初始预测值为均值
        initial_pred = torch.mean(y).repeat(len(y))
        residuals = y - initial_pred
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            pred = tree.predict(X)
            self.trees.append(tree)
            residuals -= self.learning_rate * pred

    def predict(self, X):
        pred = torch.zeros(len(X))
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred + torch.mean(pred)


if __name__ == "__main__":
    # 生成示例数据
    X = torch.randn(100, 5)
    y = 3 * X[:, 0] + 2 * X[:, 1] + torch.randn(100) * 0.1

    # 训练模型
    model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    # 预测
    y_pred = model.predict(X)
    print("MSE:", torch.mean((y - y_pred) ** 2).item())
