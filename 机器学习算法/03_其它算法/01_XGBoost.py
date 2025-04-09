# XGBoost 是一种高效的梯度提升树算法，广泛应用于分类和回归任务。
# 它具有处理缺失值、支持并行计算和自动特征选择等优点。
# 具体来说，XGBoost 通过构建多个决策树来进行预测，每棵树都是在前一棵树的基础上进行改进的。
# 这种方法可以有效地减少过拟合，提高模型的泛化能力。
# XGBoost 还提供了多种参数调节选项，可以根据具体任务进行优化。

import torch
import numpy as np


def compute_gradients(y_true, y_pred):
    g = y_pred - y_true
    h = torch.ones_like(g)
    return g, h


class TreeNode:
    def __init__(
        self, feature_idx=None, threshold=None, left=None, right=None, value=None
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class XGBoostTree:
    def __init__(
        self, max_depth=3, min_samples_split=2, reg_lambda=1.0
    ):  # 参数名必须与传入的字典键名一致
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.root = None

    def _compute_gain(self, G, H, G_left, H_left, G_right, H_right):
        gain = (
            (G_left**2 / (H_left + self.reg_lambda)).sum()
            + (G_right**2 / (H_right + self.reg_lambda)).sum()
            - ((G + G_right) ** 2 / (H + H_right + self.reg_lambda)).sum()
        ) / 2
        return gain.item()

    def _find_best_split(self, X, g, h):
        best_gain = -float("inf")
        best_feature, best_threshold = None, None

        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            unique_values = torch.unique(feature_values)

            for threshold in unique_values:
                left_mask = feature_values <= threshold
                if left_mask.sum() == 0 or (~left_mask).sum() == 0:
                    continue

                G_left = g[left_mask].sum()
                H_left = h[left_mask].sum()
                G_right = g[~left_mask].sum()
                H_right = h[~left_mask].sum()

                if (
                    left_mask.sum() < self.min_samples_split
                    or (~left_mask).sum() < self.min_samples_split
                ):
                    continue

                current_gain = self._compute_gain(
                    g.sum(), h.sum(), G_left, H_left, G_right, H_right
                )
                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, g, h, depth=0):
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            leaf_value = -g.sum() / (h.sum() + self.reg_lambda)
            return TreeNode(value=leaf_value)

        feature_idx, threshold, gain = self._find_best_split(X, g, h)

        if gain < 0:
            leaf_value = -g.sum() / (h.sum() + self.reg_lambda)
            return TreeNode(value=leaf_value)

        left_mask = X[:, feature_idx] <= threshold
        left = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], g[~left_mask], h[~left_mask], depth + 1)

        return TreeNode(feature_idx, threshold, left, right)

    def fit(self, X, g, h):
        self.root = self._build_tree(X, g, h)

    def predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

    def predict(self, X):
        return torch.tensor(
            [self.predict_single(x, self.root) for x in X], dtype=torch.float32
        )


class XGBoost:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        reg_lambda=1.0,
    ):  # 显式声明参数
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.trees = []

    def fit(self, X, y, verbose=False):
        y_pred = torch.zeros_like(y, dtype=torch.float32)
        for i in range(self.n_estimators):
            g, h = compute_gradients(y, y_pred)

            tree = XGBoostTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                reg_lambda=self.reg_lambda,
            )
            tree.fit(X, g, h)

            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

            if verbose and (i % 10 == 0):
                loss = ((y - y_pred) ** 2).mean()
                print(f"Iter {i}, MSE: {loss:.4f}")

    def predict(self, X):
        y_pred = torch.zeros(X.shape[0], dtype=torch.float32)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.randn(1000, 5)
    y = 3 * X[:, 0] + 2 * X[:, 1] - 1.5 * X[:, 2] + torch.randn(1000) * 0.1

    model = XGBoost(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        reg_lambda=1.0,
    )

    model.fit(X, y, verbose=True)
    y_pred = model.predict(X)
    print("\nFinal MSE:", ((y - y_pred) ** 2).mean().item())
