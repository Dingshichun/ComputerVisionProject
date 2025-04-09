# 梯度提升树是一种集成学习方法，通常用于回归和分类任务。
# 它通过构建一系列弱学习器（通常是决策树）来逐步改进模型的预测能力。
# 每个新树都是在前一棵树的残差上进行训练的，从而逐步减少预测误差。

import torch


class RegressionTree:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    class Node:
        def __init__(
            self, feature_idx=None, threshold=None, left=None, right=None, value=None
        ):
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            return self.Node(value=torch.mean(y).item())

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return self.Node(value=torch.mean(y).item())

        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return self.Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y):
        best_loss = float("inf")
        best_feature, best_threshold = None, None

        for feature_idx in range(X.shape[1]):
            thresholds = torch.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_idxs = X[:, feature_idx] <= threshold
                if left_idxs.sum() == 0 or (~left_idxs).sum() == 0:
                    continue
                left_loss = torch.sum((y[left_idxs] - y[left_idxs].mean()) ** 2)
                right_loss = torch.sum((y[~left_idxs] - y[~left_idxs].mean()) ** 2)
                total_loss = left_loss + right_loss
                if total_loss < best_loss:
                    best_loss = total_loss
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold

    def predict(self, X):
        return torch.tensor([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)


class GBDT:
    def __init__(
        self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        self.initial_pred = torch.mean(y).item()
        current_pred = torch.full_like(y, self.initial_pred, dtype=torch.float32)
        for _ in range(self.n_estimators):
            residual = y - current_pred
            tree = RegressionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.fit(X, residual)
            residual_pred = tree.predict(X)
            current_pred += self.learning_rate * residual_pred
            self.trees.append(tree)

    def predict(self, X):
        pred = torch.full((X.shape[0],), self.initial_pred, dtype=torch.float32)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred


# 示例使用
if __name__ == "__main__":
    # 生成样本数据
    X = torch.randn(100, 2)
    y = 3 * X[:, 0] + 2 * X[:, 1] + torch.randn(100) * 0.1

    # 训练模型
    model = GBDT(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    # 预测
    y_pred = model.predict(X)

    # 计算均方误差
    loss = torch.mean((y_pred - y) ** 2)
    print(f"Mean Squared Error: {loss.item()}")
