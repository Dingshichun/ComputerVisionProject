# 随机森林（Random Forest）是集成学习方法中的一种，主要用于分类和回归任务。
# 它通过构建多棵决策树并结合它们的预测结果来提高模型的准确性和鲁棒性。
# 随机森林的基本思想是：通过引入随机性来构建多棵决策树，
# 然后对这些树的预测结果进行投票（分类）或平均（回归），从而得到最终的预测结果。


import torch
from collections import Counter # 用于统计类别频率


# 定义决策树节点
class TreeNode:
    def __init__(
        self, feature_idx=None, threshold=None, left=None, right=None, class_label=None
    ):
        self.feature_idx = feature_idx  # 分裂特征索引
        self.threshold = threshold  # 分裂阈值
        self.left = left  # 左子树（≤阈值）
        self.right = right  # 右子树（>阈值）
        self.class_label = class_label  # 叶节点类别（分类任务）


# 定义决策树
class DecisionTree:
    def __init__(self, max_depth=None, n_features=None):
        self.max_depth = max_depth  # 最大树深
        self.n_features = n_features  # 随机选择特征数（如√总特征数）
        self.tree = None

    def fit(self, X, y):
        self.n_features = (
            X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        )
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # 终止条件：单类别或达到最大深度
        if len(torch.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return TreeNode(class_label=Counter(y.tolist()).most_common(1)[0][0])

        # 随机选择特征子集
        feature_indices = torch.randperm(X.shape[1])[: self.n_features]
        best_feature, best_threshold = self._best_split(X, y, feature_indices)

        # 分裂数据并递归构建子树
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature_idx=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _best_split(self, X, y, feature_indices):
        # 遍历特征和阈值，选择最佳分裂点（如基尼不纯度最小）
        best_gini = float("inf")
        best_feature, best_threshold = None, None
        for feat in feature_indices:
            thresholds = torch.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                gini = self._gini_impurity(y[left_mask], y[~left_mask])
                if gini < best_gini:
                    best_gini = gini
                    best_feature, best_threshold = feat, thresh
        return best_feature, best_threshold

    def _gini_impurity(self, left_y, right_y):
        # 计算左右子节点的基尼不纯度加权和
        p_left = len(left_y) / (len(left_y) + len(right_y))
        p_right = 1 - p_left
        return p_left * self._node_gini(left_y) + p_right * self._node_gini(right_y)

    def _node_gini(self, y):
        # 单个节点的基尼不纯度
        _, counts = torch.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - (p**2).sum()


# 定义随机森林
class RandomForest:
    def __init__(self, n_trees=100, max_depth=5, n_features=None):
        self.n_trees = n_trees  # 树的数量
        self.max_depth = max_depth  # 单棵树最大深度
        self.n_features = n_features  # 特征子集大小（默认√总特征数）
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            # Bootstrap采样（有放回抽取样本）
            idx = torch.randint(0, len(X), (len(X),))
            X_sub, y_sub = X[idx], y[idx]

            # 构建单棵决策树
            tree = DecisionTree(max_depth=self.max_depth, n_features=self.n_features)
            tree.fit(X_sub, y_sub)
            self.trees.append(tree)

    def predict(self, X):
        # 所有树的预测结果投票
        preds = torch.stack([self._predict_tree(tree, X) for tree in self.trees])
        return torch.mode(preds, dim=0).values

    def _predict_tree(self, tree, X):
        # 单棵树预测
        def _traverse(node, x):
            if node.class_label is not None:
                return node.class_label
            if x[node.feature_idx] <= node.threshold:
                return _traverse(node.left, x)
            else:
                return _traverse(node.right, x)

        return torch.tensor([_traverse(tree.tree, x) for x in X])


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 数据加载与转换
iris = load_iris()
X, y = iris.data, iris.target
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练随机森林
rf = RandomForest(n_trees=50, max_depth=3, n_features=int(X.shape[1] ** 0.5))
rf.fit(X_train, y_train)

# 预测与评估
preds = rf.predict(X_test)
accuracy = (preds == y_test).float().mean()
print(f"Accuracy: {accuracy:.2f}")  # 典型输出：Accuracy: 0.95
