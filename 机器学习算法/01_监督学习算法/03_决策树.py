# 决策树是一种常用的分类和回归算法，具有易于理解和解释的优点。
# 本示例实现了一个简单的决策树分类器，使用 PyTorch 进行模型构建和训练。

import torch
from torch import nn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # 计算准确率

# 数据预处理（强制类型转换）
iris = load_iris() # 加载数据集
X = torch.tensor(iris.data, dtype=torch.float32)  # numpy 转 Tensor 的正确方式
y = torch.tensor(iris.target, dtype=torch.long)

# 数据集划分（保持 Tensor 类型）
X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(), y.numpy(), test_size=0.2, random_state=42
)
X_train = torch.tensor(X_train, dtype=torch.float32)  # 强制转换类型
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

class DecisionTree(nn.Module):
    def __init__(self, max_depth=5, min_samples_split=2):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = self.build_tree(X_train, y_train, depth=0)
    
    def gini_impurity(self, y):
        # 基尼系数计算（强制类型转换）
        _, counts = torch.unique(y, return_counts=True)
        probabilities = counts.float() / counts.sum().float()  # 显式转换避免整数除法
        return 1 - (probabilities**2).sum()
    
    def get_best_split(self, X, y):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            thresholds = torch.unique(feature_values)  # 确保Tensor输入
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                
                gini_left = self.gini_impurity(y[left_mask])
                gini_right = self.gini_impurity(y[right_mask])
                weighted_gini = (gini_left * left_mask.sum().float() + 
                                gini_right * right_mask.sum().float()) / len(y)  # 网页7：统一浮点运算[7](@ref)
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'class': torch.mode(y).values.item()}  # 递归终止条件
        
        feature_idx, threshold = self.get_best_split(X, y)
        if feature_idx is None:
            return {'class': torch.mode(y).values.item()}
        
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # 递归构建子树（保持 Tensor 输入）
        node = {
            'feature_idx': feature_idx,
            'threshold': threshold.item(),
            'left': self.build_tree(X[left_mask], y[left_mask], depth+1),
            'right': self.build_tree(X[right_mask], y[right_mask], depth+1)
        }
        return node
    
    def predict_sample(self, x, node):
        if 'class' in node:
            return node['class']
        if x[node['feature_idx']] <= node['threshold']:
            return self.predict_sample(x, node['left'])
        else:
            return self.predict_sample(x, node['right'])
    
    def forward(self, X):
        # 输出统一为 Tensor
        return torch.stack([torch.tensor(self.predict_sample(x, self.tree)) for x in X])

# 训练与验证
model = DecisionTree(max_depth=3)
y_pred = model(X_test)
print(f"Test Accuracy: {accuracy_score(y_test.numpy(), y_pred.numpy()):.2%}")