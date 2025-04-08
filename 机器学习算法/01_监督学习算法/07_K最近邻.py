# KNN 是一种监督学习算法，主要用于分类和回归。
# 它的基本思想是：给定一个新的样本点，找到训练集中与该样本点距离最近的 K 个样本点，
# 然后根据这 K 个样本点的标签来预测该样本点的标签。
# KNN 算法的优点是简单易懂，易于实现，适用于小规模数据集。
# 缺点是计算复杂度高，对噪声敏感，存储空间大，不适用于大规模数据集。

# KNN 算法的应用场景包括：
# 1. 图像识别：KNN 可以用于图像分类，如手写数字识别、人脸识别等。
# 2. 文本分类：KNN 可以用于文本分类，如垃圾邮件过滤、情感分析等。
# 3. 推荐系统：KNN 可以用于推荐系统，如电影推荐、商品推荐等。
# 4. 医疗诊断：KNN 可以用于医疗诊断，如疾病预测、药物反应预测等。
# 5. 客户细分：KNN 可以用于客户细分，如市场细分、客户行为分析等。
# 6. 股票预测：KNN 可以用于股票预测，如股票价格预测、股票涨跌预测等。
# 7. 信用评分：KNN 可以用于信用评分，如信用卡申请审核、贷款申请审核等。

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # 存储训练数据（无需显式训练过程）
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        # 计算所有测试样本与训练样本的距离矩阵
        distances = torch.cdist(X_test, self.X_train, p=2)  # 欧氏距离
        # 获取前k个最小距离的索引
        _, topk_indices = torch.topk(distances, self.k, largest=False, dim=1)
        # 根据索引获取对应的标签
        topk_labels = self.y_train[topk_indices]
        # 统计多数类别
        y_pred, _ = torch.mode(topk_labels, dim=1)
        return y_pred


# 示例应用：鸢尾花分类
# 加载数据
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
) # 70% 训练集，30% 测试集，随机种子 42

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test, dtype=torch.float32)

# 训练与预测
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = (y_pred == torch.tensor(y_test)).float().mean()
print(f"准确率: {accuracy:.4f}")
