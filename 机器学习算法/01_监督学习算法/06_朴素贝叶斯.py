# 朴素贝叶斯是一种简单而有效的分类算法，适用于文本分类、垃圾邮件过滤等任务。
# 它基于贝叶斯定理，假设特征之间是条件独立的。
# 朴素贝叶斯分类器通常用于处理高维数据，尤其是在文本分类中表现良好。
# 简单、快速、易于实现，但对特征之间的独立性假设过于强烈，可能导致性能下降。
# 适用于大规模数据集，尤其是文本数据。

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None  # 类别标签
        self.class_priors = {}  # 先验概率
        self.means = {}  # 均值
        self.variances = {}  # 方差

    def fit(self, X, y):
        self.classes = torch.unique(y)  # 获取唯一类别标签
        n_samples, n_features = X.shape  # 样本数和特征数

        # 计算先验概率和特征统计量
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c.item()] = X_c.shape[0] / n_samples
            self.means[c.item()] = torch.mean(X_c, dim=0)
            self.variances[c.item()] = torch.var(X_c, dim=0, unbiased=False)  # 无偏方差

    def predict(self, X):
        probs = []  # 存储每个类别的后验概率
        for c in self.classes:
            prior = torch.log(torch.tensor(self.class_priors[c.item()]))
            mean = self.means[c.item()]
            var = self.variances[c.item()] + 1e-9  # 防止除以0

            # 高斯概率密度计算（对数形式）
            exponent = -0.5 * torch.sum((X - mean) ** 2 / var, dim=1)
            gaussian_log_prob = exponent - 0.5 * torch.sum(
                torch.log(2 * torch.pi * var)
            )
            posterior = prior + gaussian_log_prob
            probs.append(posterior.unsqueeze(1))

        # 拼接概率并取最大值对应类别
        probs = torch.cat(probs, dim=1)
        return torch.argmax(probs, dim=1)


if __name__ == "__main__":
    # 数据加载与划分
    iris = load_iris()
    X = torch.tensor(iris.data, dtype=torch.float32)
    y = torch.tensor(iris.target, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), test_size=0.2, random_state=42
    )
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 模型训练与预测
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # 计算准确率
    accuracy = (predictions == y_test).float().mean()
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")
