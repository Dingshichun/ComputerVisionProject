# 支持向量机（SVM）是一个强大的分类算法，尤其适用于高维数据。
# 下面是一个使用 PyTorch 实现的线性 SVM 的示例代码。
# 使用 Hinge Loss 作为损失函数，并实现一个简单的训练和评估流程。

import torch
import torch.nn as nn
from sklearn.datasets import make_classification  # 用于生成二分类数据
from sklearn.model_selection import train_test_split  # 划分训练集和测试集


# 1. 数据准备
def prepare_data():
    # 生成二分类数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )  # 1000 个样本，每个样本 20 个特征，样本分为 2 个类别，随机种子为 42
    y = torch.where(torch.from_numpy(y) == 0, -1.0, 1.0)  # 标签转换为-1/1

    # 数据标准化
    X = torch.from_numpy(X).float()
    X = (X - X.mean(dim=0)) / X.std(dim=0)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), test_size=0.2
    ) # 80% 训练集，20% 测试集
    return (
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test),
    ) 


# 2. SVM 模型定义
class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.linear.bias.data.zero_()  # 初始化偏置为 0

    def forward(self, x):
        return self.linear(x).squeeze()  # 输出形状(batch_size,)


# 3. Hinge Loss 定义
def hinge_loss(output, target):
    # torch.clamp 函数用于限制输出值的范围，
    # 小于 min 的值被设为 min，大于 max 的值被设为 max
    loss = torch.mean(torch.clamp(1 - target * output, min=0))
    return loss


# 4. 训练流程
def train_svm(model, X_train, y_train, epochs=500):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = hinge_loss(outputs, y_train)

        # 反向传播与优化
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    return model


# 5. 评估模块
def evaluate(model, X_test, y_test):
    with torch.no_grad(): # 评估时不需要计算梯度
        outputs = model(X_test)
        predicted = torch.sign(outputs) # 预测标签为 -1 或 1
        accuracy = (predicted == y_test).float().mean() 
        print(f"\nTest Accuracy: {accuracy.item()*100:.2f}%")

        # 混淆矩阵
        confusion = torch.zeros(2, 2)
        for t, p in zip(y_test, predicted):
            confusion[int(t > 0), int(p > 0)] += 1
        print(f"Confusion Matrix:\n{confusion}")


# 核函数扩展示例（RBF核）
class KernelSVM(nn.Module):
    def __init__(self, input_dim, gamma=1.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.gamma = gamma

    def rbf_kernel(self, x1, x2):
        return torch.exp(-self.gamma * torch.cdist(x1, x2) ** 2)

    def forward(self, x):
        kernel_out = self.rbf_kernel(x, self.support_vectors)
        return self.linear(kernel_out)


# 主程序流程
if __name__ == "__main__":
    # 准备数据
    X_train, y_train, X_test, y_test = prepare_data()

    # 初始化模型
    model = LinearSVM(input_dim=X_train.shape[1])

    # 训练模型
    trained_model = train_svm(model, X_train, y_train)

    # 评估模型
    evaluate(trained_model, X_test, y_test)
