"""
通过自定义函数实现更灵活的剪枝逻辑。
剪枝方法可以是基于权重大小、梯度信息、重要性评分等。
注意事项：
1. ​剪枝后的模型性能：剪枝可能导致精度下降，需通过微调（ Fine-tuning ）恢复性能。
2. ​剪枝粒度：非结构化剪枝更灵活但难以加速，结构化剪枝兼容硬件优化。
3. ​保存与加载剪枝模型：使用 state_dict() 保存时，注意处理掩码（ mask ）参数。
"""

import torch
import torch.nn.utils.prune as prune

# 定义一个简单模型
model = torch.nn.Sequential(
    torch.nn.Linear(5, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)
)


# 1.基于权重大小的剪枝
def custom_prune(module, amount=0.2):
    """权重的绝对值小于阈值的权重被剪枝"""
    # 获取权重
    weights = module.weight.data
    # 计算剪枝阈值（取绝对值后的权重）
    threshold = torch.quantile(torch.abs(weights).flatten(), amount)
    print(f"剪枝的权重阈值是: {threshold}")
    # 创建掩码（mask）
    mask = torch.abs(weights) > threshold
    # 应用掩码
    module.weight.data = weights * mask.float()


print("剪枝前的权重：")
print(model[0].weight)  # 查看第一个全连接层的原始权重

# 应用自定义剪枝
custom_prune(model[0], amount=0.3)

print("剪枝后的权重：")
print(model[0].weight)  # 查看剪枝后第一个全连接层的权重。


# 2. 基于梯度的剪枝。使用梯度信息来决定哪些权重被剪枝。
def gradient_based_prune(model, inputs, targets, amount=0.2):
    # 前向传播和反向传播计算梯度
    outputs = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(outputs, targets)
    loss.backward()

    # 获取梯度绝对值
    gradients = torch.abs(model[0].weight.grad)
    # 计算阈值并生成掩码
    threshold = torch.quantile(gradients.flatten(), amount)
    mask = gradients > threshold

    # 应用掩码并清除梯度
    model[0].weight.data *= mask.float()
    model.zero_grad()


model1 = torch.nn.Sequential(
    torch.nn.Linear(5, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)
)

# 3.迭代剪枝与微调（Iterative Pruning）
"""
# 需要定义一个训练函数 train_model 和数据加载器 train_loader才能运行。
# 在训练过程中逐步剪枝并微调模型。
for epoch in range(10):
    # 训练阶段
    # 假设 train_model 是训练函数，train_loader 是数据加载器
    train_model(model, train_loader) 

    # 每 2 个 epoch 剪枝一次
    if epoch % 2 == 0:
        prune.l1_unstructured(model[0], name='weight', amount=0.1)
        prune.l1_unstructured(model[2], name='weight', amount=0.1)

# 最终移除掩码
prune.remove(model1[0], 'weight')
prune.remove(model1[2], 'weight')
"""

# 4.全局剪枝（Global Pruning）
# 跨层统一剪枝，选择全局重要性最高的权重。
parameters_to_prune = [(model1[0], "weight"), (model1[2], "weight")]

# 全局剪枝（剪枝比例 20%）
prune.global_unstructured(
    parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2
)
