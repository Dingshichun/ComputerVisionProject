"""
PyTorch 提供了 torch.nn.utils.prune 模块，支持结构化/非结构化剪枝。
结构化剪枝是指对整个神经元、通道或层进行剪枝，而非结构化剪枝是指对单个权重进行剪枝。
结构化剪枝通常更易于实现和优化，因为它们可以直接影响模型的计算图，
而非结构化剪枝可能会导致稀疏矩阵的计算效率低下。
但是，非结构化剪枝可以提供更高的灵活性和精确度，因为它们可以针对特定的权重进行优化。
"""

# 1. ​非结构化剪枝（Unstructured Pruning）​
# 剪除权重矩阵中不重要的单个权重（例如：将部分权重置零）。

import torch
import torch.nn.utils.prune as prune

# 定义一个简单模型
model = torch.nn.Sequential(
    torch.nn.Linear(10, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2)
)


# 选择要剪枝的层（例如第一个全连接层）
module = model[0]
print("第一个全连接层剪枝前的权重：")
print(model[0].weight)  # 查看第一个全连接层的原始权重

# 应用 L1 非结构化剪枝（剪枝比例 20% ）
prune.l1_unstructured(module, name="weight", amount=0.2)

# 查看剪枝后的权重掩码（mask），掩码中值为 0 的权重被剪枝
print("剪枝的权重掩码：")
print(module.weight_mask)

# 永久移除剪枝（将 mask 应用到 weight 并删除 mask 参数）
prune.remove(module, "weight")
print("第一个全连接层剪枝后的权重：")
print(model[0].weight)  # 查看剪枝后第一个全连接层的权重，可以看到部分权重被置为 0


# 2. ​结构化剪枝（Structured Pruning）​
# 相当于移除某一层的某些神经元和下一层的某个或多个神经元的连接。
# 对比剪枝前后权重的变化即可了解实际含义，剪枝后的权重矩阵会变得更小。

# 对第一个全连接层进行通道剪枝（剪枝比例 40%）
# 这里的 dim=0 表示按行剪枝（即按通道剪枝），n=2 表示剪枝 2 个通道
prune.ln_structured(module, name="weight", amount=0.4, n=2, dim=0)
print("结构化剪枝后的权重：")
print(model[0].weight)
