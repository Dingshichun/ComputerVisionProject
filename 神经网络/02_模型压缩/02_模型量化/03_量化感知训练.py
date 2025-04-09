# 量化感知训练，也称为 QAT（Quantization Aware Training），
# 是一种在训练过程中考虑量化误差的技术。它通过模拟量化过程来提高模型在量化后的性能。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# 1. 定义支持量化的模型
class QuantizableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()  # 量化输入
        self.linear = nn.Linear(5, 3)  # 线性层
        self.dequant = torch.quantization.DeQuantStub()  # 反量化输出

    def forward(self, x):
        x = self.quant(x)
        x = self.linear(x)
        x = self.dequant(x)
        return x


model = QuantizableModel()  # 实例化模型
model.train()  # 设置为训练模式

# 2. 设置量化配置
model.qconfig = torch.quantization.get_default_qat_qconfig("x86")

# 3. 准备量化感知训练
model_prepared = torch.quantization.prepare_qat(model)

# 4. 正常训练循环
optimizer = optim.Adam(model_prepared.parameters())

# 创建输入数据和目标数据
input_data = torch.randn(100, 5)  # 100 个样本，每个样本有 5 个特征
target_data = torch.randn(100, 3)  # 100 个样本，每个样本有 3 个目标值

# 使用 TensorDataset 将输入和目标数据打包
dataset = TensorDataset(input_data, target_data)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
for epoch in range(10):
    for data, target in dataloader:
        optimizer.zero_grad()  # 清零梯度
        output = model_prepared(data)
        loss = nn.MSELoss()(output, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

# 5. 转换为量化模型
quantized_model = torch.quantization.convert(model_prepared)

# 6. 测试推理
input = torch.randn(1, 5)
output1= model(input)
output2 = quantized_model(input)
print(f"量化前模型：{model}")
print(f"量化后的模型：{quantized_model}")
print(f"量化前模型输出：{output1}")
print(f"量化后的模型输出：{output2}")

# 保存量化模型
# torch.jit.save(torch.jit.script(quantized_model), "quantized_model.pth")

# 加载量化模型
# loaded_model = torch.jit.load("quantized_model.pth")