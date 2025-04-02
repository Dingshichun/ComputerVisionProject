# 静态量化
# 适用场景：需要校准数据，量化权重和激活值为 int8，适合 CNN 等模型。

import torch
import torch.nn as nn

# 1. 定义模型，插入 QuantStub/DeQuantStub
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub() # 量化输入
        self.linear = nn.Linear(5, 3) # 线性层
        self.dequant = torch.quantization.DeQuantStub() # 反量化输出
    
    def forward(self, x):
        x = self.quant(x) # 量化输入
        x = self.linear(x) 
        x = self.dequant(x) # 反量化输出
        return x

model = Model() # 实例化模型
model.eval() # 设置为评估模式

# 2. 配置量化后端（根据硬件选择）
model.qconfig = torch.quantization.get_default_qconfig('x86')

# 3. 准备模型（插入观察器）
model_prepared = torch.quantization.prepare(model)

# 4. 校准（使用约 100-1000 个样本）
calibration_data = [torch.randn(1, 5) for _ in range(100)]
for sample in calibration_data:
    model_prepared(sample)

# 5. 转换为量化模型
quantized_model = torch.quantization.convert(model_prepared)

# 6. 推理测试
input = torch.randn(1, 5)
output = quantized_model(input)
print(f"量化前模型：{model}")
print(f"量化后的模型：{quantized_model}")
print(f"量化后的模型输出：{output}")