# 动态量化是指在模型推理时对权重进行量化。
# 适用场景：适用于 LSTM、线性层等，量化权重为 int8，但激活值仍为 float32。

# 在编写动态量化的例子时，需要加载模型，设置为 eval 模式，
# 然后使用 torch.quantization.quantize_dynamic。
# 这里要注意，可能需要指定要量化的层类型，比如 nn.Linear 和 nn.LSTM。
# 保存和加载量化模型需要使用 torch.jit.save 和 torch.jit.load。


import torch
import torch.nn as nn

# 1. 定义模型
model = nn.Sequential(nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 2))
print(f"量化前模型参数：{model[0].weight}")
print(f"量化前的模型：{model}")

# 2. 设置为评估模式
model.eval()

# 3. 动态量化（量化权重为 int8）
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8  # 指定要量化的层类型
)
print(f"量化后的模型：{quantized_model}")

# 4. 测试推理
input = torch.randn(1, 5)
output_origin = model(input)
output_quantized = quantized_model(input)
print(f"量化前模型输出：{output_origin}")
print(f"量化后的模型输出：{output_quantized}")

# 5. 保存量化模型
# torch.jit.save(torch.jit.script(quantized_model), "quantized_model.pth")
