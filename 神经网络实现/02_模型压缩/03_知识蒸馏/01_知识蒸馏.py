# 知识蒸馏（Knowledge Distillation）是一种将大型模型（教师模型）
# 的知识迁移到小型模型（学生模型）的技术，常用于模型压缩或提升小模型的性能。

# 在化学中，蒸馏是一种有效的分离不同沸点组分的方法，大致步骤是先升温使低沸点的组分汽化，
# 然后降温冷凝，达到分离出目标物质的目的。
# 化学蒸馏条件：（1）蒸馏的液体是混合物；（2）各组分沸点不同。
# 蒸馏的液体是混合物，这个混合物一定是包含了各种组分，
# 即在我们今天讲的知识蒸馏中指原模型包含大量的知识。
# 各组分沸点不同，蒸馏时要根据目标物质的沸点设置蒸馏温度，

# 知识蒸馏的基本思想是通过让学生模型模仿教师模型的输出分布来学习。
# 具体来说，知识蒸馏通常包括以下几个步骤:
# 1. 训练教师模型：首先训练一个大型的、性能较好的教师模型。
# 2. 生成软标签：使用教师模型对训练数据进行预测，得到每个样本的软标签（即概率分布），
# 而不是硬标签（即one-hot编码）。
# 3. 训练学生模型：使用软标签来训练一个小型的学生模型，
# 使其能够模仿教师模型的输出分布。通常，学生模型的损失函数包括两个部分：
#    - 传统的交叉熵损失，用于与真实标签进行比较。
#    - 蒸馏损失，用于与教师模型的软标签进行比较。
# 4. 调整超参数：可以通过调整温度参数来控制软标签的平滑程度，
# 使得学生模型更容易学习到教师模型的知识。
# 5. 评估学生模型：在验证集或测试集上评估学生模型的性能，
# 并与教师模型进行比较。
# 6. 部署学生模型：如果学生模型的性能足够好，可以将其部署到生产环境中，
# 以替代大型的教师模型。
# 7. 迭代优化：可以根据需要迭代优化学生模型，
# 例如通过进一步的训练或调整超参数等方式。
# 8. 监控与维护：在生产环境中监控学生模型的性能，
# 及时进行维护和更新，以确保其持续有效。
# 9. 扩展应用：可以将知识蒸馏技术应用于其他任务或领域，


import torch
import torch.nn as nn
import torch.optim as optim

# • 软标签（Soft Labels）​：教师模型输出的概率分布（带温度参数的温度缩放）包含更多信息。
# • ​温度参数（Temperature）​：软化概率分布，使学生模型学习到类别间的关系。
# • ​损失函数：结合软标签损失（KL 散度）和硬标签损失（交叉熵）。

# 1.准备教师模型和学生模型
# 假设教师模型和学生模型已定义。实际并未定义，所以代码无法运行，仅供参考。
teacher_model = TeacherModel().eval()  # 教师模型设为评估模式
student_model = StudentModel()


# 2.定义蒸馏损失函数
def distillation_loss(student_output, teacher_output, temperature, alpha):
    # 软标签损失（KL散度）
    soft_loss = nn.KLDivLoss(reduction="batchmean")(
        torch.log_softmax(student_output / temperature, dim=1),
        torch.softmax(teacher_output / temperature, dim=1),
    ) * (
        temperature**2
    )  # 缩放梯度

    # 硬标签损失（交叉熵）
    hard_loss = nn.CrossEntropyLoss()(student_output, labels)

    # 总损失 = α * 软标签损失 + (1-α) * 硬标签损失
    return alpha * soft_loss + (1 - alpha) * hard_loss


# 3.定义训练函数
def train_distillation(
    teacher_model, student_model, train_loader, optimizer, temperature, alpha, epochs
):
    student_model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            student_outputs = student_model(inputs)

            # 计算蒸馏损失
            loss = distillation_loss(
                student_outputs, teacher_outputs, temperature, alpha
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# 定义超参数
temperature = 5.0  # 温度参数（通常 3~10）
alpha = 0.7  # 软标签损失权重
learning_rate = 1e-3
epochs = 10

# 优化器
optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

# 训练蒸馏
train_distillation(
    teacher_model=teacher_model,
    student_model=student_model,
    train_loader=train_loader,
    optimizer=optimizer,
    temperature=temperature,
    alpha=alpha,
    epochs=epochs,
)
