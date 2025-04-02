# 这是一份完整的、可运行的 PyTorch 知识蒸馏代码示例，
# 包含数据加载、模型定义、蒸馏训练和评估全流程。
# 代码以 CIFAR-10 数据集为例，使用 ResNet-18 作为教师模型，简单 CNN 作为学生模型。

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  # 数据加载器
from tqdm import tqdm  # 进度条库

# 超参数配置
config = {
    "batch_size": 256,
    "teacher_epochs": 10,
    "student_epochs": 20,
    "lr": 0.001,
    "temperature": 4.0,  # 蒸馏温度
    "alpha": 0.7,  # 软标签损失权重
    "num_classes": 10,  # CIFAR-10 类别数
}

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# 加载 CIFAR-10 数据集
# 如果已经有下载好的数据集，则设置 download=False，避免重复下载
# 下载好的数据集路径为 ./data/cifar-10-batches-py。
# 注意，需要将下载好的文件解压缩才能得到可以使用的 cifar-10-batches-py 文件夹。
train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
)

train_loader = DataLoader(
    train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)


# 定义教师模型（ResNet-18）
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(num_classes=config["num_classes"])

    def forward(self, x):
        return self.resnet(x)


# 定义学生模型（简单 CNN）
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, config["num_classes"])

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# 训练教师模型
def train_teacher():
    teacher = TeacherModel().to(device) # 教师模型在 GPU 上训练
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.Adam(teacher.parameters(), lr=config["lr"])
    
    # 每 5 个 epoch 学习率衰减 0.1 倍
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    print("Training Teacher Model...")
    for epoch in range(config["teacher_epochs"]):
        teacher.train() # 设置模型为训练模式
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            # 将数据和标签移动到 GPU 上
            inputs, labels = inputs.to(device), labels.to(device) 

            optimizer.zero_grad() # 清空梯度
            outputs = teacher(inputs) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            total_loss += loss.item() # 累计损失

        scheduler.step() # 更新学习率
        print(
            f"Epoch [{epoch+1}/{config['teacher_epochs']}] Loss: {total_loss/len(train_loader):.4f}"
        )

    return teacher # 返回训练好的教师模型


# 知识蒸馏训练
def distill(teacher, student):
    student = student.to(device) # 学生模型在 GPU 上训练
    optimizer = optim.Adam(student.parameters(), lr=config["lr"])
    ce_loss = nn.CrossEntropyLoss()

    print("\nDistilling Student Model...")
    for epoch in range(config["student_epochs"]):
        student.train() # 设置模型为训练模式
        total_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            # 将数据和标签移动到 GPU 上
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad(): # 教师模型不需要梯度
                teacher_logits = teacher(inputs) # 教师模型输出 logits 

            student_logits = student(inputs) # 学生模型输出 logits

            # 计算软目标损失
            soft_loss = nn.KLDivLoss(reduction="batchmean")(
                torch.log_softmax(student_logits / config["temperature"], dim=1),
                torch.softmax(teacher_logits / config["temperature"], dim=1),
            ) * (config["temperature"] ** 2)

            # 计算硬目标损失
            hard_loss = ce_loss(student_logits, labels)

            # 组合损失
            loss = config["alpha"] * soft_loss + (1 - config["alpha"]) * hard_loss

            optimizer.zero_grad() # 清空梯度
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            total_loss += loss.item() # 累计损失

        print(
            f"Epoch [{epoch+1}/{config['student_epochs']}] Loss: {total_loss/len(train_loader):.4f}"
        )

    return student # 返回训练好的学生模型


# 评估函数
def evaluate(model, dataloader):
    model = model.to(device)  # 确保评估时模型在 GPU
    model.eval() # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad(): # 评估时不需要计算梯度
        for inputs, labels in dataloader:
            # 将数据和标签移动到 GPU 上
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# 完整流程
if __name__ == "__main__":
    # 训练教师模型
    teacher = train_teacher()
    teacher_accuracy = evaluate(teacher, test_loader)
    print(f"\nTeacher Model Test Accuracy: {teacher_accuracy:.2f}%")

    # 初始化学生模型
    student = StudentModel()
    student_accuracy = evaluate(student, test_loader)
    print(f"Student Model Baseline Accuracy: {student_accuracy:.2f}%")

    # 执行知识蒸馏
    distilled_student = distill(teacher, student)
    distilled_accuracy = evaluate(distilled_student, test_loader)
    print(f"\nDistilled Student Model Accuracy: {distilled_accuracy:.2f}%")
