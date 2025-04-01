"""
需要预先下载 MNIST 数据集，或者在运行时自动下载
"""

# 1.导入库
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 2.定义超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
batch_size = 64
lr = 0.0002
num_epochs = 50

# 3. 数据预处理与加载
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # 将像素值归一化到[-1, 1]
    ]
)

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 4.定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),  # 输出范围[-1, 1]
        )

    def forward(self, z):
        img = self.main(z)
        return img.view(-1, 1, 28, 28)


# 5.定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 输出概率值
        )

    def forward(self, img):
        flattened = img.view(img.size(0), -1)
        validity = self.main(flattened)
        return validity


# 6. 初始化模型、优化器和损失函数
generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()

optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# 7. 训练循环
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        # 真实标签和假标签
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # 训练判别器
        optimizer_D.zero_grad()

        # 真实图像的损失
        outputs_real = discriminator(real_imgs)
        d_loss_real = criterion(outputs_real, real_labels)

        # 生成假图像
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)

        # 假图像的损失
        outputs_fake = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()

        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # 打印训练状态
        if i % 200 == 0:
            print(
                f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] "
                f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}"
            )

    # 每个epoch结束后保存生成的图像
    with torch.no_grad():
        test_z = torch.randn(16, latent_dim).to(device)
        generated = generator(test_z).cpu()
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for idx, ax in enumerate(axes.flatten()):
            ax.imshow(generated[idx].squeeze(), cmap="gray")
            ax.axis("off")
        plt.savefig(f"epoch_{epoch}.png")
        plt.close()

# 8. 生成示例图像（训练后）
# 加载训练好的模型
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()

# 生成新图像
with torch.no_grad():
    z = torch.randn(16, latent_dim).to(device)
    samples = generator(z).cpu()

# 可视化结果
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(samples[idx].squeeze(), cmap="gray")
    ax.axis("off")
plt.show()
