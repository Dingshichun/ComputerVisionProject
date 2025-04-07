# 模型训练代码示例

import torch
from torch.utils.data import DataLoader
from ssd import SSD
from dataset import VOCDataset
from loss import MultiBoxLoss

# 参数设置
num_epochs = 100
batch_size = 16
lr = 1e-3

# 数据加载
dataset = VOCDataset(root='VOCdevkit/VOC2007')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型和损失函数
model = SSD(num_classes=21)
criterion = MultiBoxLoss(num_classes=21)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    for images, targets in dataloader:
        pred_loc, pred_conf = model(images)
        loss = criterion(pred_loc, pred_conf, targets['boxes'], targets['labels'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')