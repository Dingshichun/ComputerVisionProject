import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 3 * X + 4 + np.random.randn(100, 1)
x_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y)

model=nn.Linear(1,1)
loss_fn=nn.MSELoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    y_pred=model(x_train)
    loss=loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
x_test=torch.tensor([[3.0]])
print(f"predict value:{model(x_test).item():.2f}")

