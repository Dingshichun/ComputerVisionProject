# LSTM(long short term memory)是长短期记忆网络，
# 解决了 RNN 的梯度消失问题，适合长序列建模。

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 数据预处理函数，明确输入输出维度
def prepare_data(data, n_steps, output_steps=2, test_ratio=0.2):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data.values)

    # 序列构造逻辑
    X, y = [], []
    for i in range(len(scaled_data) - n_steps - output_steps + 1):
        X.append(scaled_data[i : i + n_steps, :])
        y.append(
            scaled_data[i + n_steps : i + n_steps + output_steps, 1:3]
        )  # 保持输入输出维度对齐

    # 添加维度校验
    assert len(X) == len(y), "X与y样本数量不匹配"
    X, y = np.array(X), np.array(y)
    print(
        f"输入维度检查：X.shape={X.shape}, y.shape={y.shape}"
    )  # 应输出(batch, seq_len, features)

    split = int(len(X) * (1 - test_ratio))
    return (
        torch.tensor(X[:split], dtype=torch.float32),
        torch.tensor(y[:split], dtype=torch.float32),
        torch.tensor(X[split:], dtype=torch.float32),
        torch.tensor(y[split:], dtype=torch.float32),
        scaler,
    )


# 模型定义
class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,  # 输入特征的维度
            hidden_size=hidden_size,  # LSTM 每一层的输出维度
            batch_first=True,  # # 是否调整输入输出形状为(batch, seq, feature)，便于处理批量数据
            num_layers=2,  # 堆叠的 LSTM 单元层数
            dropout=0.2,
        )
        # 调整全连接层输出维度计算
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_steps * 2),  # 输出特征数×预测步长
        )
        self.output_steps = output_steps

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])  # 取最后一个时间步
        return out.view(-1, self.output_steps, 2)  # 重塑为(batch, steps, features)


# 维度校验的训练流程
def train_model(model, X_train, y_train, epochs=200):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 维度校验
    print(f"训练数据维度：输入{X_train.shape}，输出{y_train.shape}")
    assert X_train.ndim == 3, "输入必须是三维张量 (batch, seq, features)"
    assert y_train.ndim == 3, "输出必须是三维张量 (batch, steps, features)"

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        # 维度匹配断言
        assert (
            outputs.shape == y_train.shape
        ), f"维度不匹配：模型输出{outputs.shape} vs 目标{y_train.shape}"
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")
    return model


if __name__ == "__main__":
    # 生成演示数据（使用时更改为自己的数据）
    data = pd.DataFrame(
        {
            "time_id": range(1000),
            "veh_n": np.random.randint(50, 200, 1000),
            "v_avg": np.random.uniform(30, 80, 1000),
        }
    )

    # 参数设置
    n_steps = 6  # 历史时间窗口
    output_steps = 2  # 预测未来步长
    features = 3  # 输入特征数（time_id+veh_n+v_avg）

    # 数据准备
    X_train, y_train, X_test, y_test, scaler = prepare_data(
        data[["time_id", "veh_n", "v_avg"]], n_steps=n_steps, output_steps=output_steps
    )

    # 模型初始化
    model = MultiStepLSTM(
        input_size=features, hidden_size=128, output_steps=output_steps
    )

    # 训练
    trained_model = train_model(model, X_train, y_train)

    # 预测与可视化
    with torch.no_grad():
        test_pred = trained_model(X_test).numpy()

    # 反归一化逻辑
    def inverse_scale(pred, scaler):
        dummy = np.zeros((pred.shape[0], pred.shape[1], features))
        dummy[:, :, 1:3] = pred  # 保持与原始特征位置一致
        return scaler.inverse_transform(dummy.reshape(-1, features))[:, 1:3].reshape(
            -1, output_steps, 2
        )

    final_pred = inverse_scale(test_pred, scaler)
    true_values = inverse_scale(y_test.numpy(), scaler)

    # 可视化
    plt.figure(figsize=(15, 6))
    for i in range(2):  # 绘制两个特征
        plt.subplot(1, 2, i + 1)
        plt.plot(true_values[-100:, 0, i], label="Actual")
        plt.plot(final_pred[-100:, 0, i], "--", label="Predicted")
        plt.title(["Vehicle Count", "Average Speed"][i])
        plt.legend()
    plt.tight_layout()
    plt.show()
