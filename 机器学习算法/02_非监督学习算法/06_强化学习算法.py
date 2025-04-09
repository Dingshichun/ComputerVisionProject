# 强化学习：DQN 算法，是一种深度强化学习算法，使用神经网络来近似 Q 值函数。
# 该算法使用经验回放和目标网络来提高训练的稳定性和效率。


import gym  # gym 是一个用于开发和比较强化学习算法的工具包。
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque  # deque 是双端队列，可以在两端高效地添加和删除元素。
import random

# 超参数
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
LEARNING_RATE = 0.001
MEMORY_CAPACITY = 10000
NUM_EPISODES = 500


# DQN网络定义
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.fc(x)


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)),
        )

    def __len__(self):
        return len(self.buffer)


# 初始化环境和模型
env = gym.make("CartPole-v1", render_mode="human")  # 添加render_mode参数
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
buffer = ReplayBuffer(MEMORY_CAPACITY)
epsilon = EPSILON_START

# 训练循环
for episode in range(NUM_EPISODES):
    # 处理新版reset返回值
    try:
        state, _ = env.reset()  # Gym 0.26+返回（obs, info）
    except:
        state = env.reset()  # 兼容旧版本

    episode_reward = 0

    while True:
        # Epsilon-greedy策略
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        # 处理新版step返回值
        try:
            next_state, reward, terminated, truncated, _ = env.step(
                action
            )  # 新版返回5个值
            done = terminated or truncated
        except:
            next_state, reward, done, _ = env.step(action)  # 旧版返回4个值

        episode_reward += reward

        # 存储经验
        buffer.push(state, action, reward, next_state, done)
        state = next_state

        # 经验回放
        if len(buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

            # 计算当前Q值
            current_q = policy_net(states).gather(1, actions.unsqueeze(1))

            # 计算目标Q值
            with torch.no_grad():
                next_q = target_net(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * GAMMA * next_q

            # 计算损失
            loss = nn.MSELoss()(current_q.squeeze(), target_q)

            # 优化步骤
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # 更新epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # 更新目标网络
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(
        f"Episode: {episode:03d}, Reward: {episode_reward:5.1f}, Epsilon: {epsilon:.3f}"
    )

# 测试训练好的模型
state, _ = env.reset()
while True:
    env.render()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = policy_net(state_tensor).argmax().item()

    try:
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    except:
        next_state, reward, done, _ = env.step(action)

    state = next_state
    if done:
        break

env.close()
