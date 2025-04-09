# DQN 算法是

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym
import matplotlib.pyplot as plt


# Q 网络定义（添加 Double DQN 支持）
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)  # 添加正则化

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 改进的 DQN 智能体
class DQNAgent:
    def __init__(self, state_size, action_size, use_double_dqn=True):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # 增大经验回放缓冲区
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 128
        self.update_every = 100  # 目标网络更新频率（步数）
        self.use_double_dqn = use_double_dqn

        # 初始化网络
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0005)
        self.criterion = nn.MSELoss()

        # 初始同步目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 训练计数器
        self.train_step = 0

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # 从经验回放中采样
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])

        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值（使用Double DQN）
        with torch.no_grad():
            if self.use_double_dqn:
                # 使用主网络选择动作，目标网络评估
                next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                target_q = self.target_network(next_states).gather(1, next_actions)
            else:
                # 原始DQN方式
                target_q = self.target_network(next_states).max(1)[0].unsqueeze(1)

            target_q = (
                rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * target_q
            )

        # 计算损失
        loss = self.criterion(current_q, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # 更新目标网络（固定间隔）
        self.train_step += 1
        if self.train_step % self.update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def update_epsilon(self):
        """在每个episode后更新epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]


# 改进的训练函数
def train_dqn(env, agent, episodes=500, render_interval=50):
    scores = []
    moving_avg = []

    for episode in range(episodes):
        state, _ = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            score += reward

            # 保存经验
            agent.remember(state, action, reward, next_state, done)

            # 学习
            agent.learn()

            state = next_state
            done = done or truncated

        # 更新epsilon
        agent.update_epsilon()

        # 记录得分
        scores.append(score)
        moving_avg.append(np.mean(scores[-100:]))

        # 输出训练信息
        if (episode + 1) % render_interval == 0:
            print(
                f"Episode: {episode+1}, Score: {score:.2f}, Avg Score: {moving_avg[-1]:.2f}, Epsilon: {agent.epsilon:.4f}"
            )

    # 绘制训练曲线
    plt.plot(scores, label="Score")
    plt.plot(moving_avg, label="Moving Average")
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    return scores


# 测试函数
def test_agent(env, agent, episodes=10, render=True):
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            score += reward
            state = next_state
            done = done or truncated

            if render:
                env.render()

        print(f"Test Episode: {episode+1}, Score: {score}")

    env.close()


# 主程序
if __name__ == "__main__":
    # 创建环境
    env = gym.make("CartPole-v1", render_mode="human")  # 使用新版gym的render模式
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 初始化智能体
    agent = DQNAgent(state_size, action_size, use_double_dqn=True)

    # 开始训练
    print("Start training...")
    scores = train_dqn(env, agent, episodes=300)

    # 保存模型
    agent.save("dqn_model.pth")

    # 测试训练结果
    print("\nTesting trained agent...")
    test_env = gym.make("CartPole-v1", render_mode="human")
    test_agent(test_env, agent)

    env.close()
