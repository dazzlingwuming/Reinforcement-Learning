import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical


# 定义策略网络
class Policy(nn.Module):
    def __init__(self,action_dim):#输入状态维度4，输出的动作维度是2
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=1)
        return action_probs


class Agent:
    def __init__(self):
        self.gamma = 0.98 # 折扣因子
        self.lr = 0.0002 # 学习率
        self.action_size = 2 # 动作空间大小，共两个动作：向左推和向右推

        self.memory = []
        self.pi = Policy(self.action_size) # 初始化策略神经网络
        #加载之前训练好的模型参数，如果存在的话
        try:
            self.pi.load_state_dict(torch.load("models/policy_model.pth"))
            print("已加载之前训练好的模型参数。")
        except FileNotFoundError:
            print("未找到之前训练好的模型参数，从零开始训练。")
        # 使用 Adam 优化器
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)

    def get_action(self, state):
        # 将状态转换成torch.tensor类型
        state = torch.tensor(state[np.newaxis,:])#将状态转换成torch.tensor类型，并添加一个维度，使其成为一个批次（batch）输入
        # 将状态输入策略神经网络，输出为两个动作的概率分布
        probs = self.pi(state)
        # 取出概率分布
        probs = probs[0]
        # 下面两行根据动作的分布采样出一个动作，创建了一个“按概率抽签”的分布对象
        m = Categorical(probs)
        action = m.sample().item()

        # 返回动作和动作的概率
        return action, probs[action]

    #保存回报和动作概率，用于计算后续的奖励，逆序计算
    def add(self,reward,prob):
        self.memory.append((reward,prob))
    def updata(self):
        G,loss = 0,0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G # 计算回报
        for reward, prob in self.memory:
            loss += -G * prob.log() # 计算损失函数，负号是因为我们要最大化回报

        # 反向传播
        self.optimizer.zero_grad() # 清空之前的梯度
        loss.backward() # 计算当前损失的梯度
        self.optimizer.step() # 更新策略网络的参数
        self.memory = []#清空记忆，为下一轮训练做准备



if __name__ == "__main__":
    # env = gym.make("CartPole-v1", render_mode="rgb_array")
    # state = env.reset()[0] # 获取初始状态
    # agent = Agent()
    # action , prob = agent.get_action(state)
    # print("action,prob",action, prob.item())
    # G = 100.0#虚拟权重
    # J = G*prob.log()#动作概率的对数乘以权重，作为损失函数
    # print("损失函数:", J.item())
    # #求梯度
    # J.backward()

    #在倒立摆环境中训练智能体
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    agent = Agent()
    episodes = 4000
    rewards_history = []
    for episode in range(episodes):
        #重置环境，获取初始状态
        state,_ = env.reset()
        done = False
        total_reward = 0
        while not done:
            #根据当前状态选择动作
            action, prob = agent.get_action(state)
            #执行动作，观察结果
            next_state, reward, terminated,truncated,info = env.step(action)

            #将奖励和动作概率保存到记忆中
            agent.add(reward, prob)
            total_reward += reward
            state = next_state
            done = terminated or truncated
        #每轮结束后，更新策略网络
        agent.updata()
        rewards_history.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Average Reward: {np.mean(rewards_history[-100:])}")

    torch.save(agent.pi.state_dict(), "models/policy_model.pth")
    print("训练完成，模型已保存。")

    import matplotlib.pyplot as plt
    # 训练结束后绘制奖励变化图
    plt.plot(rewards_history)
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.title('每回合奖励')
    plt.grid(True)
    plt.show()

