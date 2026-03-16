import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical


# 定义策略网络
class PolicyNet(nn.Module):
    def __init__(self,action_dim):#输入状态维度4，输出的动作维度是2
        super(PolicyNet, self).__init__()
        self.action_dim = action_dim
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=1)
        return action_probs


#定义价值网络
class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value

class Agent:
    def __init__(self):
        self.gamma = 0.98 # 折扣因子
        self.lr = 0.0001 # 学习率
        self.action_size = 2 # 动作空间大小，共两个动作：向左推和向右推

        self.memory = []
        self.pi = PolicyNet(self.action_size) # 初始化策略神经网络
        self.value_net = ValueNet() # 初始化价值网络
        #加载之前训练好的模型参数，如果存在的话
        try:
            self.pi.load_state_dict(torch.load("models/policy_model.pth"))
            self.value_net.load_state_dict(torch.load("models/value_model.pth"))
            print("已加载之前训练好的模型参数。")
        except FileNotFoundError:
            print("未找到之前训练好的模型参数，从零开始训练。")
        # 使用 Adam 优化器
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr)
        # 使用 Adam 优化器更新价值网络的参数
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=self.lr)


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

    def update(self,reward,prob,value_t,value_t1,done):
        loss_pi,loss_value = 0,0
        #这里使用的是TD(0)方法来更新策略网络的参数，计算回报时只考虑当前时间步的奖励和下一时间步的状态价值，而不是整个回合的总奖励。这种方法可以更快地更新策略网络，因为它不需要等待整个回合结束才能计算回报。
        td_target = reward + self.gamma  * value_t1.detach() # 计算 TD 目标，并阻止梯度流入 value_t1（目标网络固定）
        # 优势函数（用于策略梯度），并阻止梯度流入 value_t
        advantage = td_target - value_t.detach()
        # td_error = reward + self.gamma * value_t1 - value_t # 计算TD误差
        loss_pi = -advantage * prob.log() # 策略网络的损失函数，负号是因为我们要最大化回报
        self.optimizer_pi.zero_grad()
        loss_pi.backward()
        self.optimizer_pi.step()
        loss_value = (value_t - td_target).pow(2) # 价值网络的损失函数，使用TD误差的平方来衡量价值网络的预测与实际回报之间的差距
        self.optimizer_value.zero_grad()
        loss_value.backward()
        self.optimizer_value.step()




if __name__ == "__main__":

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
            done = terminated or truncated
            #获取当前时间步的奖励，状态价值和下一时间步的状态价值
            value_t = agent.value_net(torch.tensor(state[np.newaxis,:])).squeeze(0) # 将value_t从形状[1, 1]转换为标量
            if done:
                # 如果回合结束，下一个状态的价值为 0
                value_t1 = torch.zeros(1, 1)
            else:
                next_state_tensor = torch.tensor(next_state).unsqueeze(0)
                value_t1 = agent.value_net(next_state_tensor).detach()
            # value_t1 = agent.value_net(torch.tensor(next_state[np.newaxis,:]))
            # value_t1 = value_t1.squeeze(0) # 将value_t1从形状[1, 1]转换为标量
            #每轮一步后，更新策略网络和价值网络
            agent.update(reward,prob,value_t,value_t1,done)
            total_reward += reward
            state = next_state
        #每轮结束后，记录总奖励
        rewards_history.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}, Average Reward: {np.mean(rewards_history[-100:])}")

    torch.save(agent.pi.state_dict(), "models/policy_model.pth")
    torch.save(agent.value_net.state_dict(), "models/value_model.pth")
    print("训练完成，模型已保存。")

    import matplotlib.pyplot as plt
    # 训练结束后绘制奖励变化图
    plt.plot(rewards_history)
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.title('每回合奖励')
    plt.grid(True)
    plt.show()

