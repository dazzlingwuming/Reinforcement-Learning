import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Categorical
import torch.nn.functional as F



# -------------------- 1. 定义 Actor-Critic 网络 --------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,hidden_size=64):
        super().__init__()
        #输入维度4，输出维度1，隐藏层大小64
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Softmax(dim=-1)  # 适用于离散动作
        )
        #
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs,value


class Agent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, clip_epsilon=0.2):
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.device = torch.device("cpu")
        self.actorcritic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actorcritic.parameters(), lr=lr)

        # 尝试加载模型
        try:
            self.actorcritic.load_state_dict(torch.load("models/actorcritic_model.pth"))
            print("已加载之前训练好的模型参数。")
        except FileNotFoundError:
            print("未找到之前训练好的模型参数，从零开始训练。")


    def get_action(self, state):
        """给定单个状态，返回动作、log_prob 和 价值（用于推理或收集数据）"""
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs, value = self.actorcritic(state)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

# -------------------- 2. GAE 计算函数 --------------------
    def compute_gae(self,rewards, dones,values, next_value,):
        """
        计算广义优势估计 (GAE) 和回报 (returns)
        参数:
            rewards: 列表，每个时间步的奖励 [r0, r1, ..., r_{T-1}]
            masks: 列表，终止标志 (1.0 表示非终止，0.0 表示终止)
            values: 列表，critic 对每个状态的价值估计 [v0, v1, ..., v_{T-1}]
            next_value: float，最后一个状态之后的状态的价值 v_T
            gamma: 折扣因子
            lam: GAE lambda 参数
        返回:
            advantages: Tensor，优势估计
            returns: Tensor，折扣回报（目标值）
        """
        rewards = torch.tensor(rewards)
        values = torch.tensor(values)
        # 将所有价值拼接，最后一个是 next_value
        all_values = torch.cat([values, torch.tensor([next_value])])

        gae = 0
        advantages = torch.zeros_like(rewards)
        # 从后向前计算 GAE
        for t in reversed(range(len(rewards))):
            # TD 误差: δ_t = r_t + γ * V(s_{t+1}) * mask - V(s_t)
            delta = rewards[t] + self.gamma * all_values[t + 1] * (1 - dones[t]) - all_values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

#-------------------- 3. PPO 更新函数 --------------------

    def update(self, states, actions, old_log_probs, returns, advantages):
        """单次 PPO 更新（在一个 mini-batch 上）"""
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        probs, values = self.actorcritic(states)
        values = values.squeeze()
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # 重要性采样比率
        ratios = torch.exp(log_probs - old_log_probs)

        # 裁剪的 surrogate 目标
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # 价值函数损失（MSE）
        critic_loss = F.mse_loss(values, returns)

        # 总损失 熵系数固定为0.01，价值系数0.5，
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # -------------------- 4. 主训练循环 --------------------
    def train_ppo(self, env, num_episodes=1000, rollout_steps=500, ppo_epochs=10, mini_batch_size=64):
        """主训练循环"""
        total_steps = 0
        rewards_history = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            # 存储轨迹数据
            states = []
            actions = []
            rewards = []
            dones = []  # 1 表示终止，0 表示未终止
            old_log_probs = []
            values = []

            episode_reward = 0
            step = 0

            # 收集一条轨迹（直到达到 rollout_steps 或自然结束）
            while step < rollout_steps:
                action, log_prob, value = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(float(done))  # 终止为1
                old_log_probs.append(log_prob)
                values.append(value)

                state = next_state
                episode_reward += reward
                step += 1
                total_steps += 1

                if done:
                    state, _ = env.reset()
                    # 如果提前终止，也可以继续收集（重置环境后继续）
                    # 但注意此时 done 标志为1，GAE 计算时会正确处理

            # 最后一个状态的价值（用于引导后续回报）
            with torch.no_grad():
                _, next_value = self.actorcritic(
                    torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
                next_value = next_value.item()

            # 计算 GAE 优势和 returns
            advantages, returns = self.compute_gae(rewards, dones, values, next_value)

            # 标准化优势（在整个轨迹上标准化，有助于稳定）
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 转换为列表便于索引
            advantages = advantages.cpu().numpy()
            returns = returns.cpu().numpy()

            # PPO 多轮更新
            dataset_size = len(states)
            for _ in range(ppo_epochs):
                indices = np.random.permutation(dataset_size)
                for start in range(0, dataset_size, mini_batch_size):
                    batch_idx = indices[start:start + mini_batch_size]
                    batch_states = [states[i] for i in batch_idx]
                    batch_actions = [actions[i] for i in batch_idx]
                    batch_old_log_probs = [old_log_probs[i] for i in batch_idx]
                    batch_returns = returns[batch_idx].tolist()
                    batch_advantages = advantages[batch_idx].tolist()

                    self.update(batch_states, batch_actions, batch_old_log_probs,
                                batch_returns, batch_advantages)

            print(f"Episode {episode}, Reward: {episode_reward:.2f}")
            rewards_history.append(episode_reward)


        # 保存模型
        torch.save(self.actorcritic.state_dict(), "models/actorcritic_model.pth")

        return rewards_history



if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    agent = Agent(state_dim=4, action_dim=2, lr=3e-4)
    reward = agent.train_ppo(env, num_episodes=50, rollout_steps=500, ppo_epochs=10, mini_batch_size=64)
    import matplotlib.pyplot as plt
    # 训练结束后绘制奖励变化图
    plt.plot(reward)
    plt.xlabel('回合')
    plt.ylabel('总奖励')
    plt.title('每回合奖励')
    plt.grid(True)
    plt.show()
