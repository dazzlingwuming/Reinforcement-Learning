import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 创建 CliffWalking 环境
env = gym.make('CliffWalking-v1')
n_states = env.observation_space.n  # 48
n_actions = env.action_space.n      # 4
print(f"状态数: {n_states}, 动作数: {n_actions}")

# 超参数
alpha = 0.1        # 学习率
gamma = 0.99       # 折扣因子
epsilon = 0.1      # 探索率
episodes = 500     # 训练轮数（CliffWalking 较简单，500轮足够）
max_steps = 100    # 每轮最大步数

# 加载Q表
try:
    Q = np.load('save/q_table_sarsa_T2.npy')
    print("已加载已有 Q 表。")
except FileNotFoundError:
    Q = np.zeros((n_states, n_actions))
    print("未找到 Q 表，从零开始。")

# 用于记录每轮的总奖励
rewards = []
#epsilon-greedy 策略函数
def choose_action(state, Q, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state, :])

for episode in range(episodes):
    state, _ = env.reset()
    action = choose_action(state, Q, epsilon)   # 选择第一个动作
    total_reward = 0
    done = False

    for step in range(max_steps):
        # 执行动作，观察结果
        next_state, reward, done, truncated, info = env.step(action)

        # 根据当前策略选择下一个动作（用于更新）
        next_action = choose_action(next_state, Q, epsilon)

        # SARSA 更新
        td_target = reward + gamma * Q[next_state, next_action] * (1 - done)
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        total_reward += reward
        state = next_state
        action = next_action   # 实际执行的动作

        if done:
            break

    rewards.append(total_reward)

    # 每100轮输出一次平均奖励
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards[-100:])
        print(f"Episode {episode+1}, 最近100轮平均奖励: {avg_reward:.2f}")
# 保存 Q 表
np.save('save/q_table_sarsa_T2.npy', Q)

# 计算移动平均
def moving_average(data, window=10):
    return np.convolve(data, np.ones(window), 'valid') / window

ma_rewards = moving_average(rewards, window=10)
plt.plot(ma_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward (last 10)')
plt.title('SARSA on CliffWalking')
plt.grid(True)
plt.show()

#训练结束后，用贪婪策略（ε=0）测试智能体的表现。
test_episodes = 100
test_rewards = []

for _ in range(test_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    step = 0
    while not done and step < max_steps:
        action = np.argmax(Q[state, :])   # 贪婪选择
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1
    test_rewards.append(total_reward)

print(f"测试 {test_episodes} 轮的平均奖励: {np.mean(test_rewards):.2f}")






