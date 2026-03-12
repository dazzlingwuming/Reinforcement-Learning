import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 创建环境，is_slippery=False 可以关闭滑冰特性（变为确定性），这里设为 True 保持默认随机性
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='rgb_array')

# 重置环境，得到初始状态
state = env.reset()
print("初始状态:", state)  # 新版reset返回 (state, info)
n_states = env.observation_space.n  # 16
n_actions = env.action_space.n      # 4
print(f"状态数: {n_states}, 动作数: {n_actions}")

# 超参数
alpha = 0.1        # 学习率
gamma = 0.99       # 折扣因子
epsilon = 0.1      # 探索率
episodes = 50000    # 训练轮数
max_steps = 1000    # 每轮最大步数，防止无限循环

#加载Q表
try:
    Q = np.load('q_table.npy')
    start_episode = 0  # 如果存在最终的次数，可以从那里继续训练
    print("已加载已有 Q 表。")
except FileNotFoundError:
    Q = np.zeros((n_states, n_actions))
    start_episode = 0
    print("未找到 Q 表，从零开始。")

# 在训练中，需要根据当前 Q 表选择动作，同时以 epsilon 的概率随机探索。
def choose_action(state, Q, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()  # 随机探索
    else:
        action = np.argmax(Q[state, :])     # 利用当前最优
    return action

# 用于记录每轮的奖励
rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    # 更新epsilon，最开始探索多，后面逐渐减少探索
    epsilon = max(0.1, epsilon * 0.995)  # 每轮衰减 epsilon，但不低于 0.01

    for step in range(max_steps):
        # 选择动作
        action = choose_action(state, Q, epsilon)

        # 执行动作，观察结果
        next_state, reward, done, truncated, info = env.step(action)  # 返回 5 个值 分别是：下一个状态、奖励、是否结束、是否超时截断、额外信息
        # 对于 FrozenLake，truncated 通常为 False（没有超时截断），这里用 done 判断是否终止

        # 计算 TD 目标
        td_target = reward + gamma * np.max(Q[next_state, :]) * (1 - done)  # 如果 done，未来 Q 为 0

        # 计算 TD 误差并更新 Q 表
        td_error = td_target - Q[state, action]
        Q[state, action] += alpha * td_error

        # 累加奖励（这里奖励只有 0 或 1）
        total_reward += reward

        # 更新状态
        state = next_state

        # 如果 episode 结束，退出循环
        if done:
            break

    rewards.append(total_reward)

    # 每 1000 轮打印一次平均奖励
    if (episode + 1) % 1000 == 0:
        avg_reward = np.mean(rewards[-1000:])
        print(f"Episode {episode + 1}, 最近1000轮平均奖励: {avg_reward:.4f}")

np.save('q_table.npy', Q)
# 计算移动平均
def moving_average(data, window=100):
    return np.convolve(data, np.ones(window), 'valid') / window

ma_rewards = moving_average(rewards, window=100)

# 测试
test_episodes = 100
successes = 0
for _ in range(test_episodes):
    state, _ = env.reset()
    done = False
    step = 0
    while not done and step < max_steps:
        action = np.argmax(Q[state])
        state, reward, done, truncated, info = env.step(action)
        step += 1
    if reward == 1:  # 到达目标
        successes += 1

print(f"测试 {test_episodes} 轮，成功率: {successes/test_episodes:.2%}")

# 绘制学习曲线
ma_rewards = np.convolve(rewards, np.ones(100)/100, mode='valid')
plt.plot(ma_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward (last 100)')
plt.title('Q-learning on FrozenLake')
plt.show()

