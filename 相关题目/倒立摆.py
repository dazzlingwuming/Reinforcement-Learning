import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")
state, _ = env.reset()
print("初始状态:", state)
actions = env.action_space #动作空间
print("动作空间:", actions)
action = 0
next_state, reward, truncated,info,done = env.step(action)
print("下一状态:", next_state)
print("奖励:", reward)
print("是否结束:", done)
print("是否超时截断:", truncated)
print("额外信息:", info)

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output


done = False
action_desc ={0:"向左",1:"向右"}
#回报
total_reward = 0.0
#折扣因子
gamma = 0.99
while not done:
    # clear_output(wait=True)
    # plt.imshow(env.render()) #渲染环境
    # plt.show()
    action = np.random.choice([0,1]) #随机选择一个动作
    next_state, reward, terminated,truncated,info = env.step(action)
    total_reward += reward * gamma #计算回报
    print(f"动作: {action_desc[action]}, 下一状态: {next_state}, 奖励: {reward}, 是否结束: {done}")
    if terminated or truncated:
        break
    time.sleep(0.5) #慢一点看清楚环境变化