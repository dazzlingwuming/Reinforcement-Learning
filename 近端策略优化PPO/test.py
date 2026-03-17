import gymnasium as gym
import torch
import torch.optim as optim

from 近端策略优化PPO.近端策略优化_倒立摆 import Agent


#看测试效果
def test_render(agent, env, episodes=5):
    for episode in range(episodes):
        state,_ = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render()
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated,truncated,info = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"回合 {episode + 1}: 总奖励 = {total_reward}")
    env.close()


# 加载模型后测试
env = gym.make('CartPole-v1', render_mode='human')
agent = Agent(state_dim=4, action_dim=2, lr=3e-4)
agent.actorcritic.load_state_dict(torch.load('models/actorcritic_model.pth'))
agent.actorcritic.eval()
test_render(agent, env)