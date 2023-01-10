from datetime import datetime
import torch
import numpy as np
from utils_ppo import CPPO
from vmas import make_env

reward_list = []


def trainAgent():
    # 环境
    env_name = "wheel"
    n_agents = 2
    device = "cpu"
    env = make_env(
        scenario_name=env_name,
        num_envs=1,
        device=device,
        continuous_actions=True,
        wrapper=None,
        n_agents=n_agents
    )

    # 维度
    state_dim = env.observation_space[0].shape[0] * 2
    action_dim = env.action_space[0].shape[0] * 2
    # 学习率
    actor_lr = 0.00005
    critic_lr = 0.00005
    # 折扣因子
    gamma = 0.99
    epochs = 100
    eps = 0.2
    action_std = 0.5
    # 智能体
    agent = CPPO(state_dim, action_dim, n_agents, actor_lr, critic_lr, gamma, epochs, eps, action_std)

    # 训练总步数
    max_step = 1000000
    # 每个episode的长度
    episode_len = 200
    # 输出频率
    print_freq = 2000
    # 更新频率
    learn_freq = 1000
    std_change_freq = 250000
    step = 0
    episode = 0
    action_std_change_rate = 0.05
    min_action_std = 0.1
    # 用于输出平均奖励
    sum_avg_reward = 0
    sum_episodes = 0
    # 记录时间
    start_time = datetime.now().replace(microsecond=0)
    # 进行训练
    while step < max_step:
        state = env.reset()
        avg_reward = 0
        for i in range(episode_len):
            # states = []
            # for item in state:
            #     states.append(torch.squeeze(item))
            state1 = torch.squeeze(state[0])
            state2 = torch.squeeze(state[1])
            # 选择动作
            # actions = agent.select_action(state)
            # for k in range(len(actions)):
            #     actions[k] = torch.clamp(actions[k], -1.0, 1.0).unsqueeze(dim=0)
            # next_state, reward, done, _ = env.step(actions)
            # rewards = []
            # for k in range(len(reward)):
            #     rewards.append(float(reward[k][0]))
            state = torch.cat((state1, state2), 0)
            action = agent.select_action(state)
            action1 = torch.clamp(action, -1.0, 1.0)[0:2].unsqueeze(dim=0)
            action2 = torch.clamp(action, -1.0, 1.0)[2:4].unsqueeze(dim=0)
            next_state, reward, done, _ = env.step([action1, action2])
            rewards1 = float(reward[0][0])
            rewards2 = float(reward[1][0])
            # 保存到经验池
            # for k in range(len(rewards)):
            #     agent.memory[k].rewards.append(rewards[k])
            #     agent.memory[k].done.append(done[0])
            agent.memory.rewards.append(rewards1)
            agent.memory.done.append(done[0])
            # 累加平均奖励
            # avg_reward += sum(rewards) / len(rewards)
            avg_reward += (rewards1 + rewards2) / 2
            state = next_state
            step += 1

            # 进行学习
            if step % learn_freq == 0:
                agent.learn(n_agents)

            # 输出平均奖励
            if step % print_freq == 0:
                episode_avg_reward = sum_avg_reward / sum_episodes
                episode_avg_reward = round(episode_avg_reward, 2)
                print("Episode: {}\t\tStep: {}/{}\t\tAverage Reward: {}".format(episode + 1, step, max_step, episode_avg_reward))
                reward_list.append(episode_avg_reward)
                sum_avg_reward = 0
                sum_episodes = 0

            if step % std_change_freq == 0:
                agent.change_action_std(action_std_change_rate, min_action_std)

            if done[0]:
                break
        sum_avg_reward += avg_reward
        sum_episodes += 1
        episode += 1

    end_time = datetime.now().replace(microsecond=0)
    print("-------------------------------------------------------------------------------")
    print("cost time: ", end_time - start_time)


if __name__ == '__main__':
    trainAgent()
    reward_save = np.array(reward_list)
    np.save('./logfile.npy', reward_save)
