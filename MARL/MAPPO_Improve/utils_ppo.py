import torch
import torch.nn as nn
from utils_memory import Memory
from utils_actor_critic import ActorCritic


class MAPPO:
    def __init__(self, state_dim, action_dim, n_agents, actor_lr, critic_lr, gamma, epochs, eps, action_std_init=0.6):
        self.gamma = gamma
        self.epochs = epochs
        self.eps = eps
        # self.memory = []
        # for i in range(n_agents):
        #     self.memory.append(Memory())
        self.memory1 = Memory()
        self.memory2 = Memory()
        self.device = "cpu"
        self.policy = ActorCritic(state_dim, action_dim, n_agents, action_std_init).to(self.device)
        # 优化器
        # self.optimizer = []
        # for i in range(n_agents):
        #     self.optimizer.append(torch.optim.Adam([
        #     {'params': self.policy.actor[i].parameters(), 'lr': actor_lr},
        #     {'params': self.policy.critic.parameters(), 'lr': critic_lr}
        # ]))
        self.optimizer1 = torch.optim.Adam([
            {'params': self.policy.actor1.parameters(), 'lr': actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': critic_lr}
        ])
        self.optimizer2 = torch.optim.Adam([
            {'params': self.policy.actor2.parameters(), 'lr': actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': critic_lr}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, n_agents, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss().to(self.device)
        self.action_std = action_std_init

    # 选择动作
    def select_action(self, state1, state2):
        with torch.no_grad():
            # for i in range(len(states)):
            #     states[i] = torch.FloatTensor(states[i]).to(device).squeeze()
            state1 = torch.FloatTensor(state1).to(self.device)
            state2 = torch.FloatTensor(state2).to(self.device)
            # actions = []
            # action_log_probs = []
            # for i in range(len(states)):
            #     action, action_log_prob = self.policy_old.take_action(states[i], i)
            #     actions.append(action)
            #     action_log_probs.append(action_log_prob)
            action1, action_log_prob1 = self.policy_old.take_action(state1, 1)
            action2, action_log_prob2 = self.policy_old.take_action(state2, 2)
        # state = torch.cat(states)
        state = torch.cat([state1, state2])
        # for i in range(len(states)):
        #     self.memory[i].states.append(state)
        #     self.memory[i].actions.append(actions[i])
        #     self.memory[i].log_probs.append(action_log_probs[i])
        self.memory1.states.append(state)
        self.memory1.actions.append(action1)
        self.memory1.log_probs.append(action_log_prob1)
        self.memory2.states.append(state)
        self.memory2.actions.append(action2)
        self.memory2.log_probs.append(action_log_prob2)
        # for i in range(len(actions)):
        #     actions[i] = actions[i].detach().flatten().to(device)
        # return actions
        return action1.detach().flatten().to(self.device), action2.detach().flatten().to(self.device)

    def learn(self, n_agents):
        # states = []
        # actions = []
        # log_probs = []
        # for i in range(n_agents):
        #     states.append(torch.squeeze(torch.stack(self.memory[i].states, dim=0)).detach().to(device))
        #     actions.append(torch.squeeze(torch.stack(self.memory[i].actions, dim=0)).detach().to(device))
        #     log_probs.append(torch.squeeze(torch.stack(self.memory[i].log_probs, dim=0)).detach().to(device))
        states1 = torch.squeeze(torch.stack(self.memory1.states, dim=0)).detach().to(self.device)
        actions1 = torch.squeeze(torch.stack(self.memory1.actions, dim=0)).detach().to(self.device)
        log_probs1 = torch.squeeze(torch.stack(self.memory1.log_probs, dim=0)).detach().to(self.device)
        states2 = torch.squeeze(torch.stack(self.memory2.states, dim=0)).detach().to(self.device)
        actions2 = torch.squeeze(torch.stack(self.memory2.actions, dim=0)).detach().to(self.device)
        log_probs2 = torch.squeeze(torch.stack(self.memory2.log_probs, dim=0)).detach().to(self.device)

        # rewards_list = []
        # for i in range(n_agents):
        #     rewards = []
        #     reward = 0
        #     for r, done in zip(reversed(self.memory[i].rewards), reversed(self.memory[i].done)):
        #         if done:
        #             reward = 0
        #         reward = r + self.gamma * reward
        #         rewards.insert(0, reward)
        #     rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        #     rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)
        #     rewards_list.append(rewards)

        rewards1 = []
        reward1 = 0
        for reward, done in zip(reversed(self.memory1.rewards), reversed(self.memory1.done)):
            if done:
                reward1 = 0
            reward1 = reward + (self.gamma * reward1)
            rewards1.insert(0, reward1)
        rewards1 = torch.tensor(rewards1, dtype=torch.float32).to(self.device)
        rewards1 = (rewards1 - rewards1.mean()) / (rewards1.std() + 1e-5)

        rewards2 = []
        reward2 = 0
        for reward, done in zip(reversed(self.memory2.rewards), reversed(self.memory2.done)):
            if done:
                reward2 = 0
            reward2 = reward + (self.gamma * reward2)
            rewards2.insert(0, reward2)
        rewards2 = torch.tensor(rewards2, dtype=torch.float32).to(self.device)
        rewards2 = (rewards2 - rewards2.mean()) / (rewards2.std() + 1e-5)

        # 进行参数更新
        # for i in range(self.epochs):
        #     for k in range(n_agents):
        #         log_probs, values, dist_entropy = self.policy.evaluate(states[k], actions[k], k)
        #         values = torch.squeeze(values)
        #         ratios = torch.exp(log_probs - log_probs[k].detach())
        #         advantages = rewards_list[k] - values.detach()
        #         surr1 = ratios * advantages
        #         surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
        #         loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards_list[k]) - 0.01 * dist_entropy
        #         # 利用反向传播和梯度下降进行参数更新
        #         self.optimizer[k].zero_grad()
        #         loss.mean().backward()
        #         self.optimizer[k].step()
        for i in range(self.epochs):
            log_probs, values, dist_entropy = self.policy.evaluate(states1, actions1, 1)
            values = torch.squeeze(values)
            ratios = torch.exp(log_probs - log_probs1.detach())
            advantages = rewards1 - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            loss1 = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards1) - 0.01 * dist_entropy
            # 利用反向传播和梯度下降进行参数更新
            self.optimizer1.zero_grad()
            loss1.mean().backward()
            self.optimizer1.step()

            log_probs, values, dist_entropy = self.policy.evaluate(states2, actions2, 2)
            values = torch.squeeze(values)
            ratios = torch.exp(log_probs - log_probs2.detach())
            advantages = rewards2 - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            loss2 = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards2) - 0.01 * dist_entropy
            # 利用反向传播和梯度下降进行参数更新
            self.optimizer2.zero_grad()
            loss2.mean().backward()
            self.optimizer2.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空经验池
        # for i in range(n_agents):
        #     self.memory[i].clear()
        self.memory1.clear()
        self.memory2.clear()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def change_action_std(self, action_std_change_rate, min_action_std):
        self.action_std = self.action_std - action_std_change_rate
        self.action_std = round(self.action_std, 4)
        self.action_std = max(self.action_std, min_action_std)
        self.set_action_std(self.action_std)