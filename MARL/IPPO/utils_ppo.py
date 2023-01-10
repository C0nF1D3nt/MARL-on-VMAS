import torch
import torch.nn as nn
from utils_memory import Memory
from utils_actor_critic import ActorCritic


class IPPO:
    def __init__(self, state_dim, action_dim, n_agents, actor_lr, critic_lr, gamma, epochs, eps, action_std_init=0.6):
        self.gamma = gamma
        self.epochs = epochs
        self.eps = eps
        # self.memory = []
        # for i in range(n_agents):
        #     self.memory.append(Memory())
        self.memory = Memory()
        self.device = "cpu"
        self.policy = ActorCritic(state_dim, action_dim, n_agents, action_std_init).to(self.device)
        # 优化器
        # self.optimizer = []
        # for i in range(n_agents):
        #     self.optimizer.append(torch.optim.Adam([
        #     {'params': self.policy.actor[i].parameters(), 'lr': actor_lr},
        #     {'params': self.policy.critic.parameters(), 'lr': critic_lr}
        # ]))
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': critic_lr}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, n_agents, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss().to(self.device)
        self.action_std = action_std_init

    # 选择动作
    def select_action(self, state):
        with torch.no_grad():
            # for i in range(len(states)):
            #     states[i] = torch.FloatTensor(states[i]).to(device).squeeze()
            state = torch.FloatTensor(state).to(self.device)
            # actions = []
            # action_log_probs = []
            # for i in range(len(states)):
            #     action, action_log_prob = self.policy_old.take_action(states[i], i)
            #     actions.append(action)
            #     action_log_probs.append(action_log_prob)
            action, action_log_prob = self.policy_old.take_action(state, 1)
        # state = torch.cat(states)
        # for i in range(len(states)):
        #     self.memory[i].states.append(state)
        #     self.memory[i].actions.append(actions[i])
        #     self.memory[i].log_probs.append(action_log_probs[i])
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.log_probs.append(action_log_prob)
        # for i in range(len(actions)):
        #     actions[i] = actions[i].detach().flatten().to(device)
        # return actions
        return action.detach().flatten().to(self.device)

    def learn(self, n_agents):
        # states = []
        # actions = []
        # log_probs = []
        # for i in range(n_agents):
        #     states.append(torch.squeeze(torch.stack(self.memory[i].states, dim=0)).detach().to(device))
        #     actions.append(torch.squeeze(torch.stack(self.memory[i].actions, dim=0)).detach().to(device))
        #     log_probs.append(torch.squeeze(torch.stack(self.memory[i].log_probs, dim=0)).detach().to(device))
        states = torch.squeeze(torch.stack(self.memory.states, dim=0)).detach().to(self.device)
        actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(self.device)
        mem_log_probs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(self.device)

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

        rewards = []
        reward = 0
        for r, done in zip(reversed(self.memory.rewards), reversed(self.memory.done)):
            if done:
                reward = 0
            reward = r + (self.gamma * reward)
            rewards.insert(0, reward)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

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
            log_probs, values, dist_entropy = self.policy.evaluate(states, actions, 1)
            values = torch.squeeze(values)
            ratios = torch.exp(log_probs - mem_log_probs.detach())
            advantages = rewards - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps, 1 + self.eps) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(values, rewards) - 0.01 * dist_entropy
            # 利用反向传播和梯度下降进行参数更新
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空经验池
        # for i in range(n_agents):
        #     self.memory[i].clear()
        self.memory.clear()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def change_action_std(self, action_std_change_rate, min_action_std):
        self.action_std = self.action_std - action_std_change_rate
        self.action_std = round(self.action_std, 4)
        self.action_std = max(self.action_std, min_action_std)
        self.set_action_std(self.action_std)