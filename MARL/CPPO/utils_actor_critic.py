import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_agents, action_std_init):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = "cpu"
        self.action_std = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        # actor
        # self.actor = []
        # for i in range(n_agents):
        #     self.actor.append(nn.Sequential(
        #         nn.Linear(state_dim, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, 64),
        #         nn.Tanh(),
        #         nn.Linear(64, action_dim),
        #     ))
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    # 选择动作
    def take_action(self, state, id):
        actions_mean = self.actor(state)
        cov = torch.diag(self.action_std).unsqueeze(dim=0)
        dist = torch.distributions.MultivariateNormal(actions_mean, cov)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.detach(), action_log_prob.detach()

    # 评价函数
    def evaluate(self, state, action, id):
        actions_mean = self.actor(state)
        action_std = self.action_std.expand_as(actions_mean)
        cov = torch.diag_embed(action_std).to(self.device)
        dist = torch.distributions.MultivariateNormal(actions_mean, cov)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(state)
        return action_log_probs, values, dist_entropy

    def set_action_std(self, new_action_std):
        self.action_std = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
