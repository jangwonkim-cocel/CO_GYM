import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from co_gym.networks.base import BaseCritic, BaseActor
from co_gym.utils.utils import weight_init


class PPOCritic(BaseCritic):
    def __init__(self, obs_dim, hidden_dims, activation_fc_name):
        super(PPOCritic, self).__init__(activation_fc_name)

        self.input_layer = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.apply(weight_init)

    @staticmethod
    def to_tensor(state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state, action=None):
        x = self.to_tensor(state)
        x = self.to_tensor(x)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        value = self.output_layer(x)
        return value


class PPOPolicy(BaseActor):
    def __init__(self, obs_dim, act_dim, hidden_dims, log_std_bound, activation_fc_name):
        super(PPOPolicy, self).__init__(activation_fc_name)
        self.log_std_min = log_std_bound[0]
        self.log_std_max = log_std_bound[1]
        self.input_layer = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)

        self.mean_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(1, act_dim))
        # log-std is state-independent

        self.apply(weight_init)

    @staticmethod
    def to_tensor(state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self.to_tensor(state)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, state, eval, worker_device):
        if worker_device == 'cuda':
            state = torch.FloatTensor(state).to('cuda')
        with torch.no_grad():
            if eval:
                _, log_prob, action = self.sample(state)
            else:
                action, log_prob, _ = self.sample(state)
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]

    def sample(self, state):
        mean, log_std = self.forward(state)
        distribution = Normal(mean, torch.exp(log_std))
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob, mean

    def get_prob(self, state, action):
        mean, log_std = self.forward(state)
        distribution = Normal(mean, torch.exp(log_std))
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = distribution.entropy()

        return log_prob, entropy


