import numpy as np
import torch
import torch.nn as nn
from co_gym.networks.base import BaseCritic, BaseActor
from co_gym.utils.utils import weight_init


class DDPGCritic(BaseCritic):
    def __init__(self, obs_dim, act_dim, hidden_dims, activation_fc_name):
        super(DDPGCritic, self).__init__(activation_fc_name)

        self.input_layer = nn.Linear(obs_dim + act_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.apply(weight_init)

    def to_tensor(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action):
        x, u = self.to_tensor(state, action)
        x = torch.cat([x, u], dim=1)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)

        return x


class DDPGPolicy(BaseActor):
    def __init__(self, obs_dim, act_dim, action_bound, noise_scale, hidden_dims, activation_fc_name):
        super(DDPGPolicy, self).__init__(activation_fc_name)
        self.act_dim = act_dim
        self.action_bound = action_bound
        self.noise_scale = noise_scale

        self.out_activation_fc = torch.tanh
        self.input_layer = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], act_dim)

        self.action_rescale = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)
        self.apply(weight_init)

    def to_tensor(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)

        return x

    def get_action(self, state, eval, worker_device):
        if worker_device == 'cuda':
            state = torch.FloatTensor(state).unsqueeze(0).to('cuda')
        with torch.no_grad():
            action = self.forward(state)
            if eval:
                return action.cpu().numpy()[0], None
            else:
                noise = np.random.normal(loc=0, scale=abs(self.action_bound[1]) * self.noise_scale, size=self.act_dim)
                action = self.forward(state).cpu().numpy()[0] + noise
                action = np.clip(action, self.action_bound[0], self.action_bound[1])
                return action, None

    def forward(self, state):
        x = self.to_tensor(state)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        x = self.out_activation_fc(x)
        x = x * self.action_rescale + self.action_rescale_bias
        return x
