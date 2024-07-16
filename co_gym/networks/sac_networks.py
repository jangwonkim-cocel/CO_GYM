import torch
import torch.nn as nn
from torch.distributions import Normal
from co_gym.networks.base import BaseCritic, BaseActor
from co_gym.utils.utils import weight_init


class SACCritic(BaseCritic):
    def __init__(self, obs_dim, act_dim, hidden_dims, activation_fc_name):
        super(SACCritic, self).__init__(activation_fc_name)

        self.input_layer_A = nn.Linear(obs_dim + act_dim, hidden_dims[0])
        self.hidden_layers_A = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_A = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_A.append(hidden_layer_A)
        self.output_layer_A = nn.Linear(hidden_dims[-1], 1)

        self.input_layer_B = nn.Linear(obs_dim + act_dim, hidden_dims[0])
        self.hidden_layers_B = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer_B = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers_B.append(hidden_layer_B)
        self.output_layer_B = nn.Linear(hidden_dims[-1], 1)
        self.apply(weight_init)

    @staticmethod
    def to_tensor(state, action):
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

        x_A = self.activation_fc(self.input_layer_A(x))
        for i, hidden_layer_A in enumerate(self.hidden_layers_A):
            x_A = self.activation_fc(hidden_layer_A(x_A))
        x_A = self.output_layer_A(x_A)

        x_B = self.activation_fc(self.input_layer_B(x))
        for i, hidden_layer_B in enumerate(self.hidden_layers_B):
            x_B = self.activation_fc(hidden_layer_B(x_B))
        x_B = self.output_layer_B(x_B)

        return x_A, x_B


class SACPolicy(BaseActor):
    def __init__(self, obs_dim, act_dim, hidden_dims, action_bound, log_std_bound, activation_fc):
        super(SACPolicy, self).__init__(activation_fc)
        self.log_std_min = log_std_bound[0]
        self.log_std_max = log_std_bound[1]

        self.input_layer = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)

        self.mean_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], act_dim)

        self.action_rescale = torch.as_tensor((action_bound[1] - action_bound[0]) / 2., dtype=torch.float32)
        self.action_rescale_bias = torch.as_tensor((action_bound[1] + action_bound[0]) / 2., dtype=torch.float32)

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
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, state, eval, worker_device):
        if worker_device == 'cuda':
            state = torch.FloatTensor(state).unsqueeze(0).to('cuda')
        with torch.no_grad():
            if eval:
                _, _, action = self.sample(state)
            else:
                action, _, _ = self.sample(state)
        return action.cpu().numpy()[0], None

    def sample(self, state):
        mean, log_std = self.forward(state)
        distribution = Normal(mean, log_std.exp())

        unbounded_action = distribution.rsample()
        # [Paper: Appendix C] Enforcing Action Bounds: [a_min, a_max] -> [-1, 1]
        bounded_action = torch.tanh(unbounded_action)
        action = bounded_action * self.action_rescale + self.action_rescale_bias

        # We must recover ture log_prob from true distribution by 'The Change of Variable Formula'.
        log_prob = distribution.log_prob(unbounded_action) - torch.log(self.action_rescale *
                                                                       (1 - bounded_action.pow(2).clamp(0, 1)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        mean = torch.tanh(mean) * self.action_rescale + self.action_rescale_bias

        return action, log_prob, mean

