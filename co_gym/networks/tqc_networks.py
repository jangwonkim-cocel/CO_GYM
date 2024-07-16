import torch
import torch.nn as nn
from torch.distributions import Normal
from co_gym.networks.base import BaseCritic, BaseActor
from co_gym.utils.utils import weight_init


class QuantileCritic(BaseCritic):
    def __init__(self, obs_dim, act_dim, hidden_dims, n_quantiles, activation_fc_name):
        super(QuantileCritic, self).__init__(activation_fc_name)

        self.input_layer = nn.Linear(obs_dim + act_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], n_quantiles)
        self.apply(weight_init)

    @staticmethod
    def to_tensor(state, action):
        pass

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        quantiles = self.output_layer(x)
        return quantiles


class TQCCritic(BaseCritic):
    def __init__(self, state_dim, action_dim, hidden_dims, n_critics, n_quantiles, activation_fc_name):
        super(TQCCritic, self).__init__(activation_fc_name)
        self.critics = []
        for i in range(n_critics):
            one_critic = QuantileCritic(state_dim, action_dim, hidden_dims, n_quantiles, activation_fc_name)
            self.add_module(f'qf{i}', one_critic)
            self.critics.append(one_critic)

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
        quantiles = torch.stack(tuple(one_critic(state, action) for one_critic in self.critics), dim=1)
        return quantiles


class TQCPolicy(BaseActor):
    def __init__(self, obs_dim, act_dim, hidden_dims, action_bound, log_std_bound, activation_fc_name):
        super(TQCPolicy, self).__init__(activation_fc_name)
        self.log_std_min = log_std_bound[0]
        self.log_std_max = log_std_bound[1]

        self.input_layer = nn.Linear(obs_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
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

