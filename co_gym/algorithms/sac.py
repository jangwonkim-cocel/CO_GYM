import torch
from co_gym.algorithms.base import BaseAlgorithm
from co_gym.buffers.off_policy_buffer import OffPolicyBuffer
from co_gym.networks.sac_networks import SACPolicy, SACCritic
from co_gym.utils.utils import soft_update


class SAC(BaseAlgorithm):
    def __init__(self, env, config):
        self.name = 'SAC'
        self.type = 'off_policy'
        self.config = config
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.policy = SACPolicy(self.obs_dim, self.act_dim, config['policy_hidden_dims'], env.action_bound,
                                config['log_std_bound'], config['activation_fc'])
        self.critic = SACCritic(self.obs_dim, self.act_dim, config['critic_hidden_dims'], config['activation_fc'])

        self.worker_buffer = OffPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['max_rollout']))
        self.global_buffer = OffPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['offline_buffer_capacity']))

        self.target_entropy = -torch.prod(torch.Tensor((self.act_dim, ))).to(config['learner_device'])
        self.log_alpha = torch.zeros(1, requires_grad=True, device=config['learner_device'])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['temperature_lr'])
        self.tau = config['tau']

        super().__init__()

    def train_critic(self, *inputs):
        raise NotImplementedError

    def train_actor(self, *inputs):
        raise NotImplementedError

    def train_both(self, buffer, critic_optimizer, critic, target_critic, policy_optimizer, policy, target_policy, iteration):
        for _ in range(iteration):
            states, actions, rewards, next_states, dones = buffer.sample(self.config['batch_size'], device=self.config['learner_device'])

            # Train the Critic Loss
            critic_optimizer.zero_grad()
            with torch.no_grad():
                next_actions, next_log_pis, _ = policy.sample(next_states)
                next_q_values_A, next_q_values_B = target_critic(next_states, next_actions)
                next_q_values = torch.min(next_q_values_A, next_q_values_B) - self.log_alpha.exp() * next_log_pis
                target_q_values = rewards + (1 - dones) * self.config['gamma'] * next_q_values

            q_values_A, q_values_B = critic(states, actions)
            critic_loss = ((q_values_A - target_q_values) ** 2).mean() + ((q_values_B - target_q_values) ** 2).mean()

            critic_loss.backward()
            critic_optimizer.step()

            # Train the Actor Loss
            policy_optimizer.zero_grad()
            actions, log_pis, _ = policy.sample(states)
            q_values_A, q_values_B = critic(states, actions)
            q_values = torch.min(q_values_A, q_values_B)

            policy_loss = (self.log_alpha.exp().detach() * log_pis - q_values).mean()
            policy_loss.backward()
            policy_optimizer.step()

            # Train the Entropy Temperature
            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Target-Critic (Soft) Update
            soft_update(critic, target_critic, self.tau)

    def cal_target(self, buffer, critic):
        raise NotImplementedError
