import torch
from co_gym.algorithms.base import BaseAlgorithm
from co_gym.buffers.off_policy_buffer import OffPolicyBuffer
from co_gym.networks.tqc_networks import TQCPolicy, TQCCritic
from co_gym.utils.utils import soft_update, quantile_huber_loss_f


class TQC(BaseAlgorithm):
    def __init__(self, env, config):
        self.name = 'TQC'
        self.type = 'off_policy'
        self.config = config
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.policy = TQCPolicy(self.obs_dim, self.act_dim, config['policy_hidden_dims'], env.action_bound, config['log_std_bound'],
                                config['activation_fc'])
        self.critic = TQCCritic(self.obs_dim, self.act_dim, config['critic_hidden_dims'], config['n_critics'], config['n_quantiles'],
                                config['activation_fc'])

        self.worker_buffer = OffPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['max_rollout']))
        self.global_buffer = OffPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['offline_buffer_capacity']))

        self.target_entropy = -torch.prod(torch.Tensor((self.act_dim, ))).to(config['learner_device'])
        self.log_alpha = torch.zeros(1, requires_grad=True, device=config['learner_device'])
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config['temperature_lr'])

        self.quantiles_total = config['n_critics'] * config['n_quantiles']
        self.top_quantiles_to_drop = int(config['n_critics'] * config['n_drop_atoms'])
        self.tau = config['tau']

        super().__init__()

    def train_critic(self, *inputs):
        pass

    def train_actor(self, *inputs):
        pass

    def train_both(self, buffer, critic_optimizer, critic, target_critic, policy_optimizer, policy, target_policy, iteration):
        for _ in range(iteration):
            states, actions, rewards, next_states, dones = buffer.sample(self.config['batch_size'], device=self.config['learner_device'])
            # Train the Critic Loss
            critic_optimizer.zero_grad()
            with torch.no_grad():
                next_actions, next_log_pis, _ = policy.sample(next_states)
                next_z = target_critic(next_states, next_actions)  # batch x nets x quantiles
                sorted_z, _ = torch.sort(next_z.reshape(self.config['batch_size'], -1))
                sorted_z_part = sorted_z[:, :self.quantiles_total - self.top_quantiles_to_drop]

                target_z = rewards + (1 - dones) * self.config['gamma'] * (sorted_z_part - self.log_alpha.exp() * next_log_pis)

            cur_z = critic(states, actions)
            critic_loss = quantile_huber_loss_f(cur_z, target_z, self.config['learner_device'])

            critic_loss.backward()
            critic_optimizer.step()

            # Train the Actor Loss
            policy_optimizer.zero_grad()
            actions, log_pis, _ = policy.sample(states)
            z = critic(states, actions)
            actor_loss = (self.log_alpha.exp().detach() * log_pis - z.mean(2).mean(1, keepdim=True)).mean()
            actor_loss.backward()
            policy_optimizer.step()

            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha.exp() * (log_pis + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Target-Critic (Soft) Update
            soft_update(critic, target_critic, self.tau)

    def cal_target(self, buffer, critic):
        raise NotImplementedError
