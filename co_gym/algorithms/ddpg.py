import torch
from co_gym.algorithms.base import BaseAlgorithm
from co_gym.buffers.off_policy_buffer import OffPolicyBuffer
from co_gym.networks.ddpg_networks import DDPGPolicy, DDPGCritic
from co_gym.utils.utils import soft_update


class DDPG(BaseAlgorithm):
    def __init__(self, env, config):
        self.name = 'DDPG'
        self.type = 'off_policy'
        self.config = config
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.policy = DDPGPolicy(self.obs_dim, self.act_dim, env.action_bound, config['noise_scale'],
                                 config['policy_hidden_dims'], config['activation_fc'])

        self.critic = DDPGCritic(self.obs_dim, self.act_dim, config['critic_hidden_dims'], config['activation_fc'])

        self.worker_buffer = OffPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['max_rollout']))
        self.global_buffer = OffPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['offline_buffer_capacity']))

        self.tau = config['tau']

        super().__init__()

    def train_critic(self, *inputs):
        pass

    def train_actor(self, *inputs):
        pass

    def train_both(self, buffer, critic_optimizer, critic, target_critic, policy_optimizer, policy, target_policy, iteration):
        for _ in range(iteration):
            states, actions, rewards, next_states, dones = buffer.sample(self.config['batch_size'], device=self.config['learner_device'])
            # Calculate the Critic Loss
            with torch.no_grad():
                target_q_values = rewards + (1 - dones) * self.config['gamma']\
                                  * target_critic(next_states, target_policy(next_states))
            q_values = critic(states, actions)
            critic_loss = ((q_values - target_q_values) ** 2).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Calculate the Actor Loss
            policy_loss = -critic(states, policy(states)).mean()
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            soft_update(policy, target_policy, self.tau)
            soft_update(critic, target_critic, self.tau)

    def cal_target(self, buffer, critic):
        raise NotImplementedError
