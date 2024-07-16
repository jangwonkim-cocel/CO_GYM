import numpy as np
import torch
from co_gym.algorithms.base import BaseAlgorithm
from co_gym.buffers.on_policy_buffer import OnPolicyBuffer
from co_gym.networks.ppo_networks import PPOPolicy, PPOCritic
import torch as th


class PPO(BaseAlgorithm):
    def __init__(self, env, config):
        th.autograd.set_detect_anomaly(True)
        self.name = 'PPO'
        self.type = 'on_policy'
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.policy = PPOPolicy(self.obs_dim, self.act_dim, config['policy_hidden_dims'], config['log_std_bound'], config['activation_fc'])
        self.critic = PPOCritic(self.obs_dim, config['critic_hidden_dims'], config['activation_fc'])
        self.worker_buffer = OnPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['max_rollout']))
        self.global_buffer = OnPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['max_rollout'] * config['n_workers']))
        self.config = config
        super().__init__()

    def train_critic(self, buffer, optimizer, critic):
        total_iteration = int(buffer.size / self.config['batch_size'])
        for epoch in range(1, self.config['K_epochs'] + 1):
            for i in range(total_iteration):
                optimizer.zero_grad()
                batch_s, _, _, _, _, _, batch_v, batch_ret, _ = buffer.sample(batch_size=self.config['batch_size'])

                # Calculate Critic Loss.
                critic_loss = 0.5 * (critic(batch_s) - batch_ret) ** 2
                if self.config['clip_value'] is True:  # Value function is also clipped.
                    clipped_v = batch_v + (critic(batch_s) - batch_v).clamp(-self.config['value_clip_eps'],
                                                                            self.config['value_clip_eps'])
                    clipped_critic_loss = (clipped_v - batch_ret) ** 2
                    critic_loss = torch.max(critic_loss, clipped_critic_loss)
                critic_loss = critic_loss.mean()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), self.config['max_grad_norm'])
                optimizer.step()
        return

    def train_actor(self, buffer, optimizer, policy):
        total_iteration = int(buffer.size / self.config['batch_size'])
        for epoch in range(1, self.config['K_epochs'] + 1):
            for i in range(total_iteration):
                optimizer.zero_grad()
                batch_s, batch_a, _, _, _, batch_log_pi, _, _, batch_gae = buffer.sample(batch_size=self.config['batch_size'])
                # Normalize GAE after sampling.
                batch_gae = (batch_gae - batch_gae.mean()) / (batch_gae.std() + 1e-6)

                # Calculate Actor Loss
                new_log_pi, entropy = policy.get_prob(batch_s, batch_a)
                ratio = (new_log_pi - batch_log_pi).clamp_(max=30).exp()
                surrogate = ratio * batch_gae
                clipped_surrogate = torch.clamp(ratio, 1 - self.config['policy_clip_eps'],
                                                1 + self.config['policy_clip_eps']) * batch_gae
                policy_loss = -torch.min(surrogate, clipped_surrogate).mean() - self.config['entropy_coef'] * entropy.mean()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.config['max_grad_norm'])
                optimizer.step()
        return

    def train_both(self, *inputs):
        pass

    def cal_target(self, buffer, critic):
        critic.to(self.config['learner_device'])
        preprocessed_buffer = OnPolicyBuffer(self.obs_dim, self.act_dim, capacity=self.config['max_rollout'])
        # Calculate GAE
        states, actions, rewards, next_states, dones, log_pis, _, _, _ = buffer.get_data()
        states = torch.FloatTensor(states).to(self.config['learner_device'])
        next_states = torch.FloatTensor(next_states).to(self.config['learner_device'])
        dones = torch.FloatTensor(dones).to(self.config['learner_device'])

        with torch.no_grad():
            values = critic(states).cpu().numpy()
            last_value = (critic(next_states[-1]) * (1 - dones[-1])).cpu().numpy()

        states = states.cpu().numpy()
        next_states = next_states.cpu().numpy()
        dones = dones.cpu().numpy()

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        previous_value = last_value
        running_return = last_value
        running_advantage = np.zeros(1)

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config['gamma'] * running_return
            running_td_error = rewards[t] + self.config['gamma'] * previous_value - values[t]
            running_advantage = running_td_error + self.config['gamma'] * self.config['gae_lambda'] * running_advantage

            returns[t] = running_return
            previous_value = values[t]
            advantages[t] = running_advantage

        preprocessed_buffer.push_all(states, actions, rewards, next_states, dones, log_pis, values, returns, advantages)
        critic.to('cpu')
        return preprocessed_buffer
