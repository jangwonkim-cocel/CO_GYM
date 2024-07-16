import numpy as np
import torch
from co_gym.algorithms.base import BaseAlgorithm
from co_gym.buffers.on_policy_buffer import OnPolicyBuffer, AuxCriticBuffer, AuxActorBuffer
from co_gym.networks.ppg_networks import PPGPolicy, PPGCritic


class PPG(BaseAlgorithm):
    def __init__(self, env, config):
        self.name = 'PPG'
        self.type = 'on_policy'
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.policy = PPGPolicy(self.obs_dim, self.act_dim, config['policy_hidden_dims'], config['log_std_bound'], config['activation_fc'])
        self.critic = PPGCritic(self.obs_dim, config['critic_hidden_dims'], config['activation_fc'])
        self.worker_buffer = OnPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['max_rollout']))
        self.global_buffer = OnPolicyBuffer(self.obs_dim, self.act_dim, capacity=int(config['max_rollout'] * config['n_workers']))

        self.aux_critic_buffer = AuxCriticBuffer(self.obs_dim, capacity=int(config['max_rollout'] * config['n_workers']
                                                                            * config['aux_update_freq']))
        self.aux_actor_buffer = AuxActorBuffer(self.obs_dim, self.act_dim, capacity=int(config['max_rollout']
                                                                                        * config['n_workers']
                                                                                        * config['aux_update_freq']))
        self.aux_batch_size = int(config['aux_batch_size_coef'] * config['batch_size'])

        self.train_critic_cnt = 0
        self.train_actor_cnt = 0
        self.config = config

        super().__init__()

    def train_critic(self, buffer, optimizer, critic):
        self.train_critic_cnt += 1
        total_iteration = int(buffer.size / self.config['batch_size'])
        for epoch in range(1, self.config['K_epochs'] + 1):
            for i in range(total_iteration):
                optimizer.zero_grad()
                batch_s, _, _, _, _, _, batch_v, batch_ret, _ = buffer.sample(batch_size=self.config['batch_size'])

                # Calculate Critic Loss.
                critic_loss = 0.5 * (critic(batch_s) - batch_ret) ** 2
                critic_loss = critic_loss.mean()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                self.aux_critic_buffer.push_all(batch_s.detach().cpu().numpy(), batch_ret.detach().cpu().numpy())

        # Calculate the value loss "again" in every auxiliary phases
        if self.train_critic_cnt % self.config['aux_update_freq']:
            self.aux_critic_buffer.load_to_device(self.config['learner_device'])
            total_iteration = int(self.aux_critic_buffer.size / self.aux_batch_size)

            for epoch in range(1, self.config['aux_epochs'] + 1):
                for i in range(total_iteration):
                    optimizer.zero_grad()
                    batch_s, batch_ret = self.aux_critic_buffer.sample(batch_size=self.aux_batch_size)

                    critic_loss = 0.5 * (critic(batch_s) - batch_ret) ** 2
                    critic_loss = critic_loss.mean()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), self.config['max_grad_norm'])
                    optimizer.step()

            self.aux_critic_buffer.clear()
            self.aux_critic_buffer.unload()
        return

    def train_actor(self, buffer, optimizer, policy):
        self.train_actor_cnt += 1
        total_iteration = int(buffer.size / self.config['batch_size'])
        for epoch in range(1, self.config['K_epochs'] + 1):
            for i in range(total_iteration):
                optimizer.zero_grad()
                batch_s, batch_a, _, _, _, batch_log_pi, _, batch_ret, batch_gae = buffer.sample(batch_size=self.config['batch_size'])
                # Normalize GAE after sampling.
                batch_gae = (batch_gae - batch_gae.mean()) / (batch_gae.std() + 1e-8)

                # Calculate Actor Loss
                new_log_pi, entropy, means, log_stds = policy.get_prob(batch_s, batch_a)
                ratio = (new_log_pi - batch_log_pi).exp()
                surrogate = ratio * batch_gae
                clipped_surrogate = torch.clamp(ratio, 1 - self.config['policy_clip_eps'],
                                                1 + self.config['policy_clip_eps']) * batch_gae
                policy_loss = -torch.min(surrogate, clipped_surrogate).mean() - self.config['entropy_coef'] * entropy.mean()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), self.config['max_grad_norm'])
                optimizer.step()

                self.aux_actor_buffer.push_all(batch_s.detach().cpu().numpy(), means.detach().cpu().numpy(),
                                                      log_stds.detach().cpu().numpy(), batch_ret.detach().cpu().numpy())

        # Calculate the Joint loss (Auxiliary loss + BC loss) in every auxiliary phases
        if self.train_actor_cnt % self.config['aux_update_freq']:
            self.aux_actor_buffer.load_to_device(self.config['learner_device'])
            total_iteration = int(self.aux_critic_buffer.size / self.aux_batch_size)

            for epoch in range(1, self.config['aux_epochs'] + 1):
                for i in range(total_iteration):
                    states, old_means, old_log_stds, rets = self.aux_actor_buffer.sample(batch_size=self.aux_batch_size)
                    optimizer.zero_grad()
                    # Calculate auxiliary loss
                    aux_loss = (0.5 * (policy.get_aux_value(states) - rets) ** 2).mean()

                    # Calculate BC loss
                    _, _, new_means, new_log_stds = self.policy.sample(states)
                    old_std = old_log_stds.exp()
                    new_std = new_log_stds.exp()
                    old_var = old_std ** 2
                    new_var = new_std ** 2

                    kl = torch.log((new_std/old_std) + 1e-6) + (old_var + (old_means - new_means) ** 2) / (2 * new_var) - 0.5  # Closed form
                    kl_loss = kl.mean()
                    bc_loss = self.config['bc_coef'] * kl_loss
                    joint_loss = aux_loss + bc_loss
                    joint_loss.backward()
                    optimizer.step()

            self.aux_actor_buffer.clear()
            self.aux_actor_buffer.unload()
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

