import copy
import time

import numpy as np
from co_gym.buffers.on_policy_buffer import NormBuffer


class Worker:
    def __init__(self, algorithm, env, rank, to_data_loader_queue, from_critic_queue, from_actor_queue,
                 from_wrapper_manager_queue, to_wrapper_manager_queue, config):
        time.sleep(0.1)
        self.epochs = 0
        self.algorithm = copy.deepcopy(algorithm)
        self.env = copy.deepcopy(env)
        self.env.seed(rank)
        self.rank = rank
        self.config = config

        self.policy = self.algorithm.policy
        self.policy.to(config['worker_device'])
        self.critic = self.algorithm.critic
        self.buffer = self.algorithm.worker_buffer

        self.to_dl_queue = to_data_loader_queue
        self.from_c_queue = from_critic_queue
        self.from_a_queue = from_actor_queue
        self.from_wrapper_manager_queue = from_wrapper_manager_queue
        self.to_wrapper_manager_queue = to_wrapper_manager_queue

        self.is_env_normalized = self.config['normalize_obs'] or self.config['normalize_reward']

        self.rolling_ret = np.zeros(())
        if self.is_env_normalized:
            self.norm_buffer = NormBuffer(env.obs_dim, capacity=int(config['max_rollout']))
        print(f"Load Worker (id: {rank})")

    def cal_target(self, buffer, critic):
        preprocessed_buffer = self.algorithm.cal_target(buffer, critic)
        return preprocessed_buffer

    def work(self):
        step = 0
        while True:
            epi_return = 0
            terminated, truncated = False, False
            state, _ = self.env.reset()
            while not (terminated or truncated):
                action, log_prob = self.policy.get_action(state, eval=False, worker_device=self.config['worker_device'])
                next_states, reward, terminated, truncated, info = self.env.step(action)

                true_done = 1.0 if terminated else 0.0

                if self.is_env_normalized:
                    self.rolling_ret = self.rolling_ret * self.config['gamma'] + info['true_reward']
                    self.norm_buffer.push(info['true_next_obs'], self.rolling_ret)

                # assert self.algorithm.type == 'on_policy'
                self.buffer.push(state, action, reward, next_states, true_done, log_prob, None, None, None)

                epi_return += reward
                step += 1
                state = next_states

                if terminated or truncated:
                    worker_buffer = self.cal_target(self.buffer, self.critic)
                    time.sleep(0.005)
                    self.to_dl_queue.put(worker_buffer)
                    self.rolling_ret = np.zeros(())
                    self.buffer.clear()

                    if step == self.config['max_rollout']:
                        self.policy = self.from_a_queue.get()
                        if self.config['worker_device'] == 'cuda':
                            self.policy.to('cuda')
                        self.critic = self.from_c_queue.get()

                        if self.is_env_normalized:
                            self.to_wrapper_manager_queue.put(self.norm_buffer)
                            norm_values = self.from_wrapper_manager_queue.get()
                            if self.config['normalize_obs']:
                                obs_mean, obs_var = norm_values[0], norm_values[1]
                                self.env.set_obs_mean_var(obs_mean, obs_var)
                            if self.config['normalize_reward']:
                                ret_var = norm_values[2]
                                self.env.set_ret_var(ret_var)
                            self.norm_buffer.clear()

                        step = 0
                        self.epochs += 1
                        if self.epochs == self.config['max_epochs']:
                            return
                    break

                if step == self.config['max_rollout']:
                    worker_buffer = self.cal_target(self.buffer, self.critic)
                    time.sleep(0.005)
                    self.to_dl_queue.put(worker_buffer)
                    self.buffer.clear()
                    self.policy = self.from_a_queue.get()
                    if self.config['worker_device'] == 'cuda':
                        self.policy.to('cuda')
                    self.critic = self.from_c_queue.get()

                    if self.is_env_normalized:
                        self.to_wrapper_manager_queue.put(self.norm_buffer)
                        norm_values = self.from_wrapper_manager_queue.get()
                        if self.config['normalize_obs']:
                            obs_mean, obs_var = norm_values[0], norm_values[1]
                            self.env.set_obs_mean_var(obs_mean, obs_var)
                        if self.config['normalize_reward']:
                            ret_var = norm_values[2]
                            self.env.set_ret_var(ret_var)
                        self.norm_buffer.clear()

                    step = 0
                    self.epochs += 1
                    if self.epochs == self.config['max_epochs']:
                        return

