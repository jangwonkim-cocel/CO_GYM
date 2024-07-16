import copy
import torch
import time

from co_gym.utils.utils import move_to_cpu


class Learner:
    def __init__(self, algorithm, from_worker_queue, to_worker_queue, to_monitor_queue, config):
        self.config = config

        self.algorithm = copy.deepcopy(algorithm)
        self.buffer = self.algorithm.global_buffer

        self.policy = self.algorithm.policy
        self.target_policy = copy.deepcopy(self.policy)
        self.critic = self.algorithm.critic
        self.target_critic = copy.deepcopy(self.critic)

        self.policy = self.policy.to(config['learner_device'])
        self.target_policy = self.target_policy.to(config['learner_device'])
        self.critic = self.critic.to(config['learner_device'])
        self.target_critic = self.target_critic.to(config['learner_device'])

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'], eps=config['adam_eps'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'], eps=config['adam_eps'])

        self.from_w_queue = from_worker_queue
        self.to_w_queue = to_worker_queue
        self.to_m_queue = to_monitor_queue

        self.max_data_size = int(config['max_rollout'] * config['n_workers'])

        self.epochs = 0
        self.total_steps = 0

        print(f"Load Learner")

    def learn(self):
        while True:
            self.epochs += 1
            # Load data
            data_size = 0
            while data_size < self.max_data_size:
                worker_buffer = self.from_w_queue.get()
                data_size += worker_buffer.size
                self.buffer.extend(worker_buffer)
            assert data_size == self.max_data_size
            self.policy.to(self.config['learner_device'])

            # Update actor & critic
            if self.epochs >= self.config['update_after'] / self.max_data_size:
                iteration = self.max_data_size
                self.algorithm.train_both(self.buffer, self.critic_optimizer, self.critic, self.target_critic,
                                          self.policy_optimizer, self.policy, self.target_policy, iteration=iteration)

            # Transfer data
            self.policy.to('cpu')
            updated_policy = copy.deepcopy(self.policy)
            time.sleep(0.005)
            for _ in range(self.config['n_workers']):
                self.to_w_queue.put(updated_policy)

            # Transfer monitoring data
            if self.epochs % self.config['eval_freq'] == 0:
                updated_critic = copy.deepcopy(self.critic.to('cpu'))
                updated_critic_optimizer = copy.deepcopy(self.critic_optimizer)
                critic_optimizer_state_dict = move_to_cpu(updated_critic_optimizer.state_dict())

                updated_policy_optimizer = copy.deepcopy(self.policy_optimizer)
                policy_optimizer_state_dict = move_to_cpu(updated_policy_optimizer.state_dict())

                if self.config['save_model'] and self.epochs % self.config['model_checkpoint_freq'] == 0:
                    if self.algorithm.alpha_optimizer is not None and self.algorithm.log_alpha is not None:
                        updated_log_alpha = self.algorithm.log_alpha.item()
                        updated_alpha_optimizer = copy.deepcopy(self.algorithm.alpha_optimizer)
                        updated_alpha_optimizer = move_to_cpu(updated_alpha_optimizer.state_dict())
                    else:
                        updated_log_alpha = None
                        updated_alpha_optimizer = None

                    self.to_m_queue.put([updated_policy, updated_critic, policy_optimizer_state_dict,
                                         critic_optimizer_state_dict, updated_log_alpha, updated_alpha_optimizer,
                                         self.buffer])
                else:
                    self.to_m_queue.put([updated_policy, None, None, None, None, None, None])

                self.critic.to(self.config['learner_device'])




