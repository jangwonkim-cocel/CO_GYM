import copy
import time

import torch

from co_gym.utils.utils import move_to_cpu


class CriticLearner:
    def __init__(self, algorithm, from_data_loader_queue, to_worker_queue, to_monitor_queue, config):
        self.algorithm = copy.deepcopy(algorithm)
        self.critic = self.algorithm.critic.to(config['learner_device'])
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['actor_lr'], eps=config['adam_eps'])
        self.to_m_queue = to_monitor_queue
        self.from_dl_queue = from_data_loader_queue
        self.to_w_queue = to_worker_queue
        self.config = config
        self.epochs = 0

        print(f"Load Critic Learner")

    def learn(self):
        while True:
            self.epochs += 1
            # Load data
            buffer = self.from_dl_queue.get()
            buffer.load_to_device(device=self.config['learner_device'])
            # Update critic
            self.critic.to(self.config['learner_device'])
            self.algorithm.train_critic(buffer, self.optimizer, self.critic)

            # Transfer data
            updated_critic = self.critic.cpu()
            time.sleep(0.01)
            for _ in range(self.config['n_workers']):
                self.to_w_queue.put(updated_critic)

            # Transfer monitoring data
            if self.config['save_model'] and self.epochs % self.config['model_checkpoint_freq'] == 0:
                updated_optimizer = copy.deepcopy(self.optimizer)
                optimizer_state_dict = move_to_cpu(updated_optimizer.state_dict())
                self.to_m_queue.put([updated_critic, optimizer_state_dict])

