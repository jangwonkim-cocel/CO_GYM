import copy
import torch
import time

from co_gym.utils.utils import move_to_cpu


class ActorLearner:
    def __init__(self, algorithm, from_data_loader_queue, to_worker_queue, to_monitor_queue, config):
        self.algorithm = copy.deepcopy(algorithm)
        self.policy = self.algorithm.policy.to(config['learner_device'])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config['actor_lr'], eps=config['adam_eps'])
        self.to_m_queue = to_monitor_queue
        self.from_dl_queue = from_data_loader_queue
        self.to_w_queue = to_worker_queue
        self.config = config

        self.epochs = 0
        print(f"Load Actor Learner")

    def learn(self):
        while True:
            self.epochs += 1
            # Load data
            buffer = self.from_dl_queue.get()
            buffer.load_to_device(device=self.config['learner_device'])

            # Update policy
            self.policy.to(self.config['learner_device'])
            self.algorithm.train_actor(buffer, self.optimizer, self.policy)

            # Transfer data
            updated_policy = self.policy.cpu()
            time.sleep(0.005)
            for _ in range(self.config['n_workers']):
                self.to_w_queue.put(updated_policy)

            # Transfer monitoring data
            if self.epochs % self.config['eval_freq'] == 0 and self.config['eval']:
                updated_optimizer = copy.deepcopy(self.optimizer)
                optimizer_state_dict = move_to_cpu(updated_optimizer.state_dict())
                self.to_m_queue.put([updated_policy, optimizer_state_dict])









