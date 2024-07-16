import copy
import time


class DataLoader:
    def __init__(self, algorithm, to_critic_learner_queue, to_actor_learner_queue, from_worker_queue, config):
        self.max_data_size = config['max_rollout'] * config['n_workers']
        self.algorithm = copy.deepcopy(algorithm)
        self.buffer = self.algorithm.global_buffer
        self.to_c_l_queue = to_critic_learner_queue
        self.to_a_l_queue = to_actor_learner_queue
        self.from_w_queue = from_worker_queue
        self.config = config
        print(f"Load Data Loader")

    def load_data(self):
        while True:
            data_size = 0
            while data_size < self.max_data_size:
                worker_buffer = self.from_w_queue.get()
                data_size += worker_buffer.size
                self.buffer.extend(worker_buffer)

            buffer = copy.deepcopy(self.buffer)   # safe transfer
            time.sleep(0.005)
            self.to_c_l_queue.put(buffer)
            self.to_a_l_queue.put(buffer)
            self.buffer.clear()



