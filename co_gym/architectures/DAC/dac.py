from co_gym.architectures.DAC.learner import Learner
from co_gym.architectures.DAC.monitor import Monitor
from co_gym.architectures.DAC.worker import Worker
import torch
import torch.multiprocessing as mp
import inspect
import time
import json
from co_gym.utils.utils import close_queue


class DAC(object):
    def __init__(self, env, algorithm, config):
        assert config['random_seed'] > 0
        assert algorithm.type == 'off_policy'
        assert (config['n_workers'] * config['max_rollout'] >= config['batch_size']) or (config['update_after'] > config['batch_size'])
        assert config['offline_buffer_capacity'] > config['batch_size']

        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        self.env = env
        self.algorithm = algorithm
        self.config = config

        self.worker_to_learner_queue = mp.Queue()
        self.learner_to_worker_queue = mp.Queue()
        self.learner_to_monitor_queue = mp.Queue(maxsize=2)

        self.total_epochs = mp.Value('i', 0)

        self.learner = Learner(algorithm, self.worker_to_learner_queue, self.learner_to_worker_queue,
                               self.learner_to_monitor_queue, config)

        self.workers = [Worker(algorithm, env, config['random_seed'] + rank, self.worker_to_learner_queue,
                               self.learner_to_worker_queue,  config) for rank in range(config['n_workers'])]

        self.monitor = Monitor(env, self.learner_to_monitor_queue,  self.total_epochs, config)

        if config['load_model']:
            checkpoint = torch.load(inspect.getfile(self.__class__)[:-24] + 'log/' + config['load_checkpoint_dir'] + '/checkpoint.pt')
            self.learner.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.learner.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

            if checkpoint['log_alpha'] is not None:
                self.learner.algorithm.log_alpha = torch.asarray([checkpoint['log_alpha']], requires_grad=True, device=config['learner_device'])
                self.learner.algorithm.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            else:
                self.learner.algorithm.log_alpha = None
                self.learner.algorithm.alpha_optimizer = None

            self.learner.buffer = checkpoint['buffer']

    @staticmethod
    def work(worker):
        worker.work()
        return

    def learner_learn(self):
        self.learner.learn()
        return

    def monitor_(self):
        self.monitor.monitoring()
        return

    def train(self):
        print("Waiting for initializing all the processes ...")
        processes = []
        processes.append(mp.Process(target=self.learner_learn, args=()))
        processes.append(mp.Process(target=self.monitor_, args=()))
        for worker in self.workers:
            w = mp.Process(target=self.work, args=(worker,))
            processes.append(w)

        for p in processes:
            p.start()

        print("Done!")

        # Print the Configuration
        print(json.dumps(self.config, indent=4))

        print("Start training ... ")

        while True:
            time.sleep(10)
            if self.total_epochs.value >= self.config['max_epochs']:
                close_queue(self.worker_to_learner_queue)
                close_queue(self.learner_to_worker_queue)
                close_queue(self.learner_to_monitor_queue)
                for p in processes:
                    p.terminate()
                print("Done!")
                return


