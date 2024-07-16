from co_gym.architectures.D2AC.actor_learner import ActorLearner
from co_gym.architectures.D2AC.critic_learner import CriticLearner
from co_gym.architectures.D2AC.monitor import Monitor
from co_gym.architectures.D2AC.data_loader import DataLoader
from co_gym.architectures.D2AC.worker import Worker
from co_gym.architectures.D2AC.wrapper_manager import WrapperManager
from co_gym.utils.utils import close_queue
import torch.multiprocessing as mp
import torch
import inspect
import time
import json

MAX_QUEUE_SIZE = 3


class D2AC(object):
    def __init__(self, env, algorithm, config):
        assert config['random_seed'] > 0
        assert algorithm.type == 'on_policy'
        assert config['n_workers'] * config['max_rollout'] >= config['batch_size']
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        mp.set_sharing_strategy('file_system')

        self.env = env
        self.algorithm = algorithm
        self.config = config

        self.data_loader_to_critic_queue = mp.Queue(maxsize=MAX_QUEUE_SIZE)
        self.data_loader_to_actor_queue = mp.Queue(maxsize=MAX_QUEUE_SIZE)
        self.actor_to_worker_queue = mp.Queue()
        self.critic_to_worker_queue = mp.Queue()
        self.worker_to_data_loader = mp.Queue()
        self.actor_to_monitor_queue = mp.Queue(maxsize=MAX_QUEUE_SIZE)
        self.critic_to_monitor_queue = mp.Queue(maxsize=MAX_QUEUE_SIZE)

        self.total_epochs = mp.Value('i', 0)

        if config['normalize_obs'] or config['normalize_reward']:
            self.worker_to_wrapper_manager_queue = mp.Queue()
            self.wrapper_manager_to_worker_queue = mp.Queue()
            self.wrapper_manager_to_monitor_queue = mp.Queue()
        else:
            self.worker_to_wrapper_manager_queue = 'N/A'
            self.wrapper_manager_to_worker_queue = 'N/A'
            self.wrapper_manager_to_monitor_queue = 'N/A'

        self.monitor = Monitor(env, self.critic_to_monitor_queue, self.actor_to_monitor_queue,
                               self.wrapper_manager_to_monitor_queue, self.total_epochs, config)

        self.wrapper_manager = WrapperManager(env, self.worker_to_wrapper_manager_queue,
                                              self.wrapper_manager_to_worker_queue,
                                              self.wrapper_manager_to_monitor_queue, config)

        self.critic_learner = CriticLearner(algorithm, self.data_loader_to_critic_queue, self.critic_to_worker_queue,
                                            self.critic_to_monitor_queue, config)

        self.actor_learner = ActorLearner(algorithm, self.data_loader_to_actor_queue, self.actor_to_worker_queue,
                                          self.actor_to_monitor_queue, config)

        self.data_loader = DataLoader(algorithm, self.data_loader_to_critic_queue, self.data_loader_to_actor_queue, self.worker_to_data_loader, config)

        self.workers = [Worker(algorithm, env, config['random_seed'] + rank, self.worker_to_data_loader,
                               self.critic_to_worker_queue, self.actor_to_worker_queue,
                               self.wrapper_manager_to_worker_queue, self.worker_to_wrapper_manager_queue, config) for rank in range(config['n_workers'])]

        if config['load_model']:
            checkpoint = torch.load(inspect.getfile(self.__class__)[:-26] + 'log/' + config['load_checkpoint_dir'] + '/checkpoint.pt')
            self.actor_learner.optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.critic_learner.optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    @staticmethod
    def work(worker):
        worker.work()
        return

    def critic_learn(self):
        self.critic_learner.learn()
        return

    def actor_learn(self):
        self.actor_learner.learn()
        return

    def load_data(self):
        self.data_loader.load_data()
        return

    def wrapper_manage(self):
        self.wrapper_manager.manage()
        return

    def monitor_(self):
        self.monitor.monitoring()
        return

    def train(self):
        print("Waiting for initializing all the processes ...")
        processes = []
        processes.append(mp.Process(target=self.load_data, args=()))
        processes.append(mp.Process(target=self.critic_learn, args=()))
        processes.append(mp.Process(target=self.actor_learn, args=()))
        processes.append(mp.Process(target=self.monitor_, args=()))
        processes.append(mp.Process(target=self.wrapper_manage, args=()))
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
                close_queue(self.data_loader_to_critic_queue)
                close_queue(self.data_loader_to_actor_queue)
                close_queue(self.actor_to_worker_queue)
                close_queue(self.critic_to_worker_queue)
                close_queue(self.worker_to_data_loader)
                close_queue(self.actor_to_monitor_queue)

                for p in processes:
                    p.terminate()
                print("Done!")
                return

