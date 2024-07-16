import copy
from co_gym.buffers.on_policy_buffer import NormBuffer
from co_gym.utils.utils import RunningMeanVar
import pickle
import inspect
import time
import os


class WrapperManager:
    def __init__(self, env, from_worker_queue, to_worker_queue, to_monitor_queue, config):
        self.env = copy.deepcopy(env)
        self.epochs = 0
        self.checkpoint_no = 0
        self.config = config
        self.from_w_queue = from_worker_queue
        self.to_w_queue = to_worker_queue
        self.to_m_queue = to_monitor_queue
        self.max_data_size = self.config['max_rollout'] * self.config['n_workers']
        self.is_env_normalized = config['normalize_obs'] or config['normalize_reward']
        if self.is_env_normalized:
            self.norm_buffer = NormBuffer(env.obs_dim, capacity=int(config['max_rollout'] * config['n_workers']))
            self.obs_mean_var, self.ret_mean_var = RunningMeanVar(shape=env.obs_dim), RunningMeanVar(shape=())
            if config['load_model']:
                import co_gym
                abs_path = inspect.getfile(co_gym)
                self.checkpoint_dir = abs_path[:-11] + 'log/' + config['load_checkpoint_dir']
                with open(self.checkpoint_dir + '/pickled_data/ObsRunningMeanVar_class.pickle', 'rb') as p1:
                    self.obs_mean_var = pickle.load(p1)
                with open(self.checkpoint_dir + '/pickled_data/RetRunningMeanVar_class.pickle', 'rb') as p2:
                    self.ret_mean_var = pickle.load(p2)
        else:
            self.norm_buffer = None
            self.obs_mean_var, self.ret_mean_var = None, None
        print(f"Load Wrapper Manager")

    def update_mean_variance(self, new_obses, new_rets):
        self.obs_mean_var.update(new_obses)
        self.ret_mean_var.update(new_rets)

    def manage(self):
        if not (self.config['normalize_obs'] or self.config['normalize_reward']):
            return

        while True:
            data_size = 0
            while data_size < self.max_data_size:
                worker_norm_buffer = self.from_w_queue.get()
                data_size += worker_norm_buffer.size
                self.norm_buffer.extend(worker_norm_buffer)
            assert self.max_data_size == data_size

            new_obses, new_rets = self.norm_buffer.get_data()
            self.norm_buffer.clear()
            self.update_mean_variance(new_obses, new_rets)
            updated_obs_mean, updated_obs_var = self.obs_mean_var.get()
            _, updated_ret_var = self.ret_mean_var.get()

            normalizing_info = copy.deepcopy([updated_obs_mean, updated_obs_var, updated_ret_var])
            for _ in range(self.config['n_workers']):
                self.to_w_queue.put(normalizing_info)

            self.to_m_queue.put([self.obs_mean_var, self.ret_mean_var])
            time.sleep(0.005)

