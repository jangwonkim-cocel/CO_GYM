import numpy as np
from co_gym.envs.base import BaseEnv
import random
import torch


class QuadrotorEnv(BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.n_history = 3

        # *** Essential Properties (See BaseEnv)***
        self.id = env.id
        self.obs_dim = self.env.state_dim * self.n_history
        self.act_dim = self.env.action_dim
        self.action_bound = [-1, 1]
        # *****************************************

        self.env.rotor_condition = [1, 1, 1, 1]

        self.state = None
        self.state_dim = self.env.state_dim * self.n_history
        self.action_max = self.env.action_max
        self._max_episode_steps = 1000
        self.action_space_shape = (4,)
        self.local_step = 0
        self.reset_flag = False

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def step(self, action):
        assert self.reset_flag
        action = np.clip(action, a_min=-1, a_max=1)
        action = action * self.env.rotor_condition
        self.local_step += 1
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.state[self.env.state_dim:] = self.state[:-self.env.state_dim]
        self.state[:self.env.state_dim] = next_state

        if self.local_step == self._max_episode_steps:
            truncated = True

        if terminated or truncated:
            self.reset_flag = False

        return self.state.reshape([-1]).copy(), reward, terminated, truncated, info

    def reset(self):
        self.reset_flag = True
        self.local_step = 0
        init_obs, info = self.env.reset()
        self.state = np.concatenate([init_obs] * self.n_history)
        return self.state.reshape([-1]).copy(), info

    def random_action_sample(self):
        return np.random.uniform(-1., 1., [self.act_dim])

    def update_rotor_condition(self, rotor_condition):
        self.env.rotor_condition = rotor_condition
        if self.env.viewer is not None:
            self.env.viewer.rotor_condition = rotor_condition
        return

    def render(self):
        self.env.render()
