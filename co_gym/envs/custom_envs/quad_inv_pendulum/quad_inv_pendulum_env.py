import numpy as np
from co_gym.envs.base import BaseEnv
import random
import torch
from co_gym.utils.utils import action_clip


class QuadInvPendulumEnv(BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.env = env

        # *** Essential Properties (See BaseEnv)***
        self.id = env.id
        self.obs_dim = 14
        self.act_dim = 1
        self.action_bound = [-1, 1]
        # *****************************************

        self.action_max = self.env.action_max
        self._max_episode_steps = 1000
        self.local_step = 0
        self.reset_flag = False

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def step(self, action):
        assert self.reset_flag
        action = action_clip(action, -1, 1)
        action = self.action_max * action
        self.local_step += 1
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if self.local_step == self._max_episode_steps:
            truncated = True

        if terminated or truncated:
            self.reset_flag = False

        return next_state, reward, terminated, truncated, info

    def reset(self):
        self.reset_flag = True
        self.local_step = 0
        init_state, info = self.env.reset()
        return init_state, info

    def random_action_sample(self):
        return np.random.uniform(-1., 1., [self.act_dim])

    def render(self):
        self.env.render()
