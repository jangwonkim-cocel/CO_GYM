import numpy as np
from typing import (Tuple, TypeVar, SupportsFloat, Any, Union)
from co_gym.envs.base import BaseEnv
ActType = TypeVar("ActType")


class NormalizedEnv(BaseEnv):
    def __init__(self, env):
        self.env = env
        self.id = env.id
        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.action_bound = env.action_bound
        self.running_obs_mean = np.zeros(shape=self.obs_dim)
        self.running_obs_var = np.ones(shape=self.obs_dim)
        self.running_ret_var = np.ones(shape=())
        super().__init__()

    def seed(self, seed):
        self.env.seed(seed)
        return

    def set_obs_mean_var(self, obs_mean, obs_var):
        self.running_obs_mean = obs_mean
        self.running_obs_var = obs_var

    def set_ret_var(self, ret_var):
        self.running_ret_var = ret_var

    def normalize_obs(self, obs: Any) -> Any:
        return (obs - self.running_obs_mean) / np.sqrt(self.running_obs_var + 1e-6)

    def normalize_reward(self, reward: SupportsFloat) -> SupportsFloat:
        return reward / np.sqrt(self.running_ret_var + 1e-6)

    def reset(self) -> Tuple[Any, dict]:
        obs, info = self.env.reset()
        obs = self.normalize_obs(obs)
        return obs, info

    def step(self, action: ActType) -> Tuple[Any, SupportsFloat, bool, bool, dict]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        info['true_next_obs'] = next_obs
        info['true_reward'] = reward
        next_obs = self.normalize_obs(next_obs)
        reward = self.normalize_reward(reward)
        return next_obs, reward, terminated, truncated, info

    def random_action_sample(self) -> Union[int, list, np.ndarray]:
        return self.env.random_action_sample()

    def render(self):
        self.env.render()
