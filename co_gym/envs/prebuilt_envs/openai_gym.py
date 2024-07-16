from co_gym.envs.base import BaseEnv
import numpy as np
from typing import (Tuple, SupportsFloat)


class OpenAIGym(BaseEnv):
    def __init__(self, env_id):
        super().__init__()
        import gymnasium
        self.env = gymnasium.make(env_id)
        self.id = env_id
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.action_bound = [self.env.action_space.low[0], self.env.action_space.high[0]]

    def seed(self, seed):
        self.env.action_space.seed(seed)
        return

    def reset(self) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def random_action_sample(self) -> np.ndarray:
        return self.env.action_space.sample()

    def render(self):
        self.env.render()
        return

