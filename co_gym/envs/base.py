from abc import ABC, abstractmethod
import numpy as np
from typing import (Tuple, SupportsFloat)


class BaseEnv(ABC):
    def __init__(self):
        self.id: str
        self.obs_dim: int
        self.act_dim: int
        self.action_bound: list
        super().__init__()

    @staticmethod
    @abstractmethod
    def seed(self, seed):
        '''
        Feeding a random seed to the environment
        '''
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, dict]:
        """
        :return: observation (np.ndarray), info (dict)
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict]:
        """
        :param action: an action provided by the agent
        :return: observation (np.ndarray), reward (float), terminated (bool), truncated (bool), info (dict)
        """
        pass

    @abstractmethod
    def random_action_sample(self) -> np.ndarray:
        """
        :return: a randomly sampled action (np.ndarray)
        """
        pass

