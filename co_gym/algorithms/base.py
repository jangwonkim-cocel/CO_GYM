from abc import ABC, abstractmethod
from typing import (TypeVar, Any)

CriticType = TypeVar("CriticType")
ActorType = TypeVar("ActorType")
BufferType = TypeVar("BufferType")
OptimType = TypeVar("OptimType")


class BaseAlgorithm(ABC):
    def __init__(self):
        self.name: str
        self.type: str  # "on_policy" or "off_policy"
        self.policy: Any
        self.critic: Any
        self.worker_buffer: Any
        self.sampler_buffer: Any
        super().__init__()

    @abstractmethod
    def train_critic(self, *inputs):
        pass

    @abstractmethod
    def train_actor(self, *inputs):
        pass

    @abstractmethod
    def train_both(self, *inputs):
        pass

    @abstractmethod
    def cal_target(self, buffer: BufferType, critic: CriticType) -> BufferType:
        pass
      
