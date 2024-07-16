from abc import abstractmethod
import torch
import torch.nn as nn
from typing import (Tuple, TypeVar, Optional, SupportsFloat, Any, Union)
ActType = TypeVar("ActType")
LogProbType = TypeVar("LogProbType")


class BaseCritic(nn.Module):
    def __init__(self, activation_fc_name):
        super(BaseCritic, self).__init__()
        if activation_fc_name in ['ELU', 'elu']:
            self.activation_fc = nn.ELU()
        elif activation_fc_name in ['ReLU', 'relu', 'RELU']:
            self.activation_fc = nn.ReLU()
        else:
            raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_tensor(*inputs):
        pass

    @abstractmethod
    def forward(self, state: torch.FloatTensor, action: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        pass


class BaseActor(nn.Module):
    def __init__(self, activation_fc):
        super(BaseActor, self).__init__()
        if activation_fc in ['ELU', 'elu']:
            self.activation_fc = nn.ELU()
        elif activation_fc in ['ReLU', 'relu', 'RELU']:
            self.activation_fc = nn.ReLU()
        else:
            raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_tensor(*inputs):
        pass

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def get_action(self, state: torch.FloatTensor, eval: bool, worker_device: str) -> Tuple[ActType, Optional[LogProbType]]:
        pass
