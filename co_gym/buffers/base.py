from abc import ABC, abstractmethod
import numpy as np


class BaseBuffer(ABC):
    def __init__(self, capacity):
        self.capacity = capacity
        super().__init__()

    @abstractmethod
    def push(self, *args):
        pass

    @abstractmethod
    def push_all(self, *args):
        pass

    @abstractmethod
    def extend(self, buffer):
        pass

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def load_to_device(self, device):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    @abstractmethod
    def clear(self):
        pass

