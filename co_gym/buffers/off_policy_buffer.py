import torch
import numpy as np
from co_gym.buffers.base import BaseBuffer


class OffPolicyBuffer(BaseBuffer):
    def __init__(self, obs_dim, act_dim, capacity):
        super().__init__(capacity)
        self.size = 0
        self.position = 0

        self.state_buffer = np.empty(shape=(self.capacity, obs_dim), dtype=np.float32)
        self.action_buffer = np.empty(shape=(self.capacity, act_dim), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.empty(shape=(self.capacity, obs_dim), dtype=np.float32)
        self.done_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)

    def clear(self):
        self.size = 0
        self.position = 0
        return

    def push(self, state, action, reward, next_state, done):
        self.size = min(self.size + 1, self.capacity)

        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done

        self.position = (self.position + 1) % self.capacity
        return

    def push_all(self, states, actions, rewards, next_states, dones):
        pass

    def extend(self, buffer):
        push_len = buffer.size
        for i in range(push_len):
            self.push(buffer.state_buffer[i],  buffer.action_buffer[i], buffer.reward_buffer[i], buffer.next_state_buffer[i], buffer.done_buffer[i])
        return

    def get_data(self):
        pass

    def load_to_device(self, device):
        pass

    def sample(self, batch_size, device='cuda'):
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = self.state_buffer[idxs]
        actions = self.action_buffer[idxs]
        rewards = self.reward_buffer[idxs]
        next_states = self.next_state_buffer[idxs]
        dones = self.done_buffer[idxs]

        if device == 'cuda':
            return torch.FloatTensor(states).to(device), torch.FloatTensor(actions).to(device),\
                   torch.FloatTensor(rewards).to(device), torch.FloatTensor(next_states).to(device), torch.FloatTensor(dones).to(device)
        else:
            return states, actions, rewards, next_states, dones
