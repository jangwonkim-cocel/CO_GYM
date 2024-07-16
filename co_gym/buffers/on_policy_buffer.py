import torch
import numpy as np
from co_gym.buffers.base import BaseBuffer


class OnPolicyBuffer(BaseBuffer):
    def __init__(self, obs_dim, act_dim, capacity):
        super().__init__(capacity)
        self.size = 0
        self.position = 0

        self.state_buffer = np.empty(shape=(self.capacity, obs_dim), dtype=np.float32)
        self.action_buffer = np.empty(shape=(self.capacity, act_dim), dtype=np.float32)
        self.reward_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.empty(shape=(self.capacity, obs_dim), dtype=np.float32)
        self.done_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.log_prob_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.value_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.return_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)
        self.gae_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)

    def clear(self):
        self.size = 0
        self.position = 0
        return

    def push(self, state, action, reward, next_state, done, log_prob, value, _return, gae):
        self.size = min(self.size + 1, self.capacity)

        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = done
        self.log_prob_buffer[self.position] = log_prob
        self.value_buffer[self.position] = value
        self.return_buffer[self.position] = _return
        self.gae_buffer[self.position] = gae

        self.position = (self.position + 1) % self.capacity
        return

    def push_all(self, states, actions, rewards, next_states, dones, log_probs, values, _returns, gaes):
        added_size = len(states)
        self.size = min(self.size + added_size, self.capacity)

        self.state_buffer[self.position: self.position + added_size, :] = states[:added_size, :]
        self.action_buffer[self.position: self.position + added_size, :] = actions[:added_size, :]
        self.reward_buffer[self.position: self.position + added_size, :] = rewards[:added_size, :]
        self.next_state_buffer[self.position: self.position + added_size, :] = next_states[:added_size, :]
        self.done_buffer[self.position: self.position + added_size, :] = dones[:added_size, :]
        self.log_prob_buffer[self.position: self.position + added_size, :] = log_probs[:added_size, :]
        self.value_buffer[self.position: self.position + added_size, :] = values[:added_size, :]
        self.return_buffer[self.position: self.position + added_size, :] = _returns[:added_size, :]
        self.gae_buffer[self.position: self.position + added_size, :] = gaes[:added_size, :]

        self.position = (self.position + added_size)
        assert self.position <= self.capacity
        return

    def extend(self, buffer):
        self.size = min(self.size + buffer.size, self.capacity)

        self.state_buffer[self.position: self.position + buffer.size, :] = buffer.state_buffer[:buffer.size, :]
        self.action_buffer[self.position: self.position + buffer.size, :] = buffer.action_buffer[:buffer.size, :]
        self.reward_buffer[self.position: self.position + buffer.size, :] = buffer.reward_buffer[:buffer.size, :]
        self.next_state_buffer[self.position: self.position + buffer.size, :] = buffer.next_state_buffer[:buffer.size, :]
        self.done_buffer[self.position: self.position + buffer.size, :] = buffer.done_buffer[:buffer.size, :]
        self.log_prob_buffer[self.position: self.position + buffer.size, :] = buffer.log_prob_buffer[:buffer.size, :]
        self.value_buffer[self.position: self.position + buffer.size, :] = buffer.value_buffer[:buffer.size, :]
        self.return_buffer[self.position: self.position + buffer.size, :] = buffer.return_buffer[:buffer.size, :]
        self.gae_buffer[self.position: self.position + buffer.size, :] = buffer.gae_buffer[:buffer.size, :]

        self.position = (self.position + buffer.size)
        assert self.position <= self.capacity
        return

    def get_data(self):
        idxs = np.arange(self.size)
        states = self.state_buffer[idxs]
        actions = self.action_buffer[idxs]
        rewards = self.reward_buffer[idxs]
        next_states = self.next_state_buffer[idxs]
        dones = self.done_buffer[idxs]
        log_probs = self.log_prob_buffer[idxs]
        values = self.value_buffer[idxs]
        returns = self.return_buffer[idxs]
        gaes = self.gae_buffer[idxs]

        return states, actions, rewards, next_states, dones, log_probs, values, returns, gaes

    def load_to_device(self, device):
        self.state_buffer = torch.FloatTensor(self.state_buffer).to(device)
        self.action_buffer = torch.FloatTensor(self.action_buffer).to(device)
        self.reward_buffer = torch.FloatTensor(self.reward_buffer).to(device)
        self.next_state_buffer = torch.FloatTensor(self.next_state_buffer).to(device)
        self.done_buffer = torch.FloatTensor(self.done_buffer).to(device)
        self.log_prob_buffer = torch.FloatTensor(self.log_prob_buffer).to(device)
        self.value_buffer = torch.FloatTensor(self.value_buffer).to(device)
        self.return_buffer = torch.FloatTensor(self.return_buffer).to(device)
        self.gae_buffer = torch.FloatTensor(self.gae_buffer).to(device)
        return

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = self.state_buffer[idxs]
        actions = self.action_buffer[idxs]
        rewards = self.reward_buffer[idxs]
        next_states = self.next_state_buffer[idxs]
        dones = self.done_buffer[idxs]
        log_probs = self.log_prob_buffer[idxs]
        values = self.value_buffer[idxs]
        returns = self.return_buffer[idxs]
        gaes = self.gae_buffer[idxs]

        return states, actions, rewards, next_states, dones, log_probs, values, returns, gaes


class AuxCriticBuffer(BaseBuffer):
    def __init__(self, obs_dim, capacity):
        super().__init__(capacity)
        self.size = 0
        self.position = 0
        self.obs_dim = obs_dim

        self.state_buffer = np.empty(shape=(self.capacity, obs_dim), dtype=np.float32)
        self.return_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)

    def clear(self):
        self.size = 0
        self.position = 0
        return

    def push(self):
        pass

    def push_all(self, states, returns):
        added_size = len(states)
        self.size = min(self.size + added_size, self.capacity)
        self.return_buffer[self.position: self.position + added_size, :] = returns[:added_size, :]

        self.position = (self.position + added_size)
        assert self.position <= self.capacity

    def extend(self, buffer):
        pass

    def get_data(self):
        pass

    def load_to_device(self, device):
        self.state_buffer = torch.FloatTensor(self.state_buffer).to(device)
        self.return_buffer = torch.FloatTensor(self.return_buffer).to(device)
        return

    def unload(self):
        self.state_buffer = np.empty(shape=(self.capacity, self.obs_dim), dtype=np.float32)
        self.return_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = self.state_buffer[idxs]
        returns = self.return_buffer[idxs]

        return states, returns


class AuxActorBuffer(BaseBuffer):
    def __init__(self, obs_dim, act_dim, capacity):
        super().__init__(capacity)
        self.size = 0
        self.position = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.state_buffer = np.empty(shape=(self.capacity, obs_dim), dtype=np.float32)
        self.mean_buffer = np.empty(shape=(self.capacity, act_dim), dtype=np.float32)
        self.log_std_buffer = np.empty(shape=(self.capacity, act_dim), dtype=np.float32)
        self.return_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)

    def clear(self):
        self.size = 0
        self.position = 0
        return

    def push(self):
        pass

    def push_all(self, states, means, log_stds, returns):
        added_size = len(states)
        self.size = min(self.size + added_size, self.capacity)

        self.state_buffer[self.position: self.position + added_size, :] = states[:added_size, :]
        self.mean_buffer[self.position: self.position + added_size, :] = means[:added_size, :]
        self.log_std_buffer[self.position: self.position + added_size, :] = log_stds[:added_size, :]
        self.return_buffer[self.position: self.position + added_size, :] = returns[:added_size, :]

        self.position = (self.position + added_size)
        assert self.position <= self.capacity

    def extend(self, buffer):
        pass

    def get_data(self):
        pass

    def load_to_device(self, device):
        self.state_buffer = torch.FloatTensor(self.state_buffer).to(device)
        self.mean_buffer = torch.FloatTensor(self.mean_buffer).to(device)
        self.log_std_buffer = torch.FloatTensor(self.log_std_buffer).to(device)
        self.return_buffer = torch.FloatTensor(self.return_buffer).to(device)
        return

    def unload(self):
        self.state_buffer = np.empty(shape=(self.capacity, self.obs_dim), dtype=np.float32)
        self.mean_buffer = np.empty(shape=(self.capacity, self.act_dim), dtype=np.float32)
        self.log_std_buffer = np.empty(shape=(self.capacity, self.act_dim), dtype=np.float32)
        self.return_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = self.state_buffer[idxs]
        means = self.mean_buffer[idxs]
        log_stds = self.log_std_buffer[idxs]
        returns = self.return_buffer[idxs]

        return states, means, log_stds, returns


class NormBuffer(BaseBuffer):
    def __init__(self, obs_dim, capacity):
        super().__init__(capacity)
        self.size = 0
        self.position = 0

        self.obs_buffer = np.empty(shape=(self.capacity, obs_dim), dtype=np.float32)
        self.ret_buffer = np.empty(shape=(self.capacity, 1), dtype=np.float32)

    def clear(self):
        self.size = 0
        self.position = 0
        return

    def push(self, obs, ret):
        self.size = min(self.size + 1, self.capacity)

        self.obs_buffer[self.position] = obs
        self.ret_buffer[self.position] = ret

        self.position = (self.position + 1) % self.capacity
        return

    def push_all(self):
        pass

    def extend(self, buffer):
        self.size = min(self.size + buffer.size, self.capacity)

        self.obs_buffer[self.position: self.position + buffer.size, :] = buffer.obs_buffer[:buffer.size, :]
        self.ret_buffer[self.position: self.position + buffer.size, :] = buffer.ret_buffer[:buffer.size, :]

        self.position = (self.position + buffer.size)
        assert self.position <= self.capacity
        return

    def get_data(self):
        idxs = np.arange(self.size)
        obses = self.obs_buffer[idxs]
        rets = self.ret_buffer[idxs]

        return obses, rets

    def load_to_device(self, device):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

