import torch
import torch.nn as nn
import numpy as np
from numba import njit, vectorize

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def calculate_running_mean_var(count, mean, var, new_samples):
    sample_count = len(new_samples)
    sample_mean = np.mean(new_samples, axis=0)
    sample_var = np.var(new_samples, axis=0)

    total_count = count + sample_count
    delta = sample_mean - mean

    # 1. Update Mean
    new_mean = mean + delta * sample_count / total_count

    # 2. Update Variance
    # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    m_a = var * count
    m_b = sample_var * sample_count
    m_a_b = m_a + m_b + np.square(delta) * count * sample_count / total_count
    new_var = m_a_b / total_count

    # 3. Update Count
    new_count = total_count

    return new_mean, new_var, new_count


class RunningMeanVar:
    def __init__(self, shape=()):
        self.count = 0
        self.running_mean = np.zeros(shape=shape)
        self.running_var = np.ones(shape=shape)

    def update(self, new_samples):
        self.running_mean, self.running_var, self.count =\
            calculate_running_mean_var(self.count, self.running_mean, self.running_var, new_samples)

    def get(self):
        return self.running_mean.copy(), self.running_var.copy()


def quantile_huber_loss_f(quantiles, samples, device):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss


def soft_update(network, target_network, tau):
    with torch.no_grad():
        for param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def move_to_cpu(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            new_state_dict[key] = move_to_cpu(value)
        elif isinstance(value, torch.Tensor):
            new_state_dict[key] = value.cpu()
        else:
            new_state_dict[key] = value
    return new_state_dict


@vectorize
def action_clip(x, l, u):
    return max(min(x, u), l)

def close_queue(q):
    while True:
        try:
            o = q.get_nowait()
            del o
        except:
            break
    q.close()
