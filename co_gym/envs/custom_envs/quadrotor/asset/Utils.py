import numpy as np
import numpy.random as nr
import pandas as pd
import math
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0

sys.path.append(str(Path('Utils.py').parent.absolute()))  # 절대 경로에 추가

class DataManager:
    def __init__(self, path_dim=2, data_name=None):
        self.data = None
        self.data_name = data_name
        self.path_dim = path_dim
        self.path_data = np.empty([1, path_dim])

    def init_data(self):
        self.data = None

    def put_data(self, obs):
        if self.data is None:
            self.data = obs
        else:
            self.data = np.vstack((self.data, obs))

    def put_path(self, obs):
        self.path_data = np.vstack((self.path_data, obs[:3]))

    def mean_data(self):
        mean_data = np.mean(self.data, axis=0)
        return mean_data

    def plot_data(self, obs, label=None):
        self.put_data(obs)
        if label is None:
            if self.data_name is not None:
                plt.figure(self.data_name)
            plt.plot(self.data)
        else:
            plt.plot(self.data, label=label)
            plt.legend()
        plt.show(block=False)
        plt.pause(0.0001)
        plt.cla()

    def plot_path(self, obs, label=None):

        self.put_path(obs)
        if label is None:
            plt.plot(self.path_data[:, i] for i in range(self.path_dim))
        else:
            plt.plot([self.path_data[:, i] for i in range(self.path_dim)], label=label)
            plt.legend()
        plt.show(block=False)
        plt.pause(0.0001)
        plt.cla()

    def save_data(self, path, fname, numpy=False):

        if numpy is False:
            df = pd.DataFrame(self.data)
            df.to_csv(path + fname + ".csv")
        else:
            df = np.array(self.data)
            np.save(path + fname + ".npy", df)

    def save_path(self, path, fname, numpy=False):
        if numpy is False:
            df = pd.DataFrame(self.data)
            df.to_csv(path + fname + ".csv")
        else:
            df = np.array(self.path_data)
            np.save(path + fname + ".npy", df)

## related to control ##
def quat2mat(quat):
    """ Convert Quaternion to Rotation matrix.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def rot2rpy(R):
    temp = np.array([0, 0, 1]) @ R
    pitch = math.asin(-temp[0])
    roll = math.asin(temp[1] / math.cos(pitch))
    yaw = math.acos(R[0, 0] / math.cos(pitch))

    return roll, pitch, yaw


def quat2rpy(quat):
    R = quat2mat(quat)
    euler = rot2rpy(R)
    euler = np.array(euler)
    return euler

def add_noise(val, scale=0.1):
    val += scale*np.random.normal(size=len(val))
    return val

def rpy2quat(roll, pitch, yaw):

    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return [qw, qx, qy, qz]

class OUNoise:
    """Ornstein–Uhlenbeck process"""
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state
