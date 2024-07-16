import mujoco
from co_gym.envs.custom_envs.quadrotor.asset.Utils import *
from co_gym.envs.custom_envs.quadrotor.asset.quadrotor_asset import QuadRotorAsset
from co_gym.envs.custom_envs.quadrotor.asset.viewer import Viewer
from copy import deepcopy
import numpy as np


class QuadrotorBase(QuadRotorAsset):
    def __init__(self):
        super(QuadrotorBase, self).__init__()
        self.id = 'Quadrotor-v0'
        self.init_aggressive = 1
        self.goal = np.array([0., 0., 2.])
        self.room_size = 5.
        self.init_max_pbox = 3.
        self.init_max_vel = 1.
        self.init_max_ang_vel = 2*np.pi

        self.position_dim = 3
        self.velocity_dim = 3
        self.rotation_dim = 6
        self.angular_velocity_dim = 3
        self.state_dim = self.position_dim + self.velocity_dim + self.rotation_dim + self.angular_velocity_dim
        self.action_dim = 4
        self.action_max = np.ones(self.action_dim)
        self.rotor_condition = [1, 1, 1, 1]
        self.use_DR = True
        self.viewer = None

        self.u_hat = 0.1 * np.ones(4)
        self.random_ratio = 0.05
        self.random_min = 1. - self.random_ratio
        self.random_max = 1. + self.random_ratio
        self.thrust_noise_sigma = 0.05
        self.thrust_noise = OUNoise(self.action_dim, sigma=self.thrust_noise_sigma)

        self.mass = self.init_mass
        self.length = self.init_length
        self.kt = self.init_kt
        self.inertia = self.init_inertia
        self.kq = self.init_kq
        self.lag_ratio = 4 * self.sampling_time / 0.1


    def reset(self):
        self.state = np.zeros(12)
        self.action_tanh = np.zeros(self.action_dim)
        # Randomize initial states
        self.state[:3] = self.goal + np.random.uniform(-self.init_max_pbox, self.init_max_pbox, 3) / np.array([1, 1, 2])
        self.state[3:6] = np.random.uniform(-np.pi * self.init_aggressive, np.pi * self.init_aggressive, 3) / np.array([4, 4, 1])
        self.state[6:9] = np.random.uniform(-self.init_max_vel * self.init_aggressive, self.init_max_vel * self.init_aggressive, 3)
        self.state[9:] = np.random.uniform(-self.init_max_ang_vel * self.init_aggressive, self.init_max_ang_vel * self.init_aggressive, 3)

        if self.use_DR:
            # Randomize the parameters of the quadrotor
            self.mass = self.init_mass * np.random.uniform(self.random_min, self.random_max)
            self.length = self.init_length * np.random.uniform(self.random_min, self.random_max)
            self.kt = self.init_kt * np.random.uniform(self.random_min, self.random_max)
            self.inertia = self.init_inertia * np.random.uniform(self.random_min, self.random_max)
            self.kq = self.init_kq * np.random.uniform(self.random_min, self.random_max)
            self.lag_ratio = 4 * self.sampling_time/np.random.uniform(0.05, 0.2)

        obs, _, _ = self._get_obs()

        if self.viewer is not None:
            self.viewer.graph_reset()
        return obs, {}

    def step(self, action_tanh):
        # RL policy controller
        self.action_tanh = action_tanh

        f_hat = (action_tanh + 1)/2

        u_hat = np.sqrt(f_hat)  # rotor angular velocity

        # motor noise
        noise = self.thrust_noise.noise()

        # PID controller
        self.do_simulation(u_hat, self.lag_ratio, noise)
        new_obs, terminated, truncated = self._get_obs()
        reward_with_yaw, reward_without_yaw = self._get_reward(new_obs, f_hat)
        reward = reward_with_yaw if sum(self.rotor_condition) == 4 else reward_without_yaw
        info = {}

        return new_obs, reward, terminated, truncated, info

    def _get_reward(self, obs, act):
        pos_error = obs[:3]   # x, y, z
        vel_error = obs[3:6]  # x, y, z
        orient = obs[6:12]  # roll (sin, cos) / pitch (sin, cos) / yaw (sin, cas)
        ang_vel_error = obs[12:]   # roll (angle) / pitch (angle) / yaw (angle) (+-)

        r_pos = np.exp(-np.linalg.norm(pos_error)*2)
        r_vel = 0.7 + 0.3*np.exp(-np.linalg.norm(vel_error)*2/5)
        r_roll = 0.5 + 0.5*(1 + orient[1])/2  # cos -> 1 (theta = 0) roll -> 0  [0.5~1]
        r_pitch = 0.5 + 0.5*(1 + orient[3])/2 # cos -> 1 (theta = 0) pitch -> 0 [0.5~1]
        r_yaw = (1 + orient[5])/2  # cos -> 1 (theta = 0) yaw -> 0 [0~1] 민감도를 더 주기위해 범위가 더 크다.
        r_rpy_vel = 0.7 + 0.3*np.exp(-np.linalg.norm(ang_vel_error)*2/5)  # angular velocity -> 0
        r_act = 0.9 + 0.13 * np.max(act)  # action에 대한 페널티 (효율성 관련)  0.1 -> 0.13 로 수정 (2024-03-10)

        reward_with_yaw = r_pos * r_vel * r_roll * r_pitch * r_yaw * r_rpy_vel * r_act
        reward_without_yaw = r_pos * r_vel * r_roll * r_pitch * r_rpy_vel * r_act
        return reward_with_yaw, reward_without_yaw

    def _get_obs(self):
        position = self.state[:3]
        velocity = self.state[6:9]
        angular_velocity = self.state[9:12]

        pos_error = self.goal - position
        vel_error = velocity
        ang_vel_error = angular_velocity
        orient = self._get_euler(self.state[3:6]).reshape([-1])
        obs = np.concatenate([pos_error, vel_error, orient, ang_vel_error])

        # Get collision info. (True / False)
        crashed = (position[-1] - (-4.) <= self.length) or (abs(self.state[3]) > np.pi / 2 or abs(self.state[4]) > np.pi / 2)
        escaped = not crashed and not np.array_equal(position, np.clip(position, a_min=-self.room_size, a_max=self.room_size))

        terminated = crashed or escaped

        return obs, terminated, False

    def render(self):
        if self.viewer is None:
            self._viewer_setup()
        if not self.viewer.is_alive:
            self._viewer_reset()
        pos = np.array([self.state[1], self.state[0], -self.state[2]])
        quat = np.array(rpy2quat(self.state[4], self.state[3], -self.state[5]))
        qpos = np.concatenate((pos, quat))
        qvel = np.array([self.state[7], self.state[6], -self.state[8], self.state[10], self.state[9], -self.state[11]])
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.data.userdata = deepcopy(self.action_tanh)
        mujoco.mj_forward(self.model, self.data)
        self.viewer.render()

    def _viewer_setup(self):
        self.model = mujoco.MjModel.from_xml_path('co_gym/envs/custom_envs/quadrotor/mujoco/quadrotor.xml')
        self.data = mujoco.MjData(self.model)
        self._viewer_reset()

    def _viewer_reset(self):
        self.viewer = Viewer(rotor_condition=self.rotor_condition, model=self.model, data=self.data,
                             width=1100, height=600,
                             title='QuadSim_RL',
                             hide_menus=True)
        self.viewer.cam.distance = self.model.stat.extent * 1.25
        self.viewer.cam.lookat[2] -= 2
        self.viewer.cam.elevation += 7
        self.viewer.cam.azimuth = 0