import mujoco
import numpy as np
from copy import deepcopy
from math import sin, cos
from co_gym.envs.custom_envs.quad_inv_pendulum.asset.quad_inv_pendulum_asset import QIPAsset
from co_gym.envs.custom_envs.quad_inv_pendulum.asset.viewer import Viewer


class QuadInvPendulumBase(QIPAsset):
    def __init__(self):
        super(QuadInvPendulumBase, self).__init__()
        self.state_dim = 14
        self.action_dim = 1
        self.action_max = 40.0
        self.pos_max = 0.75
        self.viewer = None
        self.eqi_idx = [0, 1, 2, 4, 5, 7, 8, 10, 11, 13]
        self.reg_idx = [3, 6, 9, 12]
        self.id = 'QuadInvPendulum-v0'

    def set_seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.local_step = 1
        self.total_reward = 0
        q = .01 * np.random.randn(5) + np.array([0., np.pi, np.pi, np.pi, np.pi])
        qd = .01 * np.random.randn(5)
        self.state = np.concatenate([q, qd])
        self.prev_state = np.concatenate([q, qd])
        obs = self._get_obs()
        if self.viewer is not None:
            self.viewer.graph_reset()
        return obs, {}

    def step(self, action):
        # RL policy controller
        self.prev_state = self.state.copy()
        self.action_tanh = action / self.action_max
        self._do_simulation(action)
        new_obs = self._get_obs()
        reward, terminated = self._get_reward(new_obs, self.action_tanh)
        info = {}
        self.total_reward += reward
        return new_obs, reward, terminated, False, info

    def _get_reward(self, obs, act):
        pos, cos_th, th_dot = obs[0], obs[[3,6,9,12]], obs[[4,7,10,13]]
        notdone = np.isfinite(obs).all() and (np.abs(pos) <= self.pos_max)
        notdone = notdone and np.all(np.abs(th_dot) < 35.)
        r_pos = 0.5 + 0.5 * np.exp(-0.7 * pos ** 2)
        r_act = 0.8 + 0.2 * np.maximum(1 - (act ** 2), 0.0)
        target_cos = np.array([1,1,1,1])
        r_angle = np.prod(0.5 + 0.5 * target_cos * cos_th)
        r_vel = np.min(0.5 + 0.5 * np.exp(-0.2 * th_dot ** 2))
        reward = r_pos * r_act * r_angle * r_vel
        done = not notdone
        return reward, done

    def _get_obs(self):
        ang1 = self.state[1]
        ang2 = self.state[2] - self.state[1]
        ang3 = self.state[3] - self.state[2]
        ang4 = self.state[4] - self.state[3]
        return np.array([self.state[0], self.state[5], #cart_vel,
                         sin(ang1), cos(ang1), self.state[6], #ang_vel[0],
                         sin(ang2), cos(ang2), self.state[7], #ang_vel[1],
                         sin(ang3), cos(ang3), self.state[8], #ang_vel[2],
                         sin(ang4), cos(ang4), self.state[9]]) #ang_vel[3]])

    def render(self):
        if self.viewer is None:
            self._viewer_setup()
        if not self.viewer.is_alive:
            self._viewer_reset()
        ang1 = self.state[1]
        ang2 = self.state[2] - self.state[1]
        ang3 = self.state[3] - self.state[2]
        ang4 = self.state[4] - self.state[3]
        qpos = np.array([self.state[0], ang1, ang2, ang3, ang4])
        qvel = np.array([self.state[5], self.state[6], self.state[7], self.state[8], self.state[9]])
        self.data.qpos = qpos
        self.data.qvel = qvel
        self.data.userdata[0] = deepcopy(self.action_tanh)
        self.data.userdata[1] = deepcopy(self.total_reward)
        mujoco.mj_forward(self.model, self.data)
        self.viewer.render()

    def _viewer_setup(self):
        self.model = mujoco.MjModel.from_xml_path('co_gym/envs/custom_envs/quad_inv_pendulum/mujoco/quad_inv_pendulum.xml')
        self.data = mujoco.MjData(self.model)
        self._viewer_reset()

    def _viewer_reset(self):
        self.viewer = Viewer(model=self.model, data=self.data,
                             width=1100, height=600,
                             title='CoCEL_QIP',
                             hide_menus=True)
        self.viewer.cam.distance = self.model.stat.extent * 1.2
        self.viewer.cam.lookat[2] += 0.3
        self.viewer.cam.elevation += 35
        self.viewer.cam.azimuth = 205