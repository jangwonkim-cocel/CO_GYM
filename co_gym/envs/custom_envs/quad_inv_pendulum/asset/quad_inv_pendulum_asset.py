import numpy as np
from math import sin, cos


class QIPAsset():
    def __init__(self):

        # Physical parameters
        self.sample_time = 0.02
        self.frame_skip = 10
        self.step_size = self.sample_time / self.frame_skip
        assert self.sample_time % self.step_size == 0

    def _do_simulation(self, a):
        # RK4
        for i in range(self.frame_skip):
            xd1 = self._get_state_dot(self.state, a)
            xd2 = self._get_state_dot(self.state + (self.step_size / 2) * xd1, a)
            xd3 = self._get_state_dot(self.state + (self.step_size / 2) * xd2, a)
            xd4 = self._get_state_dot(self.state + self.step_size * xd3, a)
            xd = (xd1 + 2 * xd2 + 2 * xd3 + xd4) / 6
            self.state += self.step_size * xd

    def set_state(self, qpos, qvel):
        self.state = np.concatenate((qpos, qvel))

    def _get_state_dot(self, x, a):

        l1 = 0.190; l2 = 0.226; l3 = 0.262; l4 = 0.298
        a1 = 0.120; a2 = 0.145; a3 = 0.178; a4 = 0.140
        m1 = 0.5; m2 = 0.6; m3 = 0.6; m4 = 0.4
        J1 = a1*a1*m1/3; J2 = a2*a2*m2/3; J3 = a3*a3*m3/3; J4 = a4*a4*m4/3
        d1 = 0.001547562611052; d2 = 0.0008598850094355782; d3 = 0.0006480318611403285; d4 = 0.000586515
        g = 9.81

        q, qd = x[:5], x[5:]

        m11 = J1 + (a1**2)*m1 + (l1**2)*(m2 + m3 + m4)
        m12 = (a2*l1*m2 + l1*l2*m3 + l1*l2*m4) * cos(q[1]-q[2])
        m13 = (a3*l1*m3 + l1*l3*m4) * cos(q[1]-q[3])
        m14 = (a4*l1*m4) * cos(q[1]-q[4])
        m21 = (a2*l1*m2 + l1*l2*m3 + l1*l2*m4) * cos(q[1]-q[2])
        m22 = J2 + (a2**2)*m2 + (l2**2)*m3 + (l2**2)*m4
        m23 = (a3*l2*m3 + l2*l3*m4) * cos(q[2]-q[3])
        m24 = (a4*l2*m4) * cos(q[2]-q[4])
        m31 = (a3*l1*m3 + l1*l3*m4) * cos(q[1]-q[3])
        m32 = (a3*l2*m3 + l2*l3*m4) * cos(q[2]-q[3])
        m33 = J3 + (a3**2)*m3 + (l3**2)*m4
        m34 = (a4*l3*m4) * cos(q[3]-q[4])
        m41 = (a4*l1*m4) * cos(q[1]-q[4])
        m42 = (a4*l2*m4) * cos(q[2]-q[4])
        m43 = (a4*l3*m4) * cos(q[3]-q[4])
        m44 = J4 + (a4**2)*m4
        M = np.array([[m11,m12,m13,m14],
                      [m21,m22,m23,m24],
                      [m31,m32,m33,m34],
                      [m41,m42,m43,m44]])

        c11 = c22 = c33 = c44 = 0
        c12 = qd[2] * (a2*l1*m2 + l1*l2*m3 + l1*l2*m4) * sin(q[1]-q[2])
        c13 = qd[3] * (a3*l1*m3 + l1*l3*m4) * sin(q[1]-q[3])
        c14 = qd[4] * (a4*l1*m4) * sin(q[1]-q[4])
        c21 = -qd[1] * (a2*l1*m2 + l1*l2*m3 + l1*l2*m4) * sin(q[1]-q[2])
        c23 = qd[3] * (a3*l2*m3 + l2*l3*m4) * sin(q[2]-q[3])
        c24 = qd[4] * (a4*l2*m4) * sin(q[2]-q[4])
        c31 = -qd[1] * (a3*l1*m3 + l1*l3*m4) * sin(q[1]-q[3])
        c32 = -qd[2] * (a3*l2*m3 + l2*l3*m4) * sin(q[2]-q[3])
        c34 = qd[4] * (a4*l3*m4) * sin(q[3]-q[4])
        c41 = -qd[1] * (a4*l1*m4) * sin(q[1]-q[4])
        c42 = -qd[2] * (a4*l2*m4) * sin(q[2]-q[4])
        c43 = -qd[3] * (a4*l3*m4) * sin(q[3]-q[4])
        C = np.array([[c11,c12,c13,c14],
                      [c21,c22,c23,c24],
                      [c31,c32,c33,c34],
                      [c41,c42,c43,c44]])

        d11 = d1 + d2
        d12 = -d2
        d13, d14 = 0, 0
        d21 = -d2
        d22 = d2 + d3
        d23 = -d3
        d24 = 0
        d31 = 0
        d32 = -d3
        d33 = d3 + d4
        d34 = -d4
        d41, d42 = 0, 0
        d43 = -d4
        d44 = d4
        D = np.array([[d11, d12, d13, d14],
                      [d21, d22, d23, d24],
                      [d31, d32, d33, d34],
                      [d41, d42, d43, d44]])

        g11 = -g * (a1*m1 + l1*m2 + l1*m3 + l1*m4) * sin(q[1])
        g21 = -g * (a2*m2 + l2*m3 + l2*m4) * sin(q[2])
        g31 = -g * (a3*m3 + l3*m4) * sin(q[3])
        g41 = -g * (a4*m4) * sin(q[4])
        G = np.array([g11, g21, g31, g41]).T

        b11 = (a1*m2 + l1*m2 + l1*m3 + l1*m4) * cos(q[1])
        b21 = (a2*m2 + l2*m3 + l2*m4) * cos(q[2])
        b31 = (a3*m3 + l3*m4) * cos(q[3])
        b41 = (a4*m4) * cos(q[4])
        B = np.array([b11, b21, b31, b41]).T

        Minv = np.linalg.inv(M)
        F0 = -Minv @ (C @ qd[1:].T + D @ qd[1:].T + G)
        F1 = Minv @ B

        fx = np.concatenate([qd, np.zeros(1), F0.T])
        gx = np.concatenate([np.zeros(5),np.ones(1),F1.T])
        xd = fx + gx * a
        return xd