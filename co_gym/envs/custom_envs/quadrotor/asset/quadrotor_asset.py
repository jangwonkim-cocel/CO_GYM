import numpy as np
import math


class QuadRotorAsset():
    def __init__(self):
        # Carbon Mini Drone
        self.init_mass = 0.640
        self.init_length = 0.14
        self.init_inertia = np.diag([0.00728, 0.00728, 0.00993])
        self.init_kt = 4.612e-8
        self.init_max_rpm = 14945.4
        self.init_kq = 6.411e-10
        self.gravity = 9.8066

        self.step_size = 0.001
        self.sampling_time = 0.01
        self.init_lag_ratio = 1.0
        self.n_frame = int(self.sampling_time / self.step_size)

        assert self.sampling_time % self.step_size == 0
        self.mass = self.init_mass  # mass
        self.length = self.init_length  # arm length
        self.inertia = self.init_inertia  # inertia matrix

        self.u_hat = 0.1 * np.ones(4)
        self.tc = 0.06  # time constant between motor and propeller
        self.kt = self.init_kt  # RPM to Thrust coefficient
        self.max_rpm = self.init_max_rpm  # Maximum RPM
        self.kq = self.init_kq

    def do_simulation(self, u_hat, lag_ratio, noise):
        # RK4
        for _ in range(self.n_frame):
            # motor lag
            u_hat_lagged = lag_ratio * (u_hat - self.u_hat) + self.u_hat
            self.u_hat = u_hat_lagged

            # motor noise
            f = np.clip((u_hat_lagged + noise) ** 2, 0., 1.)

            # Normalized force to real force
            fmax = self.kt * self.max_rpm ** 2
            f *= fmax

            xd1 = self._get_state_dot(self.state, f)
            xd2 = self._get_state_dot(self.state + (self.step_size/2)*xd1, f)
            xd3 = self._get_state_dot(self.state + (self.step_size/2)*xd2, f)
            xd4 = self._get_state_dot(self.state + self.step_size*xd3, f)
            xd = (xd1 + 2*xd2 + 2*xd3 + xd4)/6
            self.state += self.step_size*xd

    def set_state(self, qpos, qvel):
        self.state = np.concatenate((qpos, qvel))

    def _get_state_dot(self, state, thrust):
        """
        thrust info
        f1 : x-axis, positive, ccw
        f2 : x-axis, negative, ccw
        f3 : y-axis, positive, cw
        f4 : y-axis, negative, cw
        """
        f1, f2, f3, f4 = thrust
        _, _, _, phi, tht, psi, x_dot, y_dot, z_dot, p, q, r = state.copy()

        R = self._get_rotation_matrix([phi, tht, psi])
        C_inv = self._get_C_inv([phi, tht, psi])

        x_2dot = -R[0][2]*(f1 + f2 + f3 + f4)/self.mass
        y_2dot = -R[1][2]*(f1 + f2 + f3 + f4)/self.mass
        z_2dot = -R[2][2]*(f1 + f2 + f3 + f4)/self.mass + self.gravity

        phi_dot = C_inv[0][0]*p + C_inv[0][1]*q + C_inv[0][2]*r
        tht_dot = C_inv[1][0]*p + C_inv[1][1]*q + C_inv[1][2]*r
        psi_dot = C_inv[2][0]*p + C_inv[2][1]*q + C_inv[2][2]*r

        p_dot = ((self.inertia[1][1] - self.inertia[2][2]) / self.inertia[0][0]) * q * r + self.length * (f2 - f1) / self.inertia[0][0]
        q_dot = ((self.inertia[2][2] - self.inertia[0][0]) / self.inertia[1][1]) * p * r + self.length * (f3 - f4) / self.inertia[1][1]
        r_dot = ((self.inertia[0][0] - self.inertia[1][1]) / self.inertia[2][2]) * p * q + (self.kq / self.kt) * (f1 + f2 - f3 - f4) / self.inertia[2][2]

        return np.array([x_dot, y_dot, z_dot,
                         phi_dot, tht_dot, psi_dot,
                         x_2dot, y_2dot, z_2dot,
                         p_dot, q_dot, r_dot])

    def _get_rotation_matrix(self, inp):
        if len(inp) != 0:
            phi, theta, psi = inp
        else:
            phi, theta, psi = self.state[3:6]

        return np.array([
            [math.cos(theta)*math.cos(psi), math.sin(phi)*math.sin(theta)*math.cos(psi) - math.cos(phi)*math.sin(psi), math.cos(phi)*math.sin(theta)*math.cos(psi) + math.sin(phi)*math.sin(psi)],
            [math.cos(theta)*math.sin(psi), math.sin(phi)*math.sin(theta)*math.sin(psi) + math.cos(phi)*math.cos(psi), math.cos(phi)*math.sin(theta)*math.sin(psi) - math.sin(phi)*math.cos(psi)],
            [-math.sin(theta), math.sin(phi)*math.cos(theta), math.cos(phi)*math.cos(theta)]])

    def _get_euler(self, inp):
        if len(inp) != 0:
            phi, theta, psi = inp
        else:
            phi, theta, psi = self.state[3:6]
        return np.array([
            [math.sin(phi), math.cos(phi)],
            [math.sin(theta), math.cos(theta)],
            [math.sin(psi), math.cos(psi)]])

    def _get_C_inv(self, inp):     # inverse matrix of Euler rate to Body angular rate matrix
        if len(inp) != 0:
            phi, theta, psi = inp
        else:
            phi, theta, psi = self.state[3:6]

        return np.array([
            [1, (math.sin(phi)*math.sin(theta))/math.cos(theta), (math.cos(phi)*math.sin(theta))/math.cos(theta)],
            [0, math.cos(phi), -math.sin(phi)],
            [0, math.sin(phi)/math.cos(theta), math.cos(phi)/math.cos(theta)]])

    def _thrust_to_pwm(self, thrust):   # pwm range = 122. 604573 ~ 976.45
        return -93.27*thrust**4 + 450.9*thrust**3 - 759.7*thrust**2 + 931.9*thrust + 36.57

    def _pwm_to_rpm(self, pwm):     # rpm range = 3266.50557168 ~ 14617.16784233
        return -0.006748*pwm**2 + 20.71*pwm + 828.8

    def _rpm_to_thrust(self, rpm):  # Thrust range (N) : 0.1 ~ 2.0
        return 9.168e-9*rpm**2

    def _rpm_to_torque(self, rpm): # torque range (N*m): 0.0012441288385679795 ~ 0.02491294206221667
        return 1.166e-10*rpm**2
