#####################################################################################
'''
MuJoCo viewer for QuadSim_RL (mujoco_viewer wrapper)

Author  : Jongchan Baek
Date    : 2022.10.20
Contact : paekgga@postech.ac.kr
'''
#####################################################################################

import mujoco
import mujoco_viewer
import glfw, time
from co_gym.envs.custom_envs.quadrotor.asset.Utils import quat2rpy


class Viewer(mujoco_viewer.MujocoViewer):
    def __init__(self, rotor_condition, width, height, **kwargs):
        super().__init__(width=width, height=height, **kwargs)
        self.width, self.height = width, height
        self.n_timestep = 1000
        self.rotor_condition = rotor_condition
        self.graph_reset()

    def graph_reset(self):
        fig1, fig_viewport1 = self._set_figure(0, self.viewport.height, ndata=3)
        fig2, fig_viewport2 = self._set_figure(0, 2*int(self.viewport.height/4), ndata=1)
        fig3, fig_viewport3 = self._set_figure(0, 3*int(self.viewport.height/4), ndata=1)
        fig4, fig_viewport4 = self._set_figure(0, 0, ndata=1)
        fig5, fig_viewport5 = self._set_figure(0, self.viewport.height, ndata=1)
        fig6, fig_viewport6 = self._set_figure(self.viewport.width-200, 2*int(self.viewport.height/4), ndata=1)
        fig7, fig_viewport7 = self._set_figure(self.viewport.width-200, 3*int(self.viewport.height/4), ndata=1)
        fig8, fig_viewport8 = self._set_figure(self.viewport.width-200, 0, ndata=1)

        fig9, fig_viewport9 = self._set_words(0, 0)  # My
        self.fig = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9]

        self.fig_viewport = [fig_viewport1, fig_viewport2, fig_viewport3, fig_viewport4,
                             fig_viewport5, fig_viewport6, fig_viewport7, fig_viewport8, fig_viewport9]
        fig1.range[1][0] = -4.0
        fig1.range[1][1] = 4.0
        fig1.linergb[0] = [1, 1, 1]
        fig1.linergb[1] = [1, 0, 1]
        fig1.linergb[2] = [0, 1, 0]
        fig1.linewidth = 1.3
        fig1.figurergba = [0, 0, 0, 0.1]
        fig1.flg_ticklabel[0] = 0
        fig1.linename[0] = 'x'
        fig1.linename[1] = 'y'
        fig1.linename[2] = 'z'
        fig2.range[1][0] = -1.8
        fig2.range[1][1] = 1.8
        fig2.linergb = [1, 1, 1]
        fig2.linewidth = 1.3
        fig2.figurergba = [0, 0, 0, 0.1]
        fig2.linename[0] = 'roll'
        fig2.flg_ticklabel[0] = 0
        fig3.range[1][0] = -1.8
        fig3.range[1][1] = 1.8
        fig3.linergb = [1, 0, 1]
        fig3.linewidth = 1.3
        fig3.figurergba = [0, 0, 0, 0.1]
        fig3.linename[0] = 'pitch'
        fig3.flg_ticklabel[0] = 0
        fig4.range[1][0] = -3.14
        fig4.range[1][1] = 3.14
        fig4.linergb = [0, 1, 0]
        fig4.linewidth = 1.3
        fig4.figurergba = [0, 0, 0, 0.1]
        fig4.linename[0] = 'yaw'
        fig4.flg_ticklabel[0] = 0
        fig5.range[1][0] = -1.01
        fig5.range[1][1] = 1.01
        fig5.linergb = [1, 1, 1]
        fig5.linewidth = 1.3
        fig5.figurergba = [0, 0, 0, 0.1]
        fig5.linename[0] = 'a1'
        fig5.flg_ticklabel[0] = 0
        fig6.range[1][0] = -1.01
        fig6.range[1][1] = 1.01
        fig6.linergb = [1, 1, 1]
        fig6.linewidth = 1.3
        fig6.figurergba = [0, 0, 0, 0.1]
        fig6.linename[0] = 'a2'
        fig6.flg_ticklabel[0] = 0
        fig7.range[1][0] = -1.01
        fig7.range[1][1] = 1.01
        fig7.linergb = [1, 1, 1]
        fig7.linewidth = 1.3
        fig7.figurergba = [0, 0, 0, 0.1]
        fig7.linename[0] = 'a3'
        fig7.flg_ticklabel[0] = 0
        fig8.range[1][0] = -1.01
        fig8.range[1][1] = 1.01
        fig8.linergb = [1, 1, 1]
        fig8.linewidth = 1.3
        fig8.figurergba = [0, 0, 0, 0.1]
        fig8.linename[0] = 'a4'
        fig8.flg_ticklabel[0] = 0

    def _set_figure(self, loc_x, loc_y, ndata=1):
        fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(fig)
        for i in range(0, self.n_timestep):
            for j in range(ndata):
                fig.linedata[j][2 * i] = float(-i)
        fig_viewport = mujoco.MjrRect(loc_x, loc_y, 200, int(self.viewport.height/4))
        mujoco.mjr_figure(fig_viewport, fig, self.ctx)
        fig.flg_extend = 1
        fig.flg_symmetric = 0
        fig.flg_legend = 1
        fig.range[0][1] = 0.01
        fig.gridsize = [2, 5]
        fig.legendrgba = [0, 0, 0, 0]
        fig.figurergba = [0, 0, 0, 0.8]
        fig.panergba = [0, 0, 0, 0.5]
        return fig, fig_viewport

    def _set_words(self, loc_x, loc_y):
        fig = mujoco.MjvFigure()
        fig.flg_legend = False
        fig.flg_ticklabel[0] = False
        fig.flg_ticklabel[1] = False
        fig.flg_extend = True
        fig.flg_barplot = False
        fig.flg_selection = False
        fig.textrgb[0] = 0
        fig.textrgb[1] = 255
        fig.textrgb[2] = 0

        text = []
        for condition in self.rotor_condition:
            if condition == 1:
                text.append('Normal')
            elif condition == 0:
                text.append('Failure')
            else:
                text.append('      ')

        fig.title = 'a1: {} | a2: {} | a3: {} | a4: {}'.format(text[0], text[1], text[2], text[3])
        #mujoco.mjv_defaultFigure(fig)
        fig_viewport = mujoco.MjrRect(loc_x, loc_y, 0, 0)
        mujoco.mjr_figure(fig_viewport, fig, self.ctx)
        return fig, fig_viewport

    def _sensorupdate(self):
        qpos = self.data.qpos
        action = self.data.userdata
        pos, quat = qpos[:3], qpos[3:]
        rpy = quat2rpy(quat)
        sensor_data = [pos, [rpy[0]], [rpy[1]], [rpy[2]], [action[0]], [action[1]], [action[2]], [action[3]]]
        for i in range(len(self.fig) - 1):
            pnt = int(mujoco.mju_min(self.n_timestep, self.fig[i].linepnt[0] + 1))
            n_fig = len(sensor_data[i])
            for j in range(n_fig):
                for k in range(pnt - 1, 0, -1):
                    self.fig[i].linedata[j][2 * k + 1] = self.fig[i].linedata[j][2 * k - 1]
                self.fig[i].linepnt[j] = pnt
                self.fig[i].linedata[j][1] = sensor_data[i][j]


    def _update_graph_size(self):
        for i in range(len(self.fig_viewport)):
            if i < 4:
                self.fig_viewport[i].left = 0
                self.fig_viewport[i].bottom = int((3-i)*self.viewport.height/4)
                self.fig_viewport[i].width = int(0.25*self.viewport.width)
                self.fig_viewport[i].height = int(self.viewport.height/4)+1
            else:
                self.fig_viewport[i].left = self.viewport.width - int(0.25 * self.viewport.width)
                self.fig_viewport[i].bottom = int((3-i+4) * self.viewport.height / 4)
                self.fig_viewport[i].width = int(0.25 * self.viewport.width)
                self.fig_viewport[i].height = int(self.viewport.height / 4) + 1
        self.fig_viewport[8].left = int((self.viewport.width - int(0.25 * self.viewport.width)) / 2) - 135
        self.fig_viewport[8].bottom = int((3) * self.viewport.height / 4) + 70
        self.fig_viewport[8].width = 550
        self.fig_viewport[8].height = 50

    def render(self):
        if not self.is_alive:
            raise Exception(
                "GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window)
            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn)
                # marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # render
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)
                # overlay items
                for gridpos, [t1, t2] in self._overlay.items():
                    menu_positions = [mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      mujoco.mjtGridPos.mjGRID_BOTTOMLEFT]
                    if gridpos in menu_positions and self._hide_menus:
                        continue

                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.ctx)

                if not self._paused:
                    self._sensorupdate()
                    self._update_graph_size()
                    for fig, viewport in zip(self.fig, self.fig_viewport):
                        mujoco.mjr_figure(viewport, fig, self.ctx)

                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + \
                0.1 * (time.time() - render_start)

            # clear overlay
            self._overlay.clear()

        if self._paused:
            while self._paused:
                update()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / \
                (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        self.apply_perturbations()