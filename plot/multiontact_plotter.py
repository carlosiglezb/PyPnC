import numpy as np

from plot.helper import plot_multiple_state_traj
from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import HumanoidMulticontactPlanner


class MulticontactPlotter:
    def __init__(self, robot_planner: HumanoidMulticontactPlanner):
        self._robot_planner = robot_planner
        self.lleg_joint_ids = self._get_lleg_joint_ids()
        self.rleg_joint_ids = self._get_rleg_joint_ids()

    def plot_reduced_xs_us(self):
        fddp = self._robot_planner.fddp
        lleg_jids = self.lleg_joint_ids
        rleg_jids = self.rleg_joint_ids
        horizon_lst = self._robot_planner.horizon_lst
        T = self._robot_planner.T
        xr_dim = len(self.lleg_joint_ids)
        ur_dim = len(self.lleg_joint_ids)

        xs_l_reduced = np.zeros((sum(horizon_lst) - 1, xr_dim))
        us_l_reduced = np.zeros((sum(horizon_lst) - 1, ur_dim))
        xs_r_reduced = np.zeros((sum(horizon_lst) - 1, xr_dim))
        us_r_reduced = np.zeros((sum(horizon_lst) - 1, ur_dim))
        time = np.zeros(sum(horizon_lst) - 1)
        phase = np.zeros(sum(horizon_lst) - 1, dtype=int)
        curr_idx = 0
        for (it_num, it) in enumerate(fddp):
            next_idx = curr_idx + horizon_lst[it_num] + 1   # add terminal impulse models

            log = it.getCallbacks()[0]
            xs_l_reduced[curr_idx:next_idx, :] = np.array(log.xs)[:, lleg_jids]
            us_l_reduced[curr_idx:next_idx - 1, :] = np.array(log.us)[:, lleg_jids]
            xs_r_reduced[curr_idx:next_idx, :] = np.array(log.xs)[:, rleg_jids]
            us_r_reduced[curr_idx:next_idx - 1, :] = np.array(log.us)[:, rleg_jids]
            time[curr_idx:next_idx] = np.linspace(it_num * T, (it_num+1) * T, horizon_lst[it_num] + 1)
            phase[curr_idx:next_idx] = int(it_num)
            curr_idx += horizon_lst[it_num] - 1

        # create names of signals and plot left leg joints
        xs_names = [None] * xr_dim
        us_names = [None] * ur_dim
        for jn_i, jn in enumerate(self._robot_planner.lleg_jnames):
            xs_names[jn_i] = 'q_' + jn
            us_names[jn_i] = 'u_' + jn
        signals_names = [xs_names, us_names]
        plot_multiple_state_traj(time[:-1], [xs_l_reduced[:-1, :], us_l_reduced[:-1, :]],
                                 phase, ax_labels=signals_names)

        # create names of signals and plot right leg joints
        for jn_i, jn in enumerate(self._robot_planner.rleg_jnames):
            xs_names[jn_i] = 'q_' + jn
            us_names[jn_i] = 'u_' + jn
        signals_names = [xs_names, us_names]
        plot_multiple_state_traj(time[:-1], [xs_r_reduced[:-1, :], us_r_reduced[:-1, :]],
                                 phase, ax_labels=signals_names)

    def _get_lleg_joint_ids(self):
        lleg_j_ids = []
        robot_model = self._robot_planner.robot_model
        for jname in self._robot_planner.lleg_jnames:
            lleg_j_ids.append(list(robot_model.names).index(jname) - 2 + 7)
        return lleg_j_ids

    def _get_rleg_joint_ids(self):
        rleg_j_ids = []
        robot_model = self._robot_planner.robot_model
        for jname in self._robot_planner.rleg_jnames:
           rleg_j_ids.append(list(robot_model.names).index(jname) - 2 + 7)
        return rleg_j_ids
