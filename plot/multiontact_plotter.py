import numpy as np

from plot.helper import plot_multiple_state_traj, plot_hold_vector_traj
from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import HumanoidMulticontactPlanner


class MulticontactPlotter:
    def __init__(self, robot_planner: HumanoidMulticontactPlanner):
        self._robot_planner = robot_planner
        self.lleg_joint_ids = self._get_lleg_joint_ids()
        self.rleg_joint_ids = self._get_rleg_joint_ids()
        self.larm_joint_ids = self._get_larm_joint_ids()
        self.rarm_joint_ids = self._get_rarm_joint_ids()

    def plot_reduced_xs_us(self):
        fddp = self._robot_planner.fddp
        lleg_jids = self.lleg_joint_ids
        rleg_jids = self.rleg_joint_ids
        larm_jids = self.larm_joint_ids
        rarm_jids = self.rarm_joint_ids
        horizon_lst = self._robot_planner.horizon_lst
        T = self._robot_planner.T
        xr_dim = len(self.lleg_joint_ids)
        ur_dim = len(self.lleg_joint_ids)
        xrarm_dim = len(self.larm_joint_ids)
        urarm_dim = len(self.larm_joint_ids)

        xs_l_reduced = np.zeros((sum(horizon_lst) - len(fddp) + 1, xr_dim))
        us_l_reduced = np.zeros((sum(horizon_lst) - len(fddp), ur_dim))
        xs_r_reduced = np.zeros((sum(horizon_lst) - len(fddp) + 1, xr_dim))
        us_r_reduced = np.zeros((sum(horizon_lst) - len(fddp), ur_dim))
        xs_larm_reduced = np.zeros((sum(horizon_lst) - len(fddp) + 1, xrarm_dim))
        us_larm_reduced = np.zeros((sum(horizon_lst) - len(fddp), urarm_dim))
        xs_rarm_reduced = np.zeros((sum(horizon_lst) - len(fddp) + 1, xrarm_dim))
        us_rarm_reduced = np.zeros((sum(horizon_lst) - len(fddp), urarm_dim))
        time = np.zeros(sum(horizon_lst) - len(fddp) + 1)
        phase = np.zeros(sum(horizon_lst) - len(fddp) + 1, dtype=int)
        curr_idx = 0
        for (it_num, it) in enumerate(fddp):
            if it_num == (len(fddp) - 1):
                next_idx = curr_idx + horizon_lst[it_num]  # last phase does not have impulse model
                time[curr_idx:next_idx] = np.linspace(it_num * T, (it_num + 1) * T, horizon_lst[it_num])
            else:
                next_idx = curr_idx + horizon_lst[it_num] + 1   # add terminal impulse models
                time[curr_idx:next_idx] = np.linspace(it_num * T, (it_num + 1) * T, horizon_lst[it_num] + 1)

            log = it.getCallbacks()[0]
            lleg_jid_fb = [7 + ji for ji in lleg_jids]
            rleg_jid_fb = [7 + ji for ji in rleg_jids]
            larm_jid_fb = [7 + ji for ji in larm_jids]
            rarm_jid_fb = [7 + ji for ji in rarm_jids]
            xs_l_reduced[curr_idx:next_idx - 1, :] = np.array(log.xs)[:-1, lleg_jid_fb]
            us_l_reduced[curr_idx:next_idx - 1, :] = np.array(log.us)[:, lleg_jids]
            xs_r_reduced[curr_idx:next_idx - 1, :] = np.array(log.xs)[:-1, rleg_jid_fb]
            us_r_reduced[curr_idx:next_idx - 1, :] = np.array(log.us)[:, rleg_jids]
            xs_larm_reduced[curr_idx:next_idx - 1, :] = np.array(log.xs)[:-1, larm_jid_fb]
            us_larm_reduced[curr_idx:next_idx - 1, :] = np.array(log.us)[:, larm_jids]
            xs_rarm_reduced[curr_idx:next_idx - 1, :] = np.array(log.xs)[:-1, rarm_jid_fb]
            us_rarm_reduced[curr_idx:next_idx - 1, :] = np.array(log.us)[:, rarm_jids]
            phase[curr_idx:next_idx] = int(it_num)
            curr_idx += horizon_lst[it_num] - 1

        # create names of signals and plot left/right leg joints
        xs_names = [None] * xr_dim
        us_names = [None] * ur_dim
        for jn_i, jn in enumerate(self._robot_planner.lleg_jnames):
            xs_names[jn_i] = 'q_' + jn
            us_names[jn_i] = 'u_' + jn
        signals_names = [xs_names, us_names]
        plot_multiple_state_traj(time[:-1], [xs_l_reduced[:-1, :], us_l_reduced[:, :]],
                                 phase, ax_labels=signals_names)

        # create names of signals and plot right leg joints
        for jn_i, jn in enumerate(self._robot_planner.rleg_jnames):
            xs_names[jn_i] = 'q_' + jn
            us_names[jn_i] = 'u_' + jn
        signals_names = [xs_names, us_names]
        plot_multiple_state_traj(time[:-1], [xs_r_reduced[:-1, :], us_r_reduced[:, :]],
                                 phase, ax_labels=signals_names)

        # create names of signals and plot left/right arm joints
        xs_arm_names = [None] * xrarm_dim
        us_arm_names = [None] * urarm_dim
        for jn_i, jn in enumerate(self._robot_planner.larm_jnames):
            xs_arm_names[jn_i] = 'q_' + jn
            us_arm_names[jn_i] = 'u_' + jn
        signals_names = [xs_arm_names, us_arm_names]
        plot_multiple_state_traj(time[:-1], [xs_larm_reduced[:-1, :], us_larm_reduced[:, :]],
                                 phase, ax_labels=signals_names)

        # create names of signals and plot right arm joints
        for jn_i, jn in enumerate(self._robot_planner.rarm_jnames):
            xs_arm_names[jn_i] = 'q_' + jn
            us_arm_names[jn_i] = 'u_' + jn
        signals_names = [xs_arm_names, us_arm_names]
        plot_multiple_state_traj(time[:-1], [xs_rarm_reduced[:-1, :], us_rarm_reduced[:, :]],
                                 phase, ax_labels=signals_names)


    def plot_costs(self):
        T = self._robot_planner.T
        horizon_lst = self._robot_planner.horizon_lst
        time = np.zeros((sum(horizon_lst) - 1, ))
        phase = np.zeros((sum(horizon_lst) - 1, ), dtype=int)
        costsDict = self._robot_planner.costs

        # create time vector (same for all costs)
        for contact_phase in range(len(horizon_lst)):
            # get current and next index
            curr_idx = sum(horizon_lst[:contact_phase])
            if contact_phase == (len(horizon_lst) - 1):
                next_idx = curr_idx + horizon_lst[contact_phase] - 1
                time[curr_idx:next_idx] = np.linspace(T * contact_phase, T * (contact_phase + 1),
                                                      horizon_lst[contact_phase] - 1)
            else:
                next_idx = curr_idx + horizon_lst[contact_phase]
                time[curr_idx:next_idx] = np.linspace(T * contact_phase, T * (contact_phase + 1), horizon_lst[contact_phase])
            # time[curr_idx:next_idx] = np.arange(T * contact_phase, T * (contact_phase + 1), T / horizon_lst[contact_phase])
            phase[curr_idx:next_idx] = contact_phase

        # parse all costs
        all_costs = np.zeros((sum(horizon_lst) - 1, len(costsDict.keys())))
        for cost_idx, (cost_name, costs_lst) in enumerate(costsDict.items()):
            for contact_phase in range(len(horizon_lst)):
                # get current and next index and populate current costs vector
                curr_idx = sum(horizon_lst[:contact_phase])
                if contact_phase == (len(horizon_lst) - 1):
                    next_idx = curr_idx + horizon_lst[contact_phase] - 1
                    all_costs[curr_idx:next_idx, cost_idx] = costsDict[cost_name][contact_phase][:-1]
                else:
                    next_idx = curr_idx + horizon_lst[contact_phase]
                    all_costs[curr_idx:next_idx, cost_idx] = costsDict[cost_name][contact_phase]

        plot_hold_vector_traj(time, all_costs, 'Costs', legends=list(costsDict.keys()))

    def _get_lleg_joint_ids(self):
        lleg_j_ids = []
        robot_model = self._robot_planner.robot_model
        for jname in self._robot_planner.lleg_jnames:
            lleg_j_ids.append(list(robot_model.names).index(jname) - 2)
        return lleg_j_ids

    def _get_rleg_joint_ids(self):
        rleg_j_ids = []
        robot_model = self._robot_planner.robot_model
        for jname in self._robot_planner.rleg_jnames:
           rleg_j_ids.append(list(robot_model.names).index(jname) - 2)
        return rleg_j_ids

    def _get_larm_joint_ids(self):
        larm_j_ids = []
        robot_model = self._robot_planner.robot_model
        for jname in self._robot_planner.larm_jnames:
            larm_j_ids.append(list(robot_model.names).index(jname) - 2)
        return larm_j_ids

    def _get_rarm_joint_ids(self):
        rarm_j_ids = []
        robot_model = self._robot_planner.robot_model
        for jname in self._robot_planner.rarm_jnames:
           rarm_j_ids.append(list(robot_model.names).index(jname) - 2)
        return rarm_j_ids
