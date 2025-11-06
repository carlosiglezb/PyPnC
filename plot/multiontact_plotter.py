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
        self.solver_type = robot_planner.solver_type

    def plot_reduced_xs_us(self):
        xr_dim = len(self.lleg_joint_ids)
        ur_dim = len(self.lleg_joint_ids)
        xrarm_dim = len(self.larm_joint_ids)
        urarm_dim = len(self.larm_joint_ids)

        if self.solver_type == 'seq':
            phase, time, us_l_reduced, us_larm_reduced, us_r_reduced, us_rarm_reduced, xs_l_reduced, xs_larm_reduced, xs_r_reduced, xs_rarm_reduced = self.get_seq_to_trajectories()
        else:
            phase, time, us_l_reduced, us_larm_reduced, us_r_reduced, us_rarm_reduced, xs_l_reduced, xs_larm_reduced, xs_r_reduced, xs_rarm_reduced = self.get_full_to_trajectories()

        # create names of signals and plot left/right leg joints
        xs_names = [None] * xr_dim
        us_names = [None] * ur_dim
        control_lenth = us_l_reduced.shape[0]
        for jn_i, jn in enumerate(self._robot_planner.lleg_jnames):
            xs_names[jn_i] = 'q_' + jn
            us_names[jn_i] = 'u_' + jn
        signals_names = [xs_names, us_names]
        plot_multiple_state_traj(time[:control_lenth], [xs_l_reduced[:control_lenth, :], us_l_reduced[:control_lenth, :]],
                                 phase, ax_labels=signals_names)

        # create names of signals and plot right leg joints
        for jn_i, jn in enumerate(self._robot_planner.rleg_jnames):
            xs_names[jn_i] = 'q_' + jn
            us_names[jn_i] = 'u_' + jn
        signals_names = [xs_names, us_names]
        plot_multiple_state_traj(time[:control_lenth], [xs_r_reduced[:control_lenth, :],
                    us_r_reduced[:control_lenth, :]], phase, ax_labels=signals_names)

        # create names of signals and plot left/right arm joints
        xs_arm_names = [None] * xrarm_dim
        us_arm_names = [None] * urarm_dim
        for jn_i, jn in enumerate(self._robot_planner.larm_jnames):
            xs_arm_names[jn_i] = 'q_' + jn
            us_arm_names[jn_i] = 'u_' + jn
        signals_names = [xs_arm_names, us_arm_names]
        plot_multiple_state_traj(time[:control_lenth], [xs_larm_reduced[:control_lenth, :],
                    us_larm_reduced[:control_lenth, :]], phase, ax_labels=signals_names)

        # create names of signals and plot right arm joints
        for jn_i, jn in enumerate(self._robot_planner.rarm_jnames):
            xs_arm_names[jn_i] = 'q_' + jn
            us_arm_names[jn_i] = 'u_' + jn
        signals_names = [xs_arm_names, us_arm_names]
        plot_multiple_state_traj(time[:control_lenth], [xs_rarm_reduced[:control_lenth, :],
                    us_rarm_reduced[:control_lenth, :]], phase, ax_labels=signals_names)

    def get_seq_to_trajectories(self):
        fddp, _ = self._robot_planner.get_solver_and_costs()
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

        # initialize dimensions by contact phase
        xs_l_reduced = np.zeros((sum(horizon_lst), xr_dim))
        us_l_reduced = np.zeros((sum(horizon_lst), ur_dim))
        xs_r_reduced = np.zeros((sum(horizon_lst), xr_dim))
        us_r_reduced = np.zeros((sum(horizon_lst), ur_dim))
        xs_larm_reduced = np.zeros((sum(horizon_lst), xrarm_dim))
        us_larm_reduced = np.zeros((sum(horizon_lst), urarm_dim))
        xs_rarm_reduced = np.zeros((sum(horizon_lst), xrarm_dim))
        us_rarm_reduced = np.zeros((sum(horizon_lst), urarm_dim))
        time = np.zeros(sum(horizon_lst))
        phase = np.zeros(sum(horizon_lst), dtype=int)
        curr_idx = 0
        for (it_num, it) in enumerate(fddp):
            # if it_num == (len(fddp) - 1):
            #     next_idx = curr_idx + horizon_lst[it_num] - 1  # last phase does not have impulse model
            #     time[curr_idx:next_idx] = np.linspace(it_num * T, (it_num + 1) * T, horizon_lst[it_num] - 1)
            # else:
            next_idx = curr_idx + horizon_lst[it_num]   # add terminal models
            time[curr_idx:next_idx] = np.linspace(it_num * T, (it_num + 1) * T, horizon_lst[it_num])

            log = it.getCallbacks()[0]
            lleg_jid_fb = [7 + ji for ji in lleg_jids]
            rleg_jid_fb = [7 + ji for ji in rleg_jids]
            larm_jid_fb = [7 + ji for ji in larm_jids]
            rarm_jid_fb = [7 + ji for ji in rarm_jids]
            xs_l_reduced[curr_idx:next_idx, :] = np.array(log.xs)[:, lleg_jid_fb]
            us_l_reduced[curr_idx:next_idx - 1, :] = np.array(log.us)[:, lleg_jids]
            xs_r_reduced[curr_idx:next_idx, :] = np.array(log.xs)[:, rleg_jid_fb]
            us_r_reduced[curr_idx:next_idx - 1, :] = np.array(log.us)[:, rleg_jids]
            xs_larm_reduced[curr_idx:next_idx, :] = np.array(log.xs)[:, larm_jid_fb]
            us_larm_reduced[curr_idx:next_idx - 1 , :] = np.array(log.us)[:, larm_jids]
            xs_rarm_reduced[curr_idx:next_idx, :] = np.array(log.xs)[:, rarm_jid_fb]
            us_rarm_reduced[curr_idx:next_idx - 1 , :] = np.array(log.us)[:, rarm_jids]
            phase[curr_idx:next_idx] = int(it_num)
            curr_idx += horizon_lst[it_num]
        return phase, time, us_l_reduced, us_larm_reduced, us_r_reduced, us_rarm_reduced, xs_l_reduced, xs_larm_reduced, xs_r_reduced, xs_rarm_reduced

    def get_full_to_trajectories(self):
        fddp, _ = self._robot_planner.get_solver_and_costs()
        # get xs and us from their logs or solver
        if hasattr(fddp[0], "__class__") and fddp[0].__class__.__name__ in {"SolverSQP", "SolverCSQP"}:
            xs =  fddp[0].xs
            us = fddp[0].us
        else:
            log = fddp[0].getCallbacks()[0]
            xs = log.xs
            us = log.us
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

        lleg_jid_fb = [7 + ji for ji in lleg_jids]
        rleg_jid_fb = [7 + ji for ji in rleg_jids]
        larm_jid_fb = [7 + ji for ji in larm_jids]
        rarm_jid_fb = [7 + ji for ji in rarm_jids]

        # initialize dimensions by contact phase
        xs_l_reduced = np.zeros((sum(horizon_lst) - len(horizon_lst), xr_dim))
        us_l_reduced = np.zeros((sum(horizon_lst) - len(horizon_lst), ur_dim))
        xs_r_reduced = np.zeros((sum(horizon_lst) - len(horizon_lst), xr_dim))
        us_r_reduced = np.zeros((sum(horizon_lst) - len(horizon_lst), ur_dim))
        xs_larm_reduced = np.zeros((sum(horizon_lst) - len(horizon_lst), xrarm_dim))
        us_larm_reduced = np.zeros((sum(horizon_lst) - len(horizon_lst), urarm_dim))
        xs_rarm_reduced = np.zeros((sum(horizon_lst) - len(horizon_lst), xrarm_dim))
        us_rarm_reduced = np.zeros((sum(horizon_lst) - len(horizon_lst), urarm_dim))
        time = np.zeros(sum(horizon_lst) - len(horizon_lst))
        phase = np.zeros(sum(horizon_lst) - len(horizon_lst), dtype=int)
        curr_idx, curr_cont_idx = 0, 0
        for it_num in range(len(horizon_lst)):
            next_cont_idx = curr_cont_idx + horizon_lst[it_num] - 1 # next index on continuous variables
            dt = fddp[0].problem.runningModels[curr_idx].dt
            time[curr_cont_idx:next_cont_idx] = np.arange(it_num * T, (it_num + 1) * T, dt)
            if it_num == (len(horizon_lst) - 1):
                next_idx = curr_idx + horizon_lst[it_num] - 1  # last phase does not have impulse model
            else:
                next_idx = curr_idx + horizon_lst[it_num] - 1# without impulse model

            xs_l_reduced[curr_cont_idx:next_cont_idx, :] = np.array(xs[curr_idx:next_idx])[:, lleg_jid_fb]
            us_l_reduced[curr_cont_idx:next_cont_idx, :] = np.array(us[curr_idx:next_idx])[:, lleg_jids]
            xs_r_reduced[curr_cont_idx:next_cont_idx, :] = np.array(xs[curr_idx:next_idx])[:, rleg_jid_fb]
            us_r_reduced[curr_cont_idx:next_cont_idx, :] = np.array(us[curr_idx:next_idx])[:, rleg_jids]
            xs_larm_reduced[curr_cont_idx:next_cont_idx, :] = np.array(xs[curr_idx:next_idx])[:, larm_jid_fb]
            us_larm_reduced[curr_cont_idx:next_cont_idx, :] = np.array(us[curr_idx:next_idx])[:, larm_jids]
            xs_rarm_reduced[curr_cont_idx:next_cont_idx, :] = np.array(xs[curr_idx:next_idx])[:, rarm_jid_fb]
            us_rarm_reduced[curr_cont_idx:next_cont_idx, :] = np.array(us[curr_idx:next_idx])[:, rarm_jids]
            phase[curr_idx:next_idx] = int(it_num)
            curr_idx += horizon_lst[it_num] + 1
            curr_cont_idx += horizon_lst[it_num] - 1
        return phase, time, us_l_reduced, us_larm_reduced, us_r_reduced, us_rarm_reduced, xs_l_reduced, xs_larm_reduced, xs_r_reduced, xs_rarm_reduced

    def plot_joint_limit_margins(self, to_type=None, integration_type='Euler'):
        njoints = self._robot_planner.robot_model.nv - 6
        rob_nq = self._robot_planner.robot_model.nq

        # if no solver type specified, use the last one computed by planner
        if to_type is None:
            to_type = self._robot_planner.solver_type

        b_update_xs = False
        if to_type == 'seq':
            fddp = self._robot_planner.fddp
            # xs and us need to be updated from each fddp instance
            b_update_xs = True
        elif to_type == 'full':
            fddp = [self._robot_planner.fddp_full]
            log = fddp[0].getCallbacks()[0]
            xs = log.xs
            us = log.us
        elif to_type == 'sca':
            fddp = [self._robot_planner.fddp_full_sca]
            # get xs and us from their logs or solver
            if hasattr(fddp[0], "__class__") and fddp[0].__class__.__name__ in {"SolverSQP", "SolverCSQP"}:
                xs = fddp[0].xs
                us = fddp[0].us
            else:
                log = fddp[0].getCallbacks()[0]
                xs = log.xs
                us = log.us
        else:
            raise ValueError("[Multi-contact Plotter] Unknown solver type: {}".format(to_type))

        horizon_lst = self._robot_planner.horizon_lst
        tot_knots = sum(horizon_lst) - len(horizon_lst)
        time = np.zeros(tot_knots)
        T = self._robot_planner.T
        phase = np.zeros(tot_knots, dtype=int)

        # variables to plot
        joints_pos = np.zeros((tot_knots, njoints))
        joints_vel = np.zeros((tot_knots, njoints))
        joints_tau = np.zeros((tot_knots, njoints))

        # get joint pos, vel, and tau limits
        jp_llim = self._robot_planner.robot_model.lowerPositionLimit[7:]
        jp_ulim = self._robot_planner.robot_model.upperPositionLimit[7:]
        jv_lim = self._robot_planner.robot_model.velocityLimit[6:]
        jtau_lim = self._robot_planner.robot_model.effortLimit[6:]

        # get actual pos, vel, and tau, and store margins
        curr_idx, curr_cont_idx = 0, 0
        for it_num in range(len(horizon_lst)):
            if to_type == 'seq':
                curr_dt = fddp[it_num].problem.runningModels[curr_idx].dt
            else:
                curr_dt = fddp[0].problem.runningModels[curr_idx].dt
            next_cont_idx = curr_cont_idx + horizon_lst[it_num] - 1
            next_idx = curr_idx + horizon_lst[it_num] - 1
            time[curr_cont_idx:next_cont_idx] = np.arange(it_num * T,  (it_num + 1) * T, curr_dt)

            # update xs and us from each fddp instance
            if b_update_xs:
                log = fddp[it_num].getCallbacks()[0]
                xs = log.xs
                us = log.us

            joints_pos[curr_cont_idx:next_cont_idx, :] = np.array(xs[curr_idx:next_idx])[:, 7:rob_nq]  # ignore floating base pos
            joints_vel[curr_cont_idx:next_cont_idx, :] = np.array(xs[curr_idx:next_idx])[:, rob_nq+6:] # ignore floating base vel
            joints_tau[curr_cont_idx:next_cont_idx, :] = np.array(us[curr_idx:next_idx])[:, :njoints]  # no floating base tau
            phase[curr_cont_idx:next_cont_idx] = int(it_num)

            # Compute margins: positive if within bounds, negative if out of bounds
            for k in range(curr_cont_idx, next_cont_idx):
                for j in range(njoints):
                    # joint position margins
                    pos = joints_pos[k, j]
                    lower_pos_margin = pos - jp_llim[j]
                    upper_pos_margin = jp_ulim[j] - pos
                    if lower_pos_margin < 0:
                        joints_pos[k, j] = lower_pos_margin  # negative: below lower bound
                    elif upper_pos_margin < 0:
                        joints_pos[k, j] = -upper_pos_margin  # negative: above upper bound
                    else:
                        joints_pos[k, j] = min(lower_pos_margin, upper_pos_margin)  # positive: within bounds

                    # joint velocity margins
                    vel = joints_vel[k, j]
                    vel_margin = jv_lim[j] - np.abs(vel)
                    joints_vel[k, j] = vel_margin   # neg = out of bound, pos = within bounds

                    # joint torque margins
                    tau = joints_tau[k, j]
                    tau_margin = jtau_lim[j] - np.abs(tau)
                    joints_tau[k, j] = tau_margin  # neg = out of bound, pos = within bounds

            if b_update_xs:
                curr_idx = 0
            else:
                curr_idx += horizon_lst[it_num] + 1
            curr_cont_idx += horizon_lst[it_num] - 1

        # crate margin plots
        jp_names = [None] * njoints
        jv_names = [None] * njoints
        jtau_names = [None] * njoints
        jnames = [None] * njoints
        for j in range(njoints):
            jp_names[j] = 'q_' + self._robot_planner.robot_model.names[j + 2]
            jv_names[j] = 'v_' + self._robot_planner.robot_model.names[j + 2]
            jtau_names[j] = 'tau_' + self._robot_planner.robot_model.names[j + 2]
            jnames[j] = self._robot_planner.robot_model.names[j + 2]

        # plot joint position margins
        signals_names = [jp_names, jv_names, jtau_names]
        # signals_names = [jnames, jnames, jnames]
        margins_names = ['Joint Pos Margin [rad]', 'Joint Vel Margin[rad/s]', 'Joint Tau Margin [Nm]']
        # plot_multiple_state_traj(time, [joints_pos, joints_vel, joints_tau],
        #                          phase, ylabels=margins_names)
        plot_multiple_state_traj(time, [joints_pos, joints_vel, joints_tau],
                                 phase, ax_labels=signals_names, ylabels=margins_names)

    def plot_costs(self, costs_type=None, costNames=None):
        T = self._robot_planner.T
        horizon_lst = self._robot_planner.horizon_lst
        time = np.zeros((sum(horizon_lst) - 1, ))
        phase = np.zeros((sum(horizon_lst) - 1, ), dtype=int)
        if costs_type is None:
            costs_type = self._robot_planner.solver_type
        if costs_type == 'seq':
            costsDict = self._robot_planner.costs
        elif costs_type == 'full':
            costsDict = self._robot_planner.costs_full
        elif costs_type == 'sca':
            costsDict = self._robot_planner.costs_full_sca
        else:
            raise ValueError("Unknown costs type: {}".format(costs_type))

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

        # parse and plot all costs
        all_costs = self.parse_costs(costsDict, costs_type, horizon_lst)
        plot_hold_vector_traj(time, all_costs, 'Costs', legends=list(costsDict.keys()))

        # plot only specific costs of interest
        if costNames is not None:
            target_costs = np.zeros((sum(horizon_lst) - 1, len(costNames)))
            for t_idx, cn in enumerate(costNames):
                if cn in costsDict.keys():
                    idx = list(costsDict.keys()).index(cn)
                    target_costs[:, t_idx] = all_costs[:, idx]

            plot_hold_vector_traj(time, target_costs, 'Specified Costs', legends=costNames)

    def parse_costs(self, costsDict, costs_type, horizon_lst):
        all_costs = np.zeros((sum(horizon_lst) - 1, len(costsDict.keys())))
        for cost_idx, (cost_name, costs_lst) in enumerate(costsDict.items()):
            for contact_phase in range(len(horizon_lst)):
                # get current and next index and populate current costs vector
                if costs_type == 'seq':
                    curr_idx = sum(horizon_lst[:contact_phase])
                else:
                    curr_idx = sum(horizon_lst[:contact_phase])
                    curr_full_idx = sum(horizon_lst[:contact_phase]) + contact_phase
                    next_full_idx = curr_full_idx + horizon_lst[contact_phase]

                if contact_phase == (len(horizon_lst) - 1):
                    # last phase does not have impulse model, so it's the same length on both cases
                    next_idx = curr_idx + horizon_lst[contact_phase] - 1
                    if costs_type == 'seq':
                        all_costs[curr_idx:next_idx, cost_idx] = costsDict[cost_name][contact_phase][:-1]
                    else:
                        all_costs[curr_idx:next_idx, cost_idx] = costsDict[cost_name][curr_full_idx:next_full_idx]
                else:
                    # next_idx = curr_idx + horizon_lst[contact_phase]
                    next_idx = sum(horizon_lst[:contact_phase + 1])
                    if costs_type == 'seq':
                        all_costs[curr_idx:next_idx, cost_idx] = costsDict[cost_name][contact_phase]
                    else:
                        all_costs[curr_idx:next_idx, cost_idx] = costsDict[cost_name][curr_full_idx:next_full_idx]
        return all_costs

    def plot_constraint_violations(self, constraintNames=None):
        T = self._robot_planner.T
        horizon_lst = self._robot_planner.horizon_lst
        time = np.zeros((sum(horizon_lst) - 1, ))
        phase = np.zeros((sum(horizon_lst) - 1, ), dtype=int)
        residuals = self._robot_planner.residuals

        # create time vector (same for all constraints)
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
            phase[curr_idx:next_idx] = contact_phase

        # parse and plot all constraints
        all_residuals = self.parse_costs(residuals, 'sca', horizon_lst)
        plot_hold_vector_traj(time, all_residuals, 'Constraints Residuals', legends=list(residuals.keys()))

        # plot only specific constraints of interest
        if constraintNames is not None:
            target_constraints = np.zeros((sum(horizon_lst) - 1, len(constraintNames)))
            for t_idx, cn in enumerate(constraintNames):
                if cn in residuals.keys():
                    idx = list(residuals.keys()).index(cn)
                    target_constraints[:, t_idx] = all_residuals[:, idx]

            plot_hold_vector_traj(time, target_constraints, 'Specified Constraints Violations', legends=constraintNames)

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
