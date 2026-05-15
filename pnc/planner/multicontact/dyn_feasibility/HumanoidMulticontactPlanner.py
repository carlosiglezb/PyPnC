from copy import copy

import numpy as np
import crocoddyl

from util.path_parameterization import get_frame_des_pos

class ContactSequence:
    def __init__(self, contact_planes_seq: list[dict[str: np.ndarray]],
                 phases_knots: list[int],
                 time_per_phase: float):
        self.contact_planes_seq = contact_planes_seq
        self.phases_knots = phases_knots
        self.phases_durations = time_per_phase

class HumanoidMulticontactPlanner:
    def __init__(self, robot_model,
                 contact_seqs: ContactSequence,
                 ik_cfree_planner,
                 planner_params,
                 geom_model=None):
        self.solver_type = None
        self.geom_model = geom_model
        self.solver_stats = {}
        self.frame_names_lst = ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']
        self.contact_planes_seq = contact_seqs.contact_planes_seq
        self.horizon_lst = contact_seqs.phases_knots
        tot_num_knots = sum(contact_seqs.phases_knots)
        self.lh_targets = np.zeros((tot_num_knots, 3))
        self.rh_targets  = np.zeros((tot_num_knots, 3))
        self.lf_targets = np.zeros((tot_num_knots, 3))
        self.rf_targets = np.zeros((tot_num_knots, 3))
        self.lkn_targets = np.zeros((tot_num_knots, 3))
        self.rkn_targets = np.zeros((tot_num_knots, 3))
        self.base_targets = np.zeros((tot_num_knots, 3))
        self.ee_rpy = {'LH': [0., 0., 0.], 'RH': [0., 0., 0.]}

        self.contact_phases = num_contact_phases = len(contact_seqs.phases_knots)
        self.fddp = [crocoddyl.SolverFDDP] * num_contact_phases   # if solving by sections
        self.fddp_single = crocoddyl.SolverFDDP
        self.fddp_full = crocoddyl.SolverFDDP
        self.fddp_full_sca = crocoddyl.SolverFDDP
        self.T = contact_seqs.phases_durations  # time_per_phase
        self.b_sca_converges = False

        self.planner_params = planner_params
        # TODO set some default values
        self.x0 = None
        self.plan_to_model_ids = None
        self.lleg_jnames = None
        self.rleg_jnames = None
        self.larm_jnames = None
        self.rarm_jnames = None
        self.joint_names_dict = None
        self.ik_cfree_planner = ik_cfree_planner
        self._zero_config = None


        # Crocoddyl variables / parameters
        self.state = crocoddyl.StateMultibody(robot_model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.robot_model = robot_model

        # initialize size of costs to be stored
        self.costs = {'uReg': [None] * num_contact_phases,
                      'xReg': [None] * num_contact_phases,
                      'xBounds': [None] * num_contact_phases,
                      'LF_friction': [None] * num_contact_phases,
                      'RF_friction': [None] * num_contact_phases,
                      'LH_friction': [None] * num_contact_phases,
                      'RH_friction': [None] * num_contact_phases,
                      'torso_goal': [None] * num_contact_phases,
                      'LF_goal': [None] * num_contact_phases,
                      'RF_goal': [None] * num_contact_phases,
                      'LH_goal': [None] * num_contact_phases,
                      'RH_goal': [None] * num_contact_phases,
                      'L_knee_goal': [None] * num_contact_phases,
                      'R_knee_goal': [None] * num_contact_phases}

        self.costs_full = {'uReg': [None] * (tot_num_knots + num_contact_phases - 1),
                      'xReg': [None] * (tot_num_knots + num_contact_phases - 1),
                      'xBounds': [None] * (tot_num_knots + num_contact_phases - 1),
                      'LF_friction': [None] * (tot_num_knots + num_contact_phases - 1),
                      'RF_friction': [None] * (tot_num_knots + num_contact_phases - 1),
                      'LH_friction': [None] * (tot_num_knots + num_contact_phases - 1),
                      'RH_friction': [None] * (tot_num_knots + num_contact_phases - 1),
                      'torso_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'LF_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'RF_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'LH_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'RH_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'L_knee_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'R_knee_goal': [None] * (tot_num_knots + num_contact_phases - 1)}

        self.costs_full_sca = {'uReg': [None] * (tot_num_knots + num_contact_phases - 1),
                      'xReg': [None] * (tot_num_knots + num_contact_phases - 1),
                      'xBounds': [None] * (tot_num_knots + num_contact_phases - 1),
                      'LF_friction': [None] * (tot_num_knots + num_contact_phases - 1),
                      'RF_friction': [None] * (tot_num_knots + num_contact_phases - 1),
                      'LH_friction': [None] * (tot_num_knots + num_contact_phases - 1),
                      'RH_friction': [None] * (tot_num_knots + num_contact_phases - 1),
                      'torso_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'LF_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'RF_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'LH_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'RH_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'L_knee_goal': [None] * (tot_num_knots + num_contact_phases - 1),
                      'R_knee_goal': [None] * (tot_num_knots + num_contact_phases - 1)}

        self.residuals = {
            # 'LF_friction': [None] * (tot_num_knots + num_contact_phases - 1),
            # 'RF_friction': [None] * (tot_num_knots + num_contact_phases - 1),
            # 'LH_friction': [None] * (tot_num_knots + num_contact_phases - 1),
            # 'RH_friction': [None] * (tot_num_knots + num_contact_phases - 1),
        }

        # initialize all costs
        for cost_name, cost_lst in self.costs.items():
            for di, ldata in enumerate(self.horizon_lst):
                cost_lst[di] = np.zeros((ldata,))

        self.knot_idx = 0

    def set_initial_configuration(self, x0):
        self.x0 = x0

    def set_plan_to_model_params(self, plan_to_model_ids):
        self.plan_to_model_ids = plan_to_model_ids

    def pack_current_targets(self, t):
        frame_targets_dict = {}
        for name in ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']:
            if name in self.frame_names_lst:
                idx = self.frame_names_lst.index(name)
                frame_targets_dict[name] = get_frame_des_pos(self.ik_cfree_planner[idx], t)
        return frame_targets_dict

    def get_solver_and_costs(self):
        solver_type = self.solver_type
        if solver_type == 'seq':
            fddp_solver = self.fddp
            costs = self.costs
        elif solver_type == 'single':
            fddp_solver = [self.fddp_single]
            costs = self.costs_full
        elif solver_type == 'full':
            fddp_solver = [self.fddp_full]
            costs = self.costs_full
        elif solver_type == 'sca':
            fddp_solver = [self.fddp_full_sca]
            costs = self.costs_full_sca
        else:
            raise ValueError("Unknown solver type: {}".format(solver_type))
        print("[get_solver_and_costs] Using solver type: {}".format(solver_type))
        return fddp_solver, costs

    def update_costs_from_solver(self, solver_type='seq', integration_type='Euler'):
        fddp_solver, costs = self.get_solver_and_costs()

        # Check that all cost names exist. If not, create them (e.g., for SCA constraints)
        diff_model = fddp_solver[0].problem.runningDatas[0].differential
        if integration_type != 'Euler':
            diff_model = fddp_solver[0].problem.runningDatas[0].differential[0]
        for cost_entry in diff_model.costs.costs:
            if cost_entry.key() not in costs:
                # costs[cost_entry.key()] = [None] * len(fddp_solver[0].problem.runningDatas)
                if solver_type == 'seq':
                    for fddp_idx in range(len(fddp_solver)):
                        if fddp_idx == 0:
                            costs[cost_entry.key()] = [[None] * (len(fddp_solver[fddp_idx].problem.runningDatas) + 1)]
                        else:
                            costs[cost_entry.key()].append([None] * (len(fddp_solver[fddp_idx].problem.runningDatas) + 1))
                        # costs[cost_entry.key()] = [[None] * len(costs[next(iter(costs))][0])] * len(fddp_solver)
                    # costs[cost_entry.key()] = [[None] * len(fddp_solver[0].problem.runningDatas)] * len(fddp_solver)
                else:
                    costs[cost_entry.key()] = [None] * len(fddp_solver[0].problem.runningDatas)

        for fddp_idx, fddp in enumerate(fddp_solver):
            len_datas = fddp.problem.T
            for model_idx in range(len_datas):
                runData = list(fddp.problem.runningDatas)[model_idx]
                runModel = list(fddp.problem.runningModels)[model_idx]
                if hasattr(runData, 'differential'):
                    if integration_type != 'Euler':
                        costs_vec = runData.differential[0].costs.costs
                    else:
                        costs_vec = runData.differential.costs.costs
                    costs_model = runModel.differential.costs.costs
                for cv in costs_vec:
                    if solver_type == 'seq':
                        costs[cv.key()][fddp_idx][model_idx] = costs_model[cv.key()].weight * costs_vec[cv.key()].cost
                    else:
                        costs[cv.key()][model_idx] = costs_model[cv.key()].weight * costs_vec[cv.key()].cost

    def update_constraint_residuals_from_solver(self, solver_type='seq', integration_type='Euler'):
        fddp_solver, _ = self.get_solver_and_costs()

        # Check that all constraint names exist. If not, create them (e.g., for SCA constraints)
        diff_model = fddp_solver[0].problem.runningDatas[0].differential
        if integration_type != 'Euler':
            diff_model = fddp_solver[0].problem.runningDatas[0].differential[0]
        for constraint_entry in diff_model.constraints.constraints:
            if len(diff_model.constraints.constraints[constraint_entry.key()].residual.r) > 1:
                continue  # only store scalar residuals for now
            self.residuals[constraint_entry.key()] = [None] * len(fddp_solver[0].problem.runningDatas)

        # populate with values
        for fddp_idx, fddp in enumerate(fddp_solver):
            len_datas = fddp.problem.T
            for model_idx in range(len_datas):
                runData = list(fddp.problem.runningDatas)[model_idx]
                if hasattr(runData, 'differential'):
                    if integration_type != 'Euler':
                        constraints_vec = runData.differential[0].constraints.constraints
                    else:
                        # note: impulse model won't have constraints
                        if hasattr(runData.differential.constraints, "constraints"):
                            constraints_vec = runData.differential.constraints.constraints
                        else:
                            continue
                else:
                    continue
                for cv in constraints_vec:
                    if cv.key() in constraints_vec:
                        if len(constraints_vec[cv.key()].residual.r) > 1:
                            continue  # only store scalar residuals for now
                        self.residuals[cv.key()][model_idx] = constraints_vec[cv.key()].residual.r[0]
                    else:
                        self.residuals[cv.key()][model_idx] = np.nan