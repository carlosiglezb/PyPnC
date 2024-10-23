import numpy as np
import crocoddyl


class ContactSequence:
    def __init__(self, contact_phases, phases_knots, time_per_phase):
        self.contact_phases = contact_phases
        self.phases_knots = phases_knots
        self.phases_durations = time_per_phase

class HumanoidMulticontactPlanner:
    def __init__(self, robot_model,
                 contact_seqs: ContactSequence,
                 time_per_phase: float,
                 ik_cfree_planner):
        self.frame_names_lst = ['torso', 'LH', 'RH', 'LF', 'RF', 'L_knee', 'R_knee']
        self.contact_seqs = contact_seqs.contact_phases
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
        self.fddp = [crocoddyl.SolverFDDP] * num_contact_phases
        self.T = time_per_phase

        # TODO set some default values
        self.gains = None
        self.x0 = None
        self.plan_to_model_frames = None
        self.plan_to_model_ids = None
        self.lleg_jnames = None
        self.rleg_jnames = None
        self.ik_cfree_planner = ik_cfree_planner

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

        # initialize all costs
        for cost_name, cost_lst in self.costs.items():
            for di, ldata in enumerate(self.horizon_lst):
                cost_lst[di] = np.zeros((ldata,))

        self.knot_idx = 0

    def set_initial_configuration(self, x0):
        self.x0 = x0

    def set_plan_to_model_params(self, plan_to_model_frames, plan_to_model_ids):
        self.plan_to_model_frames = plan_to_model_frames
        self.plan_to_model_ids = plan_to_model_ids

    def update_costs_from_solver(self):
        for fddp_idx, fddp in enumerate(self.fddp):
            len_datas = self.horizon_lst[fddp_idx]
            if fddp_idx == (len(self.fddp) - 1):
                len_datas -= 1
            for model_idx in range(len_datas):
                costs_vec = list(fddp.problem.runningDatas)[model_idx].differential.costs.costs
                self.costs['uReg'][fddp_idx][model_idx] = costs_vec['uReg'].cost
                self.costs['xReg'][fddp_idx][model_idx] = costs_vec['xReg'].cost
                self.costs['xBounds'][fddp_idx][model_idx] = costs_vec['xBounds'].cost

                for fr in self.frame_names_lst:
                    if fr + '_friction' in costs_vec:
                        self.costs[fr + '_friction'][fddp_idx][model_idx] = costs_vec[fr + '_friction'].cost
                    if fr + '_goal' in costs_vec:
                        self.costs[fr + '_goal'][fddp_idx][model_idx] = costs_vec[fr + '_goal'].cost
