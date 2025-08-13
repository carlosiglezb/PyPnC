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
                 planner_params):
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
        self.fddp = [crocoddyl.SolverFDDP] * num_contact_phases
        self.T = contact_seqs.phases_durations  # time_per_phase

        self.planner_params = planner_params
        # TODO set some default values
        self.x0 = None
        self.plan_to_model_ids = None
        self.lleg_jnames = None
        self.rleg_jnames = None
        self.larm_jnames = None
        self.rarm_jnames = None
        self.ik_cfree_planner = ik_cfree_planner
        self.gains = planner_params.WBC_FRAME_TRACKING_GAINS    # TODO remove?
        self._default_gains = copy(self.gains)                  # TODO remove?
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
        idx_LF = self.frame_names_lst.index('LF')
        idx_L_knee = self.frame_names_lst.index('L_knee')
        idx_RF = self.frame_names_lst.index('RF')
        idx_R_knee = self.frame_names_lst.index('R_knee')
        idx_LH = self.frame_names_lst.index('LH')
        idx_RH = self.frame_names_lst.index('RH')
        idx_torso = self.frame_names_lst.index('torso')
        lfoot_t = get_frame_des_pos(self.ik_cfree_planner[idx_LF], t)
        lknee_t = get_frame_des_pos(self.ik_cfree_planner[idx_L_knee], t)
        rfoot_t = get_frame_des_pos(self.ik_cfree_planner[idx_RF], t)
        rknee_t = get_frame_des_pos(self.ik_cfree_planner[idx_R_knee], t)
        lhand_t = get_frame_des_pos(self.ik_cfree_planner[idx_LH], t)
        rhand_t = get_frame_des_pos(self.ik_cfree_planner[idx_RH], t)
        base_t = get_frame_des_pos(self.ik_cfree_planner[idx_torso], t)
        frame_targets_dict = {
            'torso': base_t,
            'LF': lfoot_t,
            'RF': rfoot_t,
            'L_knee': lknee_t,
            'R_knee': rknee_t,
            'LH': lhand_t,
            'RH': rhand_t
        }
        return frame_targets_dict

    def update_costs_from_solver(self):
        for fddp_idx, fddp in enumerate(self.fddp):
            len_datas = fddp.problem.T
            for model_idx in range(len_datas):
                costs_vec = list(fddp.problem.runningDatas)[model_idx].differential.costs.costs
                for cv in costs_vec:
                    self.costs[cv.key()][fddp_idx][model_idx] = costs_vec[cv.key()].cost