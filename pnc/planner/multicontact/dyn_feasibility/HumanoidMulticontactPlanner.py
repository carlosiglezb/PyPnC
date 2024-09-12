import numpy as np
import crocoddyl

class HumanoidMulticontactPlanner:
    def __init__(self, robot_model, knots_lst, time_per_phase, ik_cfree_planner):
        tot_num_knots = sum(knots_lst)
        self.horizon_lst = knots_lst
        self.lh_targets = np.zeros((tot_num_knots, 3))
        self.rh_targets  = np.zeros((tot_num_knots, 3))
        self.lf_targets = np.zeros((tot_num_knots, 3))
        self.rf_targets = np.zeros((tot_num_knots, 3))
        self.lkn_targets = np.zeros((tot_num_knots, 3))
        self.rkn_targets = np.zeros((tot_num_knots, 3))
        self.base_targets = np.zeros((tot_num_knots, 3))
        self.ee_rpy = {'LH': [0., 0., 0.], 'RH': [0., 0., 0.]}

        self.contact_phases = num_contact_phases = len(knots_lst)
        self.fddp = [crocoddyl.SolverFDDP] * num_contact_phases
        self.T = time_per_phase

        # TODO set some default values
        self.gains = None
        self.x0 = None
        self.plan_to_model_frames = None
        self.plan_to_model_ids = None
        self.ik_cfree_planner = ik_cfree_planner

        # Crocoddyl variables / parameters
        self.state = crocoddyl.StateMultibody(robot_model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

        self.knot_idx = 0

    def set_initial_configuration(self, x0):
        self.x0 = x0

    def set_plan_to_model_params(self, plan_to_model_frames, plan_to_model_ids):
        self.plan_to_model_frames = plan_to_model_frames
        self.plan_to_model_ids = plan_to_model_ids
