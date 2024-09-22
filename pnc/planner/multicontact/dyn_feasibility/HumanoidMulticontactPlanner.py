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

        self.knot_idx = 0

    def set_initial_configuration(self, x0):
        self.x0 = x0

    def set_plan_to_model_params(self, plan_to_model_frames, plan_to_model_ids):
        self.plan_to_model_frames = plan_to_model_frames
        self.plan_to_model_ids = plan_to_model_ids
