import numpy as np
from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import HumanoidMulticontactPlanner


class G1MulticontactPlanner(HumanoidMulticontactPlanner):
    def __init__(self, robot_model, knots_lst, time_per_phase, ik_cfree_planner):
        super().__init__(self, robot_model, knots_lst, time_per_phase, ik_cfree_planner)

        self.gains = {
            'torso': np.array([1.0] * 3 + [0.5] * 3),  # (lin, ang)
            'feet': np.array([8.] * 3 + [0.00001] * 3),  # (lin, ang)
            'L_knee': np.array([3.] * 3 + [0.00001] * 3),
            'R_knee': np.array([3.] * 3 + [0.00001] * 3),
            'hands': np.array([2.] * 3 + [0.00001] * 3)
        }

    def plan(self):
        super().plan()

    def get_terminal_gains(self):
        self.gains['feet'] = np.array([10.] * 3 + [1.5] * 3)
