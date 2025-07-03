import numpy as np
from config.multicontact.planner_config import PlannerConfig

N_V = 32  # dimension of generalized velocities
N_U = 26  # dimension of control inputs

class MultiContactDoorConfig(PlannerConfig):
    W_RIGID_LINK = [500., 0., 50.]  # tested on single step
    # W_RIGID_LINK_SINGLE_STEP = [500., 0., 50.]
    W_RIGID_POLY = [0.1621, 0.0, 0.]        # TODO check if they need to be different from W_RIGID_LINK
    ALPHA = [0.2, 0.2, 1.0]
    FOOT_SIZE = [0.2, 0.11]  # [length, width]
    N_HORIZON_LST = [150, 150, 100]

    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([3.0] * 3 + [0.5, 0.5, 0.01]),    # (lin, ang)
            'feet': np.array([6.] * 3 + [0.00001] * 3),         # (lin, ang)
            'L_knee': np.array([2.] * 3 + [0.00001] * 3),
            'R_knee': np.array([2.] * 3 + [0.00001] * 3),
            'LH': np.array([2.] * 3 + [0.00001] * 3),
            'RH': np.array([2.] * 3 + [0.00001] * 3)
        }
    WBC_WEIGHTED_COSTS = {
        'xReg': np.array([0.1] * 3 + [10.0] * 3 + [2.] * (N_V - 6) + [4.] * N_V),
        'uReg': np.array([0.5] * N_U),
    }


class MultiContactTiltedStairsConfig(PlannerConfig):
    W_RIGID_LINK = [1., 0., 10.]
    W_RIGID_LINK_STEP_ON_DOOR = [1000., 0., 0.]
    ALPHA = [0.2, 0.2, 1.0]
    FOOT_SIZE = [0.2, 0.11]  # [length, width]
    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([3.0] * 3 + [0.5, 0.5, 0.01]),    # (lin, ang)
            'feet': np.array([6.] * 3 + [0.00001] * 3),         # (lin, ang)
            'L_knee': np.array([2.] * 3 + [0.00001] * 3),
            'R_knee': np.array([2.] * 3 + [0.00001] * 3),
            'LH': np.array([2.] * 3 + [0.00001] * 3),
            'RH': np.array([2.] * 3 + [0.00001] * 3)
        }
    WBC_WEIGHTED_COSTS = {
        'xReg': np.array([0.1] * 3 + [10.0] * 3 + [2.] * (N_V - 6) + [4.] * N_V),
        'uReg': np.array([0.5] * N_U),
    }
