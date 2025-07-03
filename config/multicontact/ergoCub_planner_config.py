import numpy as np
from config.multicontact.planner_config import PlannerConfig

N_V = 39  # dimension of generalized velocities
N_U = 33  # dimension of control inputs

class MultiContactDoorConfig(PlannerConfig):
    W_RIGID_LINK = [1.2, 0., 4.0]   # used to step on door

    ALPHA = [0.2, 0.2, 1.0]
    FOOT_SIZE = [0.15, 0.09]  # [length, width]
    N_HORIZON_LST = [100, 250, 250, 200, 150]   # tested on contact_seq in contact_seq [0, 1]
    # N_HORIZON_LST = [100, 250, 250, 200, 200]   # tested on contact_seq in contact_seq [2]

    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([2.5, 3.5, 1.5] + [1.0, 3.0, 0.001]),  # (lin, ang)
            'feet': np.array([10.] * 3 + [0.01, 1.0, 0.01]),  # (lin, ang)
            'L_knee': np.array([8.] * 3 + [0.00001] * 3),
            'R_knee': np.array([6.] * 3 + [0.00001] * 3),
            'LH': np.array([2.] * 3 + [0.00001] * 3),
            'RH': np.array([2] * 3 + [0.00001] * 3)
        }
    WBC_WEIGHTED_COSTS = {
        'xReg': np.array([0.1] * 3 + [10.0] * 3 + [2.] * (N_V - 6) + [4.] * N_V),
        'uReg': np.array([0.5] * N_U),
    }

class MultiContactTiltedStairsConfig(PlannerConfig):
    W_RIGID_LINK = [1., 0., 10.]

    ALPHA = [0.1, 0.2, 0.8]
    FOOT_SIZE = [0.15, 0.09]  # [length, width]
    N_HORIZON_LST = [100, 250, 250, 250, 280, 250]

    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([2.5, 3.5, 1.5] + [1.0, 3.0, 0.001]),  # (lin, ang)
            'feet': np.array([10.] * 3 + [0.01, 1.0, 0.01]),  # (lin, ang)
            'L_knee': np.array([8.] * 3 + [0.00001] * 3),
            'R_knee': np.array([6.] * 3 + [0.00001] * 3),
            'LH': np.array([2.] * 3 + [0.00001] * 3),
            'RH': np.array([2] * 3 + [0.00001] * 3)
        }
    WBC_WEIGHTED_COSTS = {
        'xReg': np.array([0.1] * 3 + [10.0] * 3 + [2.] * (N_V - 6) + [4.] * N_V),
        'uReg': np.array([0.5] * N_U),
    }
