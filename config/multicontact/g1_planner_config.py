import numpy as np
from config.multicontact.planner_config import PlannerConfig

N_V = 33  # dimension of generalized velocities
N_U = 27  # dimension of control inputs

class MultiContactDoorConfig(PlannerConfig):
    W_RIGID_LINK = [10., 0., 0.]
    # W_RIGID_LINK_SINGLE_STEP = [5., 0., 0.]
    # W_RIGID_LINK_STEP_ON_DOOR = [1000., 0., 0.]

    ALPHA = [1, 0, 0.1]    # seq 0: single hand, step through door
    # ALPHA = [0.5, 0.1, 0.01]
    FOOT_SIZE = [0.15, 0.08]  # [length, width]
    N_HORIZON_LST = [180, 200, 220, 200, 200]

    # Note: contact seq 1 tested with:
    # 'torso', np.array([2.5, 3.5, 1.5] + [0.5, 0.5, 0.001])
    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([3, 3, 1.0] + [0.1, 0.1, 0.01]),  # (lin, ang)
            'feet': np.array([8.] * 3 + [0.00001] * 3),  # (lin, ang)
            'L_knee': np.array([8.] * 3 + [0.00001] * 3),
            'R_knee': np.array([10.] * 3 + [0.00001] * 3),
            'LH': np.array([4.] * 3 + [0.00001] * 3),
            'RH': np.array([4.] * 3 + [0.00001] * 3)
        }
    WBC_WEIGHTED_COSTS = {
        'xReg': np.array([0.1] * 3 + [5.0] * 3 + [2.] * (N_V - 6) + [0.2] * 6 + [4.] * (N_V - 6)),
        'uReg': np.array([0.5] * N_U),
    }

class MultiContactTiltedStairsConfig(PlannerConfig):
    W_RIGID_LINK = [1., 0., 10.]
    ALPHA = [0.1, 0.2, 0.8]
    FOOT_SIZE = [0.15, 0.08]  # [length, width]
    N_HORIZON_LST = [180, 250, 250, 250, 280, 250]

    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([2.0, 1.5, 1.0, 0.5, 0.5, 0.1]),
            'feet': np.array([12.0] * 3 + [0.05, 0.00001, 0.00001]),
            'L_knee': np.array([2.0, 2.5, 3.0] + [0.0001] * 3),
            'R_knee': np.array([2.0, 2.5, 3.0] + [0.0001] * 3),
            'LH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
            'RH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
        }
    WBC_WEIGHTED_COSTS = {
        'xReg': np.array([0.1] * 3 + [10.0] * 3 + [2.] * (N_V - 6) + [4.] * N_V),
        'uReg': np.array([0.5] * N_U),
    }