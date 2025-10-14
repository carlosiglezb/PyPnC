import numpy as np
from config.multicontact.planner_config import PlannerConfig

N_V = 38  # dimension of generalized velocities (removed camera)
N_U = 32  # dimension of control inputs

class MultiContactDoorConfig(PlannerConfig):
    FOOT_SIZE = [0.15, 0.09]  # [length, width]

    # ----- seq 0 (step over): single hand, step through door
    # W_RIGID_LINK = [2.0, 0., 1.0]   # used to step over door
    # ALPHA = [0.1, 0.0, 0.1]
    # N_HORIZON_LST = [180, 280, 280, 250, 250]
    # ----- seq 1 (step on): opposite hand-foot pair at each contact
    # ----- seq 2 (step on balanced)
    W_RIGID_LINK = [30.0, 0., 5.]       # option 1: knee forward
    ALPHA = [1.0, 0.01, 0.05]           # option 1: knee forward
    N_HORIZON_LST = [180, 280, 280, 250, 250]


    # ----- seq 0 (step over): single hand, step through door
    # WBC_FRAME_TRACKING_GAINS = {
    #         'torso': np.array([2.0, 2.0, 1.5] + [1.0, 1.0, 0.001]),  # (lin, ang)
    #         'feet': np.array([10.] * 3 + [0.01, 0.01, 0.01]),  # (lin, ang)
    #         'L_knee': np.array([6.] * 3 + [0.00001] * 3),
    #         'R_knee': np.array([6.] * 3 + [0.00001] * 3),
    #         'LH': np.array([2.] * 3 + [0.00001] * 3),
    #         'RH': np.array([2] * 3 + [0.00001] * 3)
    #     }
    # ----- seq 1 (step on)
    # WBC_FRAME_TRACKING_GAINS = {
    #         'torso': np.array([2.0, 2.0, 1.5] + [1.0, 1.0, 0.001]),  # (lin, ang)
    #         'feet': np.array([10.] * 3 + [0.01, 0.01, 0.01]),  # (lin, ang)
    #         'L_knee': np.array([6.] * 3 + [0.00001] * 3),
    #         'R_knee': np.array([6.] * 3 + [0.00001] * 3),
    #         'LH': np.array([2.] * 3 + [0.00001] * 3),
    #         'RH': np.array([2] * 3 + [0.00001] * 3)
    #     }
    # ----- seq 2 (step on balanced)
    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([1.5, 1.5, 1.0] + [0.1, 0.1, 0.001]),
            'feet': np.array([10.] * 3 + [0.01] * 3),  # (lin, ang)
            'L_knee': np.array([4.] * 3 + [0.00001] * 3),
            'R_knee': np.array([4.] * 3 + [0.00001] * 3),
            'LH': np.array([4.] * 3 + [0.00001] * 3),
            'RH': np.array([4.] * 3 + [0.00001] * 3)
        }

    WBC_FINAL_FRAME_TRACKING_GAINS = {
            'torso': np.array([4, 4.0, 3.5] + [0.5, 0.5, 0.1]),  # (lin, ang)  step over, on_balanced
            # 'torso': np.array([5, 5.0, 5.0] + [1.0, 1.0, 1.]),  # (lin, ang)   step on
            'feet': np.array([12.] * 3 + [6.5] * 3),  # (lin, ang)
            'L_knee': np.array([6.] * 3 + [0.00001] * 3),
            'R_knee': np.array([6.] * 3 + [0.00001] * 3),
            'LH': np.array([4.] * 3 + [0.00001] * 3),
            'RH': np.array([4.] * 3 + [0.00001] * 3)
        }
    WBC_WEIGHTED_COSTS = {
        # 'xReg': np.array([0] * 3 + [10.0] * 3 + [2.] * (N_V - 6) + [4.] * N_V),   # step over
        'xReg': np.array([0] * 3 + [1.0] * 3 + [2.] * (N_V - 6) + [0.4] * N_V),
        'uReg': np.array([0.5] * N_U),
    }

    WBC_FINAL_WEIGHTED_COSTS = {
        # 'xReg': np.array([0] * 3 + [30.0] * 3 + [5.] * (N_V - 6) + [20.] * N_V),    # step over
        'xReg': np.array([0] * 3 + [30.0] * 3 + [2.] * (N_V - 6) + [40.] * N_V),    # step on
    }

class MultiContactTiltedStairsConfig(PlannerConfig):
    W_RIGID_LINK = [1., 0., 10.]

    # ALPHA = [0.1, 0.2, 0.8]
    ALPHA = [0.1, 0.0, 0.01]
    FOOT_SIZE = [0.15, 0.09]  # [length, width]
    N_HORIZON_LST = [180, 250, 250, 250, 280, 250]

    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([3.5, 3.5, 1.0] + [1.0, 1.0, 0.1]),  # (lin, ang)
            'feet': np.array([10.] * 3 + [0.1, 0.5, 0.1]),  # (lin, ang)
            'L_knee': np.array([6.] * 3 + [0.00001] * 3),
            'R_knee': np.array([6.] * 3 + [0.00001] * 3),
            'LH': np.array([4.] * 3 + [0.00001] * 3),
            'RH': np.array([4] * 3 + [0.00001] * 3)
        }
    WBC_FINAL_FRAME_TRACKING_GAINS = {
            'torso': np.array([3, 3.0, 2.0] + [1.0, 1.0, 1.]),  # (lin, ang)   step on
            'feet': np.array([12.] * 3 + [12] * 3),  # (lin, ang)
            'L_knee': np.array([6.] * 3 + [0.00001] * 3),
            'R_knee': np.array([6.] * 3 + [0.00001] * 3),
            'LH': np.array([4.] * 3 + [0.00001] * 3),
            'RH': np.array([4.] * 3 + [0.00001] * 3)
        }
    WBC_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [10.0] * 3 + [0.8] * (N_V - 6) + [3.0] * N_V),
        'uReg': np.array([4.0] * N_U),
    }

    WBC_FINAL_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [40.0] * 3 + [5.] * (N_V - 6) + [20.] * N_V),    # step on, over
    }