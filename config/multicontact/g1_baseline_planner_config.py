import numpy as np

from config.multicontact.baseline_planner_config import BaselinePlannerConfig

N_V = 33  # dimension of generalized velocities
N_U = 27  # dimension of control inputs

class MultiContactDoorConfig(BaselinePlannerConfig):
    N_HORIZON_LST = [180, 280, 280, 250, 250]
    FOOT_SIZE = [0.15, 0.08]  # [length, width]

    # ----- seq 0 (step over, on_balanced, on)
    # WBC_FRAME_TRACKING_GAINS = {
    #         'torso': np.array([1.0, 1.0, 0.5] + [0.5, 0.5, 0.01]),  # (lin, ang)
    #         'feet': np.array([8.] * 3 + [0.001] * 3),  # (lin, ang)
    #         'L_knee': np.array([4.] * 3 + [0.00001] * 3),
    #         'R_knee': np.array([4.] * 3 + [0.00001] * 3),
    #         'LH': np.array([4.] * 3 + [0.00001] * 3),
    #         'RH': np.array([4.] * 3 + [0.00001] * 3)
    #     }
    # ----- seq 1 (step on)
    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([2.0, 2.5, 1.5] + [0.5, 0.5, 0.001]),
            'feet': np.array([8.] * 3 + [0.001] * 3),  # (lin, ang)
            'L_knee': np.array([4.] * 3 + [0.00001] * 3),
            'R_knee': np.array([4.] * 3 + [0.00001] * 3),
            'LH': np.array([4.] * 3 + [0.00001] * 3),
            'RH': np.array([4.] * 3 + [0.00001] * 3)
        }
    # ----- seq 2 (step on balanced)
    # WBC_FRAME_TRACKING_GAINS = {
    #         'torso': np.array([2.5, 3.5, 2.5] + [0.5, 0.5, 0.001]),
    #         'feet': np.array([8.] * 3 + [0.00001] * 3),  # (lin, ang)
    #         'L_knee': np.array([3.] * 3 + [0.00001] * 3),
    #         'R_knee': np.array([3.] * 3 + [0.00001] * 3),
    #         'LH': np.array([4.] * 3 + [0.00001] * 3),
    #         'RH': np.array([4.] * 3 + [0.00001] * 3)
    #     }
    WBC_FINAL_FRAME_TRACKING_GAINS = {
            'torso': np.array([3, 3.0, 1.5] + [0.5, 0.5, 0.1]),  # (lin, ang)  step over, on_balanced
            # 'torso': np.array([5, 5.0, 5.0] + [1.0, 1.0, 1.]),  # (lin, ang)   step on
            'feet': np.array([12.] * 3 + [6.5] * 3),  # (lin, ang)
            'L_knee': np.array([6.] * 3 + [0.00001] * 3),
            'R_knee': np.array([6.] * 3 + [0.00001] * 3),
            'LH': np.array([4.] * 3 + [0.00001] * 3),
            'RH': np.array([4.] * 3 + [0.00001] * 3)
        }
    WBC_WEIGHTED_COSTS = {
        #                 q_b_lin, q_b_ang, q_j, (v_b_lin, v_b_ang), v_j
        'xReg': np.array([0] * 3 + [0] * 3 + [0] * (N_V - 6) + [0.2] * 6 + [2.0] * (N_V - 6)), # step on
        'uReg': np.array([0.5] * N_U),  # over, on
    }
    WBC_PHASE_END_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [0] * 3 + [0] * (N_V - 6) + [40.] * N_V),    # step on balanced
    }
    WBC_FINAL_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [0] * 3 + [0] * (N_V - 6) + [40.] * N_V),    # step on balanced
    }

class MultiContactTiltedStairsConfig(BaselinePlannerConfig):
    FOOT_SIZE = [0.15, 0.08]  # [length, width]
    N_HORIZON_LST = [180, 250, 250, 250, 280, 250]

    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([2.0, 1.5, 1.0, 0.5, 0.5, 0.1]),
            'feet': np.array([8.0] * 3 + [0.1, 0.1, 0.1]),
            'L_knee': np.array([6.0] * 3 + [0.0001] * 3),
            'R_knee': np.array([6.0] * 3 + [0.0001] * 3),
            'LH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
            'RH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
        }
    WBC_FINAL_FRAME_TRACKING_GAINS = {
            'torso': np.array([2.0, 1.5, 1.0, 0.5, 0.5, 0.1]),
            'feet': np.array([10.0] * 3 + [10.0] * 3),
            'L_knee': np.array([4.0] * 3+ [0.0001] * 3),
            'R_knee': np.array([4.0] * 3+ [0.0001] * 3),
            'LH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
            'RH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
        }
    WBC_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [10.0] * 3 + [1.] * (N_V - 6) + [4.] * N_V),
        'uReg': np.array([15.] * N_U),
    }
    WBC_FINAL_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [50.0] * 3 + [10.] * (N_V - 6) + [40.] * N_V),
    }