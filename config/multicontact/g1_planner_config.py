import numpy as np
from config.multicontact.planner_config import PlannerConfig

N_V = 33  # dimension of generalized velocities
N_U = 27  # dimension of control inputs

class MultiContactDoorConfig(PlannerConfig):
    # W_RIGID_LINK_SINGLE_STEP = [5., 0., 0.]
    # W_RIGID_LINK_STEP_ON_DOOR = [1000., 0., 0.]

    # ----- seq 0 (step over): single hand, step through door
    B_FINAL_VEL_CONSTRAINT = False
    # W_RIGID_LINK = [10., 0., 0.]  # option 0: knees forward
    # ALPHA = [1., 0., 0.01]           # option 0: knees forward
    # W_RIGID_LINK = [5., 0., 0.]  # option 1: knees forward
    # ALPHA = [1.4, 0., 0.1]           # option 1: knees forward
    # W_RIGID_LINK = [5., 0., 0.]   # option 2: roll shins
    # ALPHA = [1, 0., 0.05]         # option 2: roll shins
    # W_RIGID_LINK = [20., 0., {5., 10.}]  # option 3: roll shins (push up on last step)
    # ALPHA = [1, 0., 0.001]        # option 3: roll shins (push up on last step)
    # W_RIGID_LINK = [20, 0., 10.]  # option 4: roll shins high
    # ALPHA = [1, 0., 0.005]        # option 4: roll shins high
    # W_RIGID_LINK = [10, 0., 30.]    # option 5: roll shins w/ knee on right side
    # ALPHA = [1, 0.0, {0.0, 0.001}]           # option 5: roll shins w/ knee on right side
    # W_RIGID_LINK = [0., 0., 0.8]    # option 6: high knees
    # ALPHA = [0.5, 0.0, 0.1]         # option 6: high knees
    # W_RIGID_LINK = [5, 0., 2]       # option 7: roll shins outwards
    # ALPHA = [0.5, 0.0, 0.1]         # option 7: roll shins outwards
    # W_RIGID_LINK = [5, 0., 2]       # option 8: knees forward
    # ALPHA = [0.01, 0.1, 0.5]         # option 8: knee forward
    W_RIGID_LINK = [1., 0., 0.]   # option 9: knees fwd (RAL)
    # W_RIGID_LINK = [0., 0., 1.]   # option 10: knees-up (RAL)
    # W_RIGID_LINK = [0.2, 0., 0.8]   # option 11: balanced (RAL)
    ALPHA = [1, 0., 0.1]         # options 9-11 (RAL)
    # N_HORIZON_LST = [180, 240, 280, 250, 250]
    # ----- seq 1 (step on): opposite hand-foot pair at each contact
    # W_RIGID_LINK = [1.0, 0., 8.]  # option 1: high knees
    # ALPHA = [0.01, 0.01, 0.]        # option 1: high knees
    # N_HORIZON_LST = [200, 250, 280, 250, 250]
    # ----- seq 2 (step on balanced, also works with on)
    # W_RIGID_LINK = [30.0, 0., 5.]       # option 1: knee forward
    # ALPHA = [5.0, 0.01, 0.01]           # option 1: knee forward
    # W_RIGID_LINK = [20.0, 0., 5.]       # option 2: knee forward
    # ALPHA = [1.0, 0.01, 0.01]           # option 2: knee forward
    # N_HORIZON_LST = [250, 250, 250, 250, 250]
    N_HORIZON_LST = [300] * 5

    FOOT_SIZE = [0.15, 0.08]  # [length, width]

    # ----- seq 0 (step over, on_balanced, on)
    # WBC_FRAME_TRACKING_GAINS = {
    #         'torso': np.array([1.0, 1.0, 0.5] + [0.5, 0.5, 0.01]),  # (lin, ang)
    #         'feet': np.array([6.] * 3 + [0.001] * 3),  # (lin, ang)
    #         'L_knee': np.array([3.] * 3 + [0.00001] * 3),
    #         'R_knee': np.array([3.] * 3 + [0.00001] * 3),
    #         'LH': np.array([4.] * 3 + [0.00001] * 3),
    #         'RH': np.array([4.] * 3 + [0.00001] * 3)
    #     }
    # ----- seq 1 (step on)
    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([4, 4, 0.5] + [0.02, 0.02, 0.001]),
            'feet': np.array([6.] * 3 + [0.01] * 3),  # (lin, ang)
            'L_knee': np.array([4.] * 3 + [0.00001] * 3),
            'R_knee': np.array([4.] * 3 + [0.00001] * 3),
            'LH': np.array([4.] * 3 + [0.00001] * 3),
            'RH': np.array([4.] * 3 + [0.00001] * 3)
        }
    # ----- seq 2 (step on balanced)
    # WBC_FRAME_TRACKING_GAINS = {
    #         'torso': np.array([2.5, 2.5, 2.5] + [0.5, 0.5, 0.001]),
    #         'feet': np.array([8.] * 3 + [0.00001] * 3),  # (lin, ang)
    #         'L_knee': np.array([2.] * 3 + [0.00001] * 3),
    #         'R_knee': np.array([2.] * 3 + [0.00001] * 3),
    #         'LH': np.array([4.] * 3 + [0.00001] * 3),
    #         'RH': np.array([4.] * 3 + [0.00001] * 3)
    #     }
    WBC_FINAL_FRAME_TRACKING_GAINS = {
            'torso': np.array([8, 8, 8] + [2, 2, 2]),  # (lin, ang)  step over, on_balanced
            'feet': np.array([12.] * 3 + [6.5] * 3),  # (lin, ang)
            'L_knee': np.array([6.] * 3 + [0.00001] * 3),
            'R_knee': np.array([6.] * 3 + [0.00001] * 3),
            'LH': np.array([8.] * 3 + [0.00001] * 3),
            'RH': np.array([8.] * 3 + [0.00001] * 3)
        }
    WBC_WEIGHTED_COSTS = {
        #                 q_b_lin, q_b_ang, q_j, (v_b_lin, v_b_ang), v_j
        'xReg': np.array([0] * 6 + [0.01] * (N_V - 6) + [1.0] * 3 + [0.1] * 3 + [2.0] * (N_V - 6)), # step on
        'uReg': np.array([1.0] * N_U),
    }
    WBC_PHASE_END_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [20.0] * 3 + [5.] * (N_V - 6) + [30.] * N_V),    # step on balanced
    }
    WBC_FINAL_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [20.0] * 3 + [5.] * (N_V - 6) + [30.] * N_V),    # step on balanced
    }

class MultiContactTiltedStairsConfig(PlannerConfig):
    B_FINAL_VEL_CONSTRAINT = True
    W_RIGID_LINK = [0., 0., 1.]
    ALPHA = [1, 0.0, 0.01]
    FOOT_SIZE = [0.15, 0.08]  # [length, width]
    # N_HORIZON_LST = [180, 250, 250, 250, 280, 250]
    N_HORIZON_LST = [300] * 6

    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([5, 5, 4, 0.2, 0.2, 0.01]),
            'feet': np.array([6.0] * 3 + [0.5, 0.1, 0.1]),
            'L_knee': np.array([3.0] * 3 + [0.0001] * 3),
            'R_knee': np.array([3.0] * 3 + [0.0001] * 3),
            'LH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
            'RH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
        }
    WBC_FINAL_FRAME_TRACKING_GAINS = {
            'torso': np.array([8.0, 8.0, 8.0, 2, 2, 2]),
            'feet': np.array([12.0] * 3 + [8.0] * 3),
            'L_knee': np.array([6.0] * 3+ [0.0001] * 3),
            'R_knee': np.array([6.0] * 3+ [0.0001] * 3),
            'LH': np.array([8] * 3 + [0.0001] * 3),
            'RH': np.array([8] * 3 + [0.0001] * 3),
        }
    WBC_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 6 + [0.05] * (N_V - 6) + [2.0] * N_V),
        'uReg': np.array([1.] * N_U),
    }

    WBC_PHASE_END_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [10.0] * 3 + [5.] * (N_V - 6) + [20.] * N_V),
    }
    WBC_FINAL_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [10.0] * 3 + [5.] * (N_V - 6) + [20.] * N_V),
    }