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
    W_RIGID_LINK = [5., 0., 0.]   # option 9: knees fwd
    ALPHA = [5, 0., 0.1]         # option 9: knees fwd
    # W_RIGID_LINK = [0., 0., 5.]   # option 10: knees-up
    # ALPHA = [1, 0., 0.1]         # option 10: knees-up
    # W_RIGID_LINK = [5., 0., 5.]   # option 11: balanced
    # ALPHA = [1, 0., 0.1]         # option 11: balanced
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
    N_HORIZON_LST = [250, 250, 250, 250, 250]
    # N_HORIZON_LST = [180, 280, 280, 250, 250]

    FOOT_SIZE = [0.15, 0.08]  # [length, width]

    # ----- seq 0 (step over, on_balanced, on)
    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([3.0, 3.0, 2.5] + [0.5, 0.5, 0.01]),  # (lin, ang)
            'feet': np.array([8.] * 3 + [0.001] * 3),  # (lin, ang)
            'L_knee': np.array([4.] * 3 + [0.00001] * 3),
            'R_knee': np.array([4.] * 3 + [0.00001] * 3),
            'LH': np.array([4.] * 3 + [0.00001] * 3),
            'RH': np.array([4.] * 3 + [0.00001] * 3)
        }
    # ----- seq 1 (step on)
    # WBC_FRAME_TRACKING_GAINS = {
    #         'torso': np.array([0.5, 0.5, 0.1] + [0.2, 0.2, 0.001]),
    #         'feet': np.array([4.] * 3 + [0.001] * 3),  # (lin, ang)
    #         'L_knee': np.array([2.] * 3 + [0.00001] * 3),
    #         'R_knee': np.array([2.] * 3 + [0.00001] * 3),
    #         'LH': np.array([4.] * 3 + [0.00001] * 3),
    #         'RH': np.array([4.] * 3 + [0.00001] * 3)
    #     }
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
            # 'torso': np.array([5, 5.0, 5.0] + [1.0, 1.0, 1.]),  # (lin, ang)   step on
            'feet': np.array([12.] * 3 + [6.5] * 3),  # (lin, ang)
            'L_knee': np.array([6.] * 3 + [0.00001] * 3),
            'R_knee': np.array([6.] * 3 + [0.00001] * 3),
            'LH': np.array([8.] * 3 + [0.00001] * 3),
            'RH': np.array([8.] * 3 + [0.00001] * 3)
        }
    WBC_WEIGHTED_COSTS = {
        #                 q_b_lin, q_b_ang, q_j, (v_b_lin, v_b_ang), v_j
        # 'xReg': np.array([0] * 3 + [0.0] * 3 + [2.] * (N_V - 6) + [0.2] * 6 + [2.0] * (N_V - 6)), # step on
        'xReg': np.array([0] * 3 + [0.0] * 3 + [2.0] * (N_V - 6) + [0.2] * 6 + [0.8] * (N_V - 6)),  # step over
        # 'xReg': np.array([0] * 3 + [0] * 3 + [2.] * (N_V - 6) + [0.2] * 6 + [3.0] * (N_V - 6)), # step on balanced
        'uReg': np.array([0.4] * N_U),
        # 'uReg': np.array([0.5] * N_U),  # over, on
    }
    WBC_PHASE_END_WEIGHTED_COSTS = {
        # 'xReg': np.array([0] * 3 + [6.0] * 3 + [4.] * (N_V - 6) + [5.] * N_V),    # step on balanced
        'xReg': np.array([0] * 3 + [20.0] * 3 + [5.] * (N_V - 6) + [30.] * N_V),    # step on balanced
    }
    WBC_FINAL_WEIGHTED_COSTS = {
        # 'xReg': np.array([0] * 3 + [30.0] * 3 + [5.] * (N_V - 6) + [20.] * N_V),    # step on, over
        'xReg': np.array([0] * 3 + [20.0] * 3 + [5.] * (N_V - 6) + [30.] * N_V),    # step on balanced
    }

class MultiContactTiltedStairsConfig(PlannerConfig):
    B_FINAL_VEL_CONSTRAINT = True
    W_RIGID_LINK = [0., 0., 5.]
    ALPHA = [0.05, 0.0, 0.1]
    # ALPHA = [0.1, 0.0, 1.0]
    # ALPHA = [0.5, 0.2, 0.0]
    FOOT_SIZE = [0.15, 0.08]  # [length, width]
    N_HORIZON_LST = [180, 250, 250, 250, 280, 250]
    # N_HORIZON_LST = [300] * 6

    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([0.5, 0.5, 0.3, 0.5, 0.5, 0.1]),
            'feet': np.array([4.0] * 3 + [0.5, 0.1, 0.1]),
            'L_knee': np.array([2.0] * 3 + [0.0001] * 3),
            'R_knee': np.array([2.0] * 3 + [0.0001] * 3),
            'LH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
            'RH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
        }
    WBC_FINAL_FRAME_TRACKING_GAINS = {
            'torso': np.array([5.0, 5.0, 5.0, 0.5, 0.5, 0.1]),
            'feet': np.array([10.0] * 3 + [10.0] * 3),
            'L_knee': np.array([4.0] * 3+ [0.0001] * 3),
            'R_knee': np.array([4.0] * 3+ [0.0001] * 3),
            'LH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
            'RH': np.array([4.0, 4.0, 4.0] + [0.0001] * 3),
        }
    WBC_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [0.0] * 3 + [0.5] * (N_V - 6) + [4.] * N_V),
        'uReg': np.array([15.] * N_U),
    }

    WBC_PHASE_END_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [30.0] * 3 + [5.] * (N_V - 6) + [40.] * N_V),
    }
    WBC_FINAL_WEIGHTED_COSTS = {
        'xReg': np.array([0] * 3 + [30.0] * 3 + [5.] * (N_V - 6) + [40.] * N_V),
    }