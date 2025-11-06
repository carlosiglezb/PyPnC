import numpy as np
from abc import ABC

class PlannerConfig(ABC):
    W_RIGID_LINK : list[float] = None     # weights on rigid link relaxation (wx, wy, wz)
    ALPHA : list[float] = None            # weights on the task cost function (derivatives)
    FOOT_SIZE : list[float] = None        # size of the feet in the form [length, width]
    N_HORIZON_LST : list[int] = None      # list of horizon lengths for the whole-body planner
    WBC_FRAME_TRACKING_GAINS : dict[str: np.ndarray] = {}  # gains for the whole-body planner frame tracking tasks
    WBC_FINAL_FRAME_TRACKING_GAINS : dict[str: np.ndarray] = {}  # gains for the terminal state of whole-body planner frame tracking tasks
    WBC_WEIGHTED_COSTS : dict[str: np.ndarray] = {}  # costs for the whole-body planner frame tracking tasks
    WBC_PHASE_END_WEIGHTED_COSTS : dict[str: np.ndarray] = {}  # costs for WBP at end of each contact phase
    WBC_FINAL_WEIGHTED_COSTS : dict[str: np.ndarray] = {}  # costs for the whole-body planner frame regularization task
    # ------ seq 0
    # WBP_CONTACT_JVEL_SCALE : float = 4.0   # weight on joint velocity scale of contact limb in the whole-body planner
    # WBP_BAUMGARTE_GAINS3D : list[float] = [1e-6, 1e-4]  # gains for the Baumgarte stabilization of contact constraints in the whole-body planner [pos ref, velocity]
    # WBP_BAUMGARTE_GAINS6D : list[float] = [1e-6, 1e-6]  # gains for the Baumgarte stabilization of contact constraints in the whole-body planner [rot ref, velocity]
    # ------ seq 1
    WBP_CONTACT_JVEL_SCALE : float = 4.0   # weight on joint velocity scale of contact limb in the whole-body planner -- seq 1
    WBP_BAUMGARTE_GAINS3D : list[float] = [1e-6, 1e-6]  # gains for the Baumgarte stabilization of contact constraints in the whole-body planner [pos ref, velocity]
    WBP_BAUMGARTE_GAINS6D : list[float] = [1e-6, 1e-6]  # gains for the Baumgarte stabilization of contact constraints in the whole-body planner [rot ref, velocity]
    # ------ seq 2
    # WBP_CONTACT_JVEL_SCALE : float = 6.0   # weight on joint velocity scale of contact limb in the whole-body planner -- seq 1
    # WBP_BAUMGARTE_GAINS3D : list[float] = [1e-6, 1e-6]  # gains for the Baumgarte stabilization of contact constraints in the whole-body planner [pos ref, velocity]
    # WBP_BAUMGARTE_GAINS6D : list[float] = [1e-6, 1e-6]  # gains for the Baumgarte stabilization of contact constraints in the whole-body planner [rot ref, velocity]
    WBC_COST_WEIGHTS = {
        'friction': 3e0,    # step_over, on_balanced, step_on
        # 'friction': 2e0,    # ergoCub (stairs)
        'frame_goal': 3e3,  # step over
        # 'frame_goal': 2e4,  # ergo (stairs):
        # 'frame_goal': 1e2,  # step on
        'xReg': 5e-1,
        'uReg': 5e0,
        'xBounds': 1e4,   # step on, step over, on_balanced
        # 'xBounds': 1e3,   # ergoCub (stairs):
        'sca': 5e3     # using Exponential activation
        # 'sca': -1e2   # using QuadFlatExp activation
    }
    WBC_FINAL_COST_WEIGHTS = {
        'friction': 1e-2,   # step_over, on_balanced, step_on
        # 'frame_goal': 3e3,
        'frame_goal': 4e4,  # ergo (stairs):
        # 'xReg': 2e0,
        'xReg': 5e-1,    # step over, on_balanced
        'uReg': 5e0,
        'xBounds': 6e3,     # step on, step over, on_balanced
        # 'xBounds': 1e3,     # ergoCub (stairs)
    }
    WBC_IMPULSE_COST_WEIGHTS = {
        'frame_goal': 1e1,
        'xReg': 5e-1,
    }
