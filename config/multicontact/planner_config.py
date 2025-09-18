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
    WBC_FINAL_WEIGHTED_COSTS : dict[str: np.ndarray] = {}  # costs for the whole-body planner frame regularization task
    WBC_COST_WEIGHTS = {
        # 'friction': 5e0,  # on_balanced
        'friction': 2e0,    # step_over
        'frame_goal': 5e3,
        'xReg': 5e-1,
        'uReg': 1e-1,
        'xBounds': 8e3,
        # 'xBounds': 6e4,   # step over/on_balanced
        # 'sca': 2e-2     # using Exponential activation
        'sca': -1e2   # using QuadFlatExp activation
    }
    WBC_FINAL_COST_WEIGHTS = {
        # 'friction': 1e0,  # on_balanced
        'friction': 2e0,   # step_over
        'frame_goal': 5e3,
        'xReg': 2e0,
        'uReg': 1e-1,
        # 'xBounds': 5e4,   # step over/on_balanced
        'xBounds': 6e3,
    }
    WBC_IMPULSE_COST_WEIGHTS = {
        'frame_goal': 1e3,
        'xReg': 5e-2,
    }
