import numpy as np
from abc import ABC

class PlannerConfig(ABC):
    W_RIGID_LINK : list[float] = None     # weights on rigid link relaxation (wx, wy, wz)
    ALPHA : list[float] = None            # weights on the task cost function (derivatives)
    FOOT_SIZE : list[float] = None        # size of the feet in the form [length, width]
    N_HORIZON_LST : list[int] = None      # list of horizon lengths for the whole-body planner
    WBC_FRAME_TRACKING_GAINS : dict[str: np.ndarray] = {}  # gains for the whole-body planner frame tracking tasks
    WBC_WEIGHTED_COSTS : dict[str: np.ndarray] = {}  # costs for the whole-body planner frame tracking tasks
    WBC_COST_WEIGHTS = {
        'friction': 1e1,
        'frame_goal': 6e4,
        'xReg': 5e-3,
        'uReg': 1e-4,
        'xBounds': 5e4,
    }
    WBC_FINAL_COST_WEIGHTS = {
        'friction': 1e1,
        'frame_goal': 5e6,
        'xReg': 5e-5,
        'uReg': 1e-4,
        'xBounds': 1000,
    }
    WBC_IMPULSE_COST_WEIGHTS = {
        'frame_goal': 100,
        'xReg': 5e-2,
    }
