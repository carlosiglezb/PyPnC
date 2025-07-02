import numpy as np
from config.multicontact.planner_config import PlannerConfig


class MultiContactDoorConfig(PlannerConfig):
    W_RIGID_LINK = [500., 0., 50.]  # tested on single step
    # W_RIGID_LINK_SINGLE_STEP = [500., 0., 50.]
    W_RIGID_POLY = [0.1621, 0.0, 0.]        # TODO check if they need to be different from W_RIGID_LINK
    N_HORIZON_LST = [150, 150, 100]

    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([3.0] * 3 + [0.5, 0.5, 0.01]),    # (lin, ang)
            'feet': np.array([6.] * 3 + [0.00001] * 3),         # (lin, ang)
            'L_knee': np.array([2.] * 3 + [0.00001] * 3),
            'R_knee': np.array([2.] * 3 + [0.00001] * 3),
            'LH': np.array([2.] * 3 + [0.00001] * 3),
            'RH': np.array([2.] * 3 + [0.00001] * 3)
        }

class MultiContactTiltedStairsConfig(PlannerConfig):
    W_RIGID_LINK = [1., 0., 10.]
    W_RIGID_LINK_STEP_ON_DOOR = [1000., 0., 0.]

    WBC_FRAME_TRACKING_GAINS = {
            'torso': np.array([3.0] * 3 + [0.5, 0.5, 0.01]),    # (lin, ang)
            'feet': np.array([6.] * 3 + [0.00001] * 3),         # (lin, ang)
            'L_knee': np.array([2.] * 3 + [0.00001] * 3),
            'R_knee': np.array([2.] * 3 + [0.00001] * 3),
            'LH': np.array([2.] * 3 + [0.00001] * 3),
            'RH': np.array([2.] * 3 + [0.00001] * 3)
        }