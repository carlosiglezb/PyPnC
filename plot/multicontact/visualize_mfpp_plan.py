import os
import pickle
import sys

import numpy as np

cwd = os.getcwd()
sys.path.append(cwd)

from util.path_parameterization import get_frame_des_pos

def main():
    kin_plan_path = 'data/g1_five_stage_plan.pkl'
    with open(str(kin_plan_path), 'rb') as file:
        while True:
            try:
                d = pickle.load(file)
                ik_cfree_planner = d['bez_path']
                frame_names = d['fixed_frames'][-1]     # can be used to double-check the frame names
                final_time = d['bez_points_transition_times'][0][-1]
            except EOFError:
                break

    torso_paths = ik_cfree_planner[0]
    lfoot_paths = ik_cfree_planner[1]
    rfoot_paths = ik_cfree_planner[2]
    lknee_paths = ik_cfree_planner[3]
    rknee_paths = ik_cfree_planner[4]
    lhand_paths = ik_cfree_planner[5]
    rhand_paths = ik_cfree_planner[6]
    for t in np.linspace(0, final_time, 50):
        print(f"torso pos at t={t:.4g}: {np.array2string(get_frame_des_pos(torso_paths, t), precision=4, suppress_small=True)}")

if __name__ == "__main__":
    main()