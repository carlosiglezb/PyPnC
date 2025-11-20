#
# Baseline Planner
#
import numpy as np
import pinocchio
from .planner_surface_contact import MotionFrameSequencer

frame_names_lst = ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']

class BaselineFramePlanner:
    def __init__(self,
                 robot_data: pinocchio.Data,
                 plan_to_model_ids: dict[str: int],
                 motion_frames_seq: MotionFrameSequencer,
                 fixed_frames: list[list[str]],
                 interpolation:str = "constant",):
        self.robot_data = robot_data
        self.plan_to_model_ids = plan_to_model_ids
        self.all_frame_targets = self.create_frame_targets_dict(motion_frames_seq, fixed_frames)
        self.interpolation = interpolation

    def create_frame_targets_dict(self,
                                  motion_frames_seq: MotionFrameSequencer,
                                  fixed_frames: list[list[str]],
                                  ) -> list[dict[str, np.array]]:
        robot_data = self.robot_data
        plan_to_model_ids = self.plan_to_model_ids

        all_frame_targets = []
        current_frame_targets = {}
        phase = 0

        # get initial frame positions
        for fr_name in frame_names_lst:
            current_frame_targets[fr_name] = robot_data.oMf[plan_to_model_ids[fr_name]].translation
        phase += 1
        all_frame_targets.append(current_frame_targets)

        # get the next frame positions from either fixed frames or motion frames
        for _ in range(len(motion_frames_seq.motion_frame_lst)):
            current_frame_targets = {}
            for fr_name in frame_names_lst:
                if fr_name in fixed_frames[phase - 1]:
                    # fixed frame: keep previous position
                    current_frame_targets[fr_name] = all_frame_targets[-1][fr_name]
                elif fr_name in motion_frames_seq.motion_frame_lst[phase - 1].keys():
                    # motion frame: get new target position
                    current_frame_targets[fr_name] = motion_frames_seq.motion_frame_lst[phase - 1][fr_name]
                else:
                    continue
            all_frame_targets.append(current_frame_targets)
            phase += 1

        return all_frame_targets

    def get_phase_targets(self, phase: int) -> dict[str, np.array]:
        return self.all_frame_targets[phase]

    def get_linear_targets(self, phase: int,
                           time: float,
                           final_t: float) -> dict[str, np.array]:
        s = np.clip(time / final_t, 0.0, 1.0)

        current_phase_targets = self.get_phase_targets(phase)
        next_phase_targets = self.get_phase_targets(phase + 1)

        # generate intermediate frames (exclude previous, include final)
        interp_targets = {}
        for fr_name in frame_names_lst:
            if fr_name in current_phase_targets.keys() and fr_name in next_phase_targets.keys():
                start = np.array(current_phase_targets[fr_name], dtype=float)
                end = np.array(next_phase_targets[fr_name], dtype=float)
                interp_targets[fr_name] = (1.0 - s) * start + s * end
            elif fr_name in current_phase_targets.keys() and fr_name not in next_phase_targets.keys():
                interp_targets[fr_name] = current_phase_targets[fr_name]
        return interp_targets