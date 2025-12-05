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
                 interpolation_time: float=3.0,
                 interpolation:str = "constant",):
        self.robot_data = robot_data
        self.plan_to_model_ids = plan_to_model_ids
        self.all_frame_targets = self.create_frame_targets_dict(motion_frames_seq, fixed_frames)
        self.interpolation = interpolation
        self.T = interpolation_time

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

        # fill in any missing target frames
        # loop through each frame backwards in time and record last known position and time
        for fr_name in frame_names_lst:
            last_known_pos = None
            missing_segments = 0
            for phase in reversed(range(len(all_frame_targets))):
                if fr_name in all_frame_targets[phase].keys():
                    if missing_segments > 0 and last_known_pos is not None:
                        segment_inc = (last_known_pos - all_frame_targets[phase][fr_name]) / (missing_segments+1)
                        # fill in missing segments with intermediate values
                        for fill_phase in range(phase + 1, phase + 1 + missing_segments):
                            all_frame_targets[fill_phase][fr_name] = all_frame_targets[fill_phase-1][fr_name] + segment_inc
                        missing_segments = 0
                    last_known_pos = all_frame_targets[phase][fr_name]
                else:
                    missing_segments+=1

        return all_frame_targets

    def get_phase_targets(self, phase: int) -> dict[str, np.array]:
        return self.all_frame_targets[phase]

    def get_linear_targets(self, phase: int,
                           time: float) -> dict[str, np.array]:
        start_t = phase * self.T
        s = np.clip((time - start_t) / self.T, 0.0, 1.0)

        current_phase_targets = self.get_phase_targets(phase)
        # clip to last phase
        if phase < len(self.all_frame_targets) - 1:
            next_phase_targets = self.get_phase_targets(phase + 1)
        else:
            next_phase_targets = current_phase_targets

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