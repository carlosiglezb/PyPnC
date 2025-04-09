import numpy as np


class PlannerSurfaceContact:
    """
    Surface contact information of a particular plane.

    Parameters:
        f_name: String
            Name of the frame coming in contact with this surface
        s_normal: nparray(3D)
            Vector of the normal component of the planar surface
    """
    def __init__(self, f_name: str,
                 s_normal: np.array):
        self.contact_frame_name = f_name
        self.surface_normal = s_normal

        self.b_initial_vel = False
        self.b_initial_acc = False
        self.b_final_vel = False
        self.b_final_acc = False

        self.previous_normal = [0, 0, 0]

        self.eps_vel = 0.05     # default to 1 cm/s

    def set_contact_breaking_velocity(self, prev_normal):
        self.b_initial_vel = True
        self.previous_normal = prev_normal

    def get_contact_breaking_velocity(self):
        return (1. / self.eps_vel) * self.previous_normal

    def get_surface_normal(self):
        return self.surface_normal


class MotionFrameSequencer:
    def __init__(self):
        self.motion_frame_lst = []
        self.contact_frame_lst = []

        self.b_initial_vel = False
        self.b_initial_acc = False
        self.b_final_vel = False
        self.b_final_acc = False

        self.eps_vel = 0.01     # default to 1 cm/s

    def add_motion_frame(self, frame_goal_dict: dict[str, np.ndarray]):
        self.motion_frame_lst.append(frame_goal_dict)

    def add_contact_surface(self, contact_surface: PlannerSurfaceContact):
        self.contact_frame_lst.append(contact_surface)

    def get_motion_frames(self):
        return self.motion_frame_lst

    def get_contact_surfaces(self):
        return self.contact_frame_lst
