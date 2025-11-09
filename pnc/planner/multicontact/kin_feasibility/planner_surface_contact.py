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
        self.motion_frame_lst: list[dict[str: np.ndarray]] = []
        self.contact_frame_lst: list[list[PlannerSurfaceContact]] = []
        self.b_initial_vel = False
        self.b_initial_acc = False
        self.b_final_vel = False
        self.b_final_acc = False

        self.eps_vel = 0.01     # default to 1 cm/s

    def add_motion_frame(self, frame_goal_dict: dict[str, np.ndarray]):
        self.motion_frame_lst.append(frame_goal_dict)

    def add_contact_surfaces(self, contact_surfaces: list[PlannerSurfaceContact]):
        self.contact_frame_lst.append(contact_surfaces)

    def get_motion_frames(self):
        return self.motion_frame_lst

    def get_contact_surfaces(self):
        return self.contact_frame_lst

def get_contact_seq_from_fixed_frames_seq(fixed_frames_seq: list[list[str]]) -> list[list[str]]:
    contact_frames = ['LF', 'RF', 'LH', 'RH']
    contact_frames_seq = []
    for seq in fixed_frames_seq:
        cf = []
        for ff in seq:
            # check for frames that can be in contact:
            if ff in contact_frames:
               cf.append(ff)

        # throw error if no contact frames were detected
        if len(cf) == 0:
            print(f"No contact frames found in fixed frames")
            return

        # append contact frames found in current sequence
        contact_frames_seq.append(cf)

    # move contact-making to the end
    for cs_i, cs in enumerate(contact_frames_seq):
        if cs_i == 0:   # ignore first contact sequence
            continue

        b_phase_done = False
        # find new contact to place them at the end for impulse model
        for ccon in cs:
            if (not b_phase_done) and (ccon not in contact_frames_seq[cs_i - 1]):
                contact_frames_seq[cs_i].remove(ccon)
                contact_frames_seq[cs_i].append(ccon)
                b_phase_done = True

    # remove hands from last sequence when added for smoothing
    if contact_frames_seq[-1] == contact_frames:
        contact_frames_seq.pop(-1)
        contact_frames_seq.pop(-1)

    return contact_frames_seq

def get_contact_planes_from_motion_frames_seq(contact_seq: list[str],
                                              motion_frames_seq: MotionFrameSequencer) -> list[dict[str: np.ndarray]]:
    contact_planes: dict[str: np.ndarray] = []
    for i, seq_contact in enumerate(contact_seq):
        seq_contact_planes = {}
        if i == 0 or i == len(motion_frames_seq.motion_frame_lst) - 1:
            # currently, we are assuming we start/end on a flat surface
            seq_contact_planes['LF'] = np.array([0, 0, 1])
            seq_contact_planes['RF'] = np.array([0, 0, 1])
        else:
            for fr_name in seq_contact:
                b_found_frame = False
                # search for latest assigned contact plane
                for j in range(i, -1, -1):
                    for k, fr_contact in enumerate(motion_frames_seq.contact_frame_lst[j]):
                        if b_found_frame:
                            break
                        # check if the contact frame name matches
                        if fr_name == fr_contact.contact_frame_name:
                            seq_contact_planes[fr_name] = motion_frames_seq.contact_frame_lst[j][k].surface_normal
                            b_found_frame = True
                            break
                        if j == 0:
                            # if no contact plane was found, check the initial contacts
                            if fr_name in contact_planes[0]:
                                seq_contact_planes[fr_name] = contact_planes[0][fr_name]
        contact_planes.append(seq_contact_planes)

    return contact_planes