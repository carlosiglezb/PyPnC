class PlannerSurfaceContact:
    """
    Surface contact information of a particular plane.

    Parameters:
        f_name: String
            Name of the frame coming in contact with this surface
        s_normal: nparray(3D)
            Vector of the normal component of the planar surface
    """
    def __init__(self, f_name, s_normal):
        self.contact_frame_name = f_name
        self.surface_normal = s_normal

        self.b_initial_vel = False
        self.b_initial_acc = False
        self.b_final_vel = False
        self.b_final_acc = False

        self.previous_normal = [0, 0, 0]

        self.eps_vel = 0.01     # default to 1 cm/s

    def set_contact_breaking_velocity(self, prev_normal):
        self.b_initial_vel = True
        self.previous_normal = prev_normal

    def get_contact_breaking_velocity(self):
        return self.eps_vel * self.previous_normal


class ContactFrameLocation:
    def __init__(self, pos_w, surface):
        self.pos = pos_w
        self.surface = surface
        