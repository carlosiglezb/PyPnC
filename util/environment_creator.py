import numpy as np
from pydrake.geometry.optimization import HPolyhedron


class TiltedBox:
    def __init__(self,
                 box_width: float,
                 box_depth: float,
                 box_height: float,
                 tilt_angle: float,
                 origin_pos: list = None):
        self.box_width = box_width
        self.box_depth = box_depth
        self.box_height = box_height
        self.tilt_angle = tilt_angle

        if origin_pos is None:
            origin_pos = np.zeros(3)
        self.origin_pos = origin_pos

        # create plane equations
        box_h2 = box_height + box_width * np.tan(tilt_angle)
        self.plane_a = np.array([0., -(box_h2 - box_height), box_width])
        self.d_box = self.plane_a @ self.origin_pos
        self.polytope = self.create()

    def create(self) -> HPolyhedron:
        box_width = self.box_width
        box_depth = self.box_depth
        d_box = self.d_box
        origin_pos = self.origin_pos

        # create a tilted box
        tilted_box_A = np.array([[1, 0., 0.],
                                 [0, 1., 0.],
                                 self.plane_a,
                                 [-1, 0., 0.],
                                 [0, -1., 0.],
                                 [0, 0., -1.]])
        tilted_box_b = np.array([[box_depth/2 + origin_pos[0]] ,
                                 [box_width/2 + origin_pos[1]],
                                 [d_box],
                                 [box_depth/2 - origin_pos[0]],
                                 [box_width/2 - origin_pos[1]],
                                 [0]])
        return HPolyhedron(tilted_box_A, tilted_box_b)

    def get_polytope(self):
        return self.polytope