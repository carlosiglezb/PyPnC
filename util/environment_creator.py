import numpy as np
from pydrake.geometry.optimization import HPolyhedron
# import pydrake.geometry.optimization as mut


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
                                 [0., 1., 0.],
                                 self.plane_a,
                                 [-1, 0., 0.],
                                 [0., -1., 0.],
                                 [0., 0., -1.]])
        tilted_box_b = np.array([[box_depth/2 + origin_pos[0]] ,
                                 [box_width/2 + origin_pos[1]],
                                 [d_box],
                                 [box_depth/2 - origin_pos[0]],
                                 [box_width/2 - origin_pos[1]],
                                 [0.]])
        return HPolyhedron(tilted_box_A, tilted_box_b)

    def get_polytope(self):
        return self.polytope


class TiltedStairs:
    def __init__(self):
        dom_lb = np.array([-1.6, -0.8, -0.])
        dom_ub = np.array([1.6, 0.8, 2.4])

        # stairs parameters
        box_width = 0.35
        box_depth = 0.2
        clearance = 0.05

        # left box
        box_h1_left_vis = 0.25
        box_h2_left_vis = 0.5
        box_h1_left = box_h1_left_vis - clearance
        box_h2_left = box_h2_left_vis - clearance
        lbox_angle = np.arctan((box_h2_left - box_h1_left) / box_width)
        b_lbox_origin = [0.35, box_width/2, (box_h1_left + box_h2_left)/2]
        tilted_left_box = TiltedBox(box_width, box_depth, box_h1_left, lbox_angle, b_lbox_origin)
        tilted_left_step = tilted_left_box.get_polytope()
        b_lbox_origin_vis = [0.35, box_width/2, (box_h1_left_vis + box_h2_left_vis)/2]
        tilted_left_box_vis = TiltedBox(box_width, box_depth, box_h1_left_vis, lbox_angle, b_lbox_origin_vis)
        tilted_left_step_vis = tilted_left_box_vis.get_polytope()

        # right box
        box_h1_right_vis = 0.55
        box_h2_right_vis = 0.8
        box_h1_right = box_h1_right_vis - clearance
        box_h2_right = box_h2_right_vis - clearance
        rbox_angle = -np.arctan((box_h2_right - box_h1_right) / box_width)
        b_rbox_origin = [0.35 + box_depth, -box_width/2, (box_h1_right + box_h2_right)/2]
        tilted_right_box = TiltedBox(box_width, box_depth, box_h1_right, rbox_angle, b_rbox_origin)
        tilted_right_step = tilted_right_box.get_polytope()
        b_rbox_origin_vis = [0.35 + box_depth, -box_width/2, (box_h1_right_vis + box_h2_right_vis)/2]
        tilted_right_box_vis = TiltedBox(box_width, box_depth, box_h1_right_vis, rbox_angle, b_rbox_origin_vis)
        tilted_right_step_vis = tilted_right_box_vis.get_polytope()

        # center box
        box_center_origin = np.array([0.4 + 2.5*box_depth, 0., 0.])
        cbox_lbounds = [box_depth, box_depth, 0.]
        cbox_ubounds = [box_depth, box_depth, 1.0]
        center_box = HPolyhedron.MakeBox(
            np.array(box_center_origin - cbox_lbounds),
            np.array(box_center_origin + cbox_ubounds),
        )
        cbox_lbounds_vis = [box_depth-0.05, box_depth, 0.]
        cbox_ubounds_vis = [box_depth-0.05, box_depth, 1.0]
        center_box_vis = HPolyhedron.MakeBox(
            np.array(box_center_origin - cbox_lbounds_vis),
            np.array(box_center_origin + cbox_ubounds_vis),
        )

        floor = HPolyhedron.MakeBox(
                                np.array([-2, -0.9, -0.5]),
                                np.array([2, 0.9, -0.001]))
        lwall = HPolyhedron.MakeBox(
            np.array([-2, box_width/2 + b_lbox_origin[1], -0.05]),
            np.array([2, box_width/2 + b_lbox_origin[1] + 0.2, 2.5]))
        rwall = HPolyhedron.MakeBox(
            np.array([-2, -(box_width/2 + b_lbox_origin[1] + 0.2), -0.05]),
            np.array([2, -(box_width/2 + b_lbox_origin[1]), 2.5]))

        # create tilted stairs environment
        self.obstacles = [floor,
                     lwall,
                     rwall,
                     tilted_left_step,
                     tilted_right_step,
                     center_box]
        self.obstacles_vis = [floor,
                     lwall,
                     rwall,
                     tilted_left_step_vis,
                     tilted_right_step_vis,
                     center_box_vis]
        self.domain = HPolyhedron.MakeBox(dom_lb, dom_ub)

        self.box_width = box_width
        self.box_depth = box_depth
        self.box_h1_left = box_h1_left
        self.box_h2_left = box_h2_left
        self.box_h1_right = box_h1_right
        self.box_h2_right = box_h2_right
