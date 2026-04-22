import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from pydrake.math import RotationMatrix, RollPitchYaw
# import pydrake.geometry.optimization as mut

class FloatingTriangle:
    """
    Creates a 3D triangular prism (polytope) defined by its three side lengths,
    thickness, position, and orientation.
    """

    def __init__(self,
                 side_lengths: list,
                 thickness: float,
                 pos: np.ndarray = np.zeros(3),
                 R: RotationMatrix = RotationMatrix()):
        self.s1, self.s2, self.s3 = side_lengths
        self.thickness = thickness
        self.pos = pos
        self.R = R.matrix()  # Convert Drake RotationMatrix to numpy

        self.polytope = self._create_polytope()

    def _get_local_vertices(self) -> np.ndarray:
        """Calculates 2D vertices in the XY plane based on side lengths."""
        # Vertex A at origin
        v1 = np.array([0, 0])
        # Vertex B along X-axis
        v2 = np.array([self.s1, 0])
        # Vertex C using Law of Cosines: s3^2 = s1^2 + s2^2 - 2*s1*s2*cos(theta)
        cos_theta = (self.s1 ** 2 + self.s2 ** 2 - self.s3 ** 2) / (2 * self.s1 * self.s2)
        sin_theta = np.sqrt(max(0, 1 - cos_theta ** 2))
        v3 = np.array([self.s2 * cos_theta, self.s2 * sin_theta])

        return np.array([v1, v2, v3])

    def _create_polytope(self) -> HPolyhedron:
        vertices_2d = self._get_local_vertices()

        # 1. Define local half-space constraints (Ax <= b) for the 2D triangle
        # For each edge (v_i, v_j), the normal pointing outward is n
        A_local = []
        b_local = []

        edges = [(0, 1), (1, 2), (2, 0)]
        # Geometric center for inside-testing (simple average of vertices)
        center_2d = np.mean(vertices_2d, axis=0)

        for i, j in edges:
            p1, p2 = vertices_2d[i], vertices_2d[j]
            edge = p2 - p1
            # Normal vector (perpendicular to edge)
            normal = np.array([-edge[1], edge[0]])
            # Ensure normal points outward
            if normal @ (center_2d - p1) > 0:
                normal = -normal

            # Normalize and add to A, b (normal * x <= normal * p1)
            unit_normal = normal / np.linalg.norm(normal)
            A_local.append([unit_normal[0], unit_normal[1], 0.0])
            b_local.append(unit_normal @ p1)

        # 2. Add Top and Bottom constraints for thickness (Z-axis)
        # z <= thickness/2 and -z <= thickness/2
        A_local.append([0, 0, 1])
        b_local.append(self.thickness / 2.0)
        A_local.append([0, 0, -1])
        b_local.append(self.thickness / 2.0)

        A_mat = np.array(A_local)
        b_vec = np.array(b_local).reshape(-1, 1)

        # 3. Transform to World Frame
        # If Ax_local <= b and x_world = R * x_local + p
        # then x_local = R^T * (x_world - p)
        # Substituting: A * R^T * (x_world - p) <= b
        # A * R^T * x_world <= b + A * R^T * p
        A_world = A_mat @ self.R.T
        b_world = b_vec + (A_world @ self.pos).reshape(-1, 1)

        return HPolyhedron(A_world, b_world)

    def get_polytope(self) -> HPolyhedron:
        return self.polytope

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

class HoleInWallObstructed:
    def __init__(self, door_pos):
        door_width = np.array([0.03, 0., 0.])

        dom_lb = np.array([-1.6, -0.8, -0.])
        dom_ub = np.array([1.6, 0.8, 2.1])

        # Domain
        # domain_lbody = HPolyhedron.MakeBox(dom_lbody_lb, dom_lbody_ub)
        # domain_ubody = HPolyhedron.MakeBox(dom_ubody_lb, dom_ubody_ub)

        # Obstacles
        floor = HPolyhedron.MakeBox(
            np.array([-2, -0.9, -0.1]) + door_pos + door_width,
            np.array([2, 0.9, -0.001]) + door_pos + door_width)
        knee_knocker_base = HPolyhedron.MakeBox(
            np.array([-0.04, -0.9, 0.0]) + door_pos + door_width,
            np.array([0.04, 0.9, 0.4]) + door_pos + door_width)
        knee_knocker_lwall = HPolyhedron.MakeBox(
            np.array([-0.025, 0.9 - 0.518 - 0.025, 0.0]) + door_pos + door_width,
            np.array([0.025, 0.9, 2.2]) + door_pos + door_width)
        knee_knocker_rwall = HPolyhedron.MakeBox(
            np.array([-0.025, -0.9, 0.0]) + door_pos + door_width,
            np.array([0.025, -(0.9 - 0.518 - 0.025), 2.2]) + door_pos + door_width)
        knee_knocker_top = HPolyhedron.MakeBox(
            np.array([-0.025, -0.9, 1.40]) + door_pos + door_width,
            np.array([0.025, 0.9, 2.25]) + door_pos + door_width)

        # Additional obstacle covering hole
        sharp_triangle_sides = [1.0, 0.7, 0.4453]   # for contact [0.34, -0.15, 1.15]
        triangle_sides = [1.0, 0.6, 0.46301]    # for contact [0.34, -0.25, 1.15]
        triangle_pos = np.array([door_pos[0] + door_width[0], -0.765/1.8, 0.2])
        triangle_rot = RotationMatrix(RollPitchYaw([0, -np.pi/2, 0]))
        right_triangle = FloatingTriangle(triangle_sides, door_width[0], triangle_pos, triangle_rot)
        bottom_triangle_rot = RotationMatrix(RollPitchYaw([np.pi/2, 0., -np.pi/2]))
        bottom_triangle_pos = np.array([door_pos[0] + door_width[0], 0.75, 0.25])
        bottom_triangle = FloatingTriangle(sharp_triangle_sides, door_width[0], bottom_triangle_pos, bottom_triangle_rot)
        self.obstacles = [floor,
                          knee_knocker_base,
                          knee_knocker_lwall,
                          knee_knocker_rwall,
                          knee_knocker_top,
                          right_triangle.get_polytope(),
                          bottom_triangle.get_polytope()
                          ]
        self.domain = HPolyhedron.MakeBox(dom_lb, dom_ub)
        self.door_pos = door_pos