import pinocchio as pin
import numpy as np


# ---------------------------------------------------------------------------
# True-size (alpha=1) disjointness checks between primitive pairs.
#
# The DCOL self-collision constraints built in
# optimize_multiple_bezier_iris_casadi (see casadi_ocp_constraints/
# casadi_ocp_functions.py) find the minimal uniform scale factor alpha>=0
# about each shape's own center at which the two (grown/shrunk) shapes touch,
# and require alpha >= 1. Because that scaling is a monotonic dilation about
# a fixed center, alpha* >= 1 holds if and only if the two shapes at their
# TRUE size (alpha=1) do not overlap -- so a plain distance/overlap test at
# the candidate positions is an exact stand-in for the alpha>=1 constraint,
# without needing to solve anything. All the constraints in that function
# additionally use Q=eye(3) (world-axis-aligned primitives, centered exactly
# at the frame's tracked position, no local placement offset) -- these
# helpers mirror that convention exactly.
# ---------------------------------------------------------------------------

def _sphere_sphere_disjoint(c1, r1, c2, r2) -> bool:
    return np.linalg.norm(c1 - c2) >= (r1 + r2)


def _box_sphere_disjoint(box_center, half_extent, sph_center, sph_radius) -> bool:
    offset = sph_center - box_center
    closest = np.clip(offset, -half_extent, half_extent)
    return np.linalg.norm(offset - closest) >= sph_radius


def _capsule_sphere_disjoint(cap_center, R, L, sph_center, sph_radius) -> bool:
    # capsule axis is world z (matches the Q=eye(3) convention above)
    half_length = L / 2.0
    t = np.clip(sph_center[2] - cap_center[2], -half_length, half_length)
    closest = cap_center + np.array([0., 0., t])
    return np.linalg.norm(sph_center - closest) >= (R + sph_radius)


def _box_box_disjoint(c1, h1, c2, h2) -> bool:
    return bool(np.any(np.abs(c1 - c2) > (h1 + h2)))


class SCARobotGeometry:
    def __init__(self, robot_model_path: str,
                 urdf_path: str,
                 plan_to_model_frames: dict[str: str]):
        # Load the robot and geometry model and collision models from urdf
        robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path, robot_model_path, root_joint=pin.JointModelFreeFlyer())
        geom_model = pin.buildGeomFromUrdf(robot.model,
                                           urdf_path,
                                           pin.GeometryType.COLLISION)
        geom_model.addAllCollisionPairs()

        sca_link_names = plan_to_model_frames.values()

        # extract description of primitives for each link in the sca_link_names
        self._geometry_primitives = {}
        for lnk in sca_link_names:
            for gm in geom_model.geometryObjects:
                # TODO figure out what to do with cascaded collision bodies
                # This will currently only save the settings of the last collision item
                if lnk in gm.name:
                    halfspace_params = {}
                    R = gm.placement.rotation
                    origin = gm.placement.translation.reshape(-1, 1)
                    if gm.meshPath == 'BOX':
                        box_half_side = gm.geometry.halfSide.reshape(-1, 1)
                        halfspace_params['A'] = np.vstack((np.eye(3), -np.eye(3)))
                        halfspace_params['b'] = np.vstack((box_half_side, box_half_side))
                        halfspace_params['polytope_origin'] = origin
                        halfspace_params['polytope_rotation'] = R
                    elif gm.meshPath == 'SPHERE':
                        sphere_radius = gm.geometry.radius
                        halfspace_params['U'] = 1./sphere_radius * np.eye(3)
                    elif gm.meshPath == 'CAPSULE':
                        halfspace_params['R'] = gm.geometry.radius
                        halfspace_params['L'] = 2 * gm.geometry.halfLength
                    else:
                        raise NotImplementedError("Only box primitives are currently supported")
                    self._geometry_primitives[lnk] = halfspace_params

        self.geom_model = geom_model
        self._plan_to_model_frames = plan_to_model_frames

    def get_box_representation(self, link_name: str) -> dict[str, np.array]:
        return self._geometry_primitives[self._plan_to_model_frames.get(link_name)]

    def get_sphere_representation(self, link_name: str) -> dict[str, np.array]:
        return self._geometry_primitives[self._plan_to_model_frames.get(link_name)]

    def get_capsule_representation(self, link_name: str) -> dict[str, np.array]:
        return self._geometry_primitives[self._plan_to_model_frames.get(link_name)]

    def get_primitive_shape_type(self, link_name: str) -> str:
        if 'A' in self._geometry_primitives[self._plan_to_model_frames.get(link_name)]:
            return 'box'
        elif 'U' in self._geometry_primitives[self._plan_to_model_frames.get(link_name)]:
            return 'sphere'
        elif 'R' in self._geometry_primitives[self._plan_to_model_frames.get(link_name)]:
            return 'capsule'
        return 'unknown'

    def is_link_in_sca_list(self, link_name) -> bool:
        return self._plan_to_model_frames[link_name] in self._geometry_primitives.keys()

    def get_self_collision_pairs(self, frame_list: list) -> list:
        """Frame-index pairs checked for self-collision.

        Mirrors the DCOL constraint set built by
        optimize_multiple_bezier_iris_casadi: every SCA-listed link paired
        with the torso, plus the feet-to-feet and RF-to-L_knee pairs added
        unconditionally there. Shared by that constraint builder and
        is_trajectory_self_collision_free so the two can't drift apart.
        """
        torso_idx = frame_list.index('torso')
        pairs = [(torso_idx, i) for i, fr in enumerate(frame_list)
                 if fr != 'torso' and self.is_link_in_sca_list(fr)]
        for fr_a, fr_b in (('LF', 'RF'), ('RF', 'L_knee')):
            if fr_a in frame_list and fr_b in frame_list:
                pairs.append((frame_list.index(fr_a), frame_list.index(fr_b)))
        return pairs

    def is_pair_disjoint(self, frame1: str, pos1: np.array, frame2: str, pos2: np.array) -> bool:
        """True if the true-size primitives of frame1/frame2, centered at
        pos1/pos2, do not overlap (see the module-level docstring above)."""
        type1 = self.get_primitive_shape_type(frame1)
        type2 = self.get_primitive_shape_type(frame2)
        shape_types = {type1, type2}

        if shape_types == {'sphere'}:
            r1 = 1. / self.get_sphere_representation(frame1)['U'][0, 0]
            r2 = 1. / self.get_sphere_representation(frame2)['U'][0, 0]
            return _sphere_sphere_disjoint(pos1, r1, pos2, r2)
        if shape_types == {'box', 'sphere'}:
            box_fr, sph_fr = (frame1, frame2) if type1 == 'box' else (frame2, frame1)
            box_pos, sph_pos = (pos1, pos2) if type1 == 'box' else (pos2, pos1)
            half_extent = self.get_box_representation(box_fr)['b'][:3, 0]
            radius = 1. / self.get_sphere_representation(sph_fr)['U'][0, 0]
            return _box_sphere_disjoint(box_pos, half_extent, sph_pos, radius)
        if shape_types == {'capsule', 'sphere'}:
            cap_fr, sph_fr = (frame1, frame2) if type1 == 'capsule' else (frame2, frame1)
            cap_pos, sph_pos = (pos1, pos2) if type1 == 'capsule' else (pos2, pos1)
            R = self.get_capsule_representation(cap_fr)['R']
            L = self.get_capsule_representation(cap_fr)['L']
            radius = 1. / self.get_sphere_representation(sph_fr)['U'][0, 0]
            return _capsule_sphere_disjoint(cap_pos, R, L, sph_pos, radius)
        if shape_types == {'box'}:
            h1 = self.get_box_representation(frame1)['b'][:3, 0]
            h2 = self.get_box_representation(frame2)['b'][:3, 0]
            return _box_box_disjoint(pos1, h1, pos2, h2)
        raise NotImplementedError(
            f"Self-collision check not implemented for shape pair {(type1, type2)}.")

    def is_trajectory_self_collision_free(self, points: dict, frame_list: list,
                                          num_iris_tot: int) -> bool:
        """Cheap feasibility check on a candidate cvxpy/casadi solution.

        Evaluates the same alpha>=1 DCOL self-collision condition enforced in
        optimize_multiple_bezier_iris_casadi, but directly via closed-form
        distance checks at each Bezier position control point (see
        is_pair_disjoint) instead of constructing/solving the DCOL casadi
        callbacks. Returns True only if every checked pair, at every control
        point of every box, is already collision-free -- i.e. the (much more
        expensive) casadi SCA refinement would not change the trajectory.
        """
        for idx1, idx2 in self.get_self_collision_pairs(frame_list):
            fr1, fr2 = frame_list[idx1], frame_list[idx2]
            for box in range(num_iris_tot):
                pos1 = points[idx1 * num_iris_tot + box][0]
                pos2 = points[idx2 * num_iris_tot + box][0]
                pos1 = pos1.value if hasattr(pos1, 'value') else pos1
                pos2 = pos2.value if hasattr(pos2, 'value') else pos2
                for row in range(pos1.shape[0]):
                    if not self.is_pair_disjoint(fr1, pos1[row], fr2, pos2[row]):
                        return False
        return True

    def get_shape_origin(self, link_name: str) -> np.array:
        return self._geometry_primitives[self._plan_to_model_frames.get(link_name)].get('polytope_origin', None)

    def get_shape_rotation(self, link_name: str) -> np.array:
        return self._geometry_primitives[self._plan_to_model_frames.get(link_name)].get('polytope_rotation', None)