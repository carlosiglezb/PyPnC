"""Per-frame environment obstacle inflation for swept-sphere collision avoidance.

Instead of eroding IRIS containment constraints after the fact, each
sphere-approximated robot frame (feet, knees, hands) gets its own view of the
environment in which every obstacle is inflated by that frame's collision
sphere radius (configuration-space dilation: planning the sphere CENTER as a
point in the inflated world guarantees the whole sphere is collision-free).
IRIS regions grown in these per-frame worlds then encode swept-sphere
avoidance directly through the ordinary containment constraints Ax <= b, and
the box sequencer operates on the same free space the optimizer enforces.

Contact faces are exempted from inflation (per face, not per obstacle): a
frame that lands ON a surface would otherwise have its pinned target inside
the inflated obstacle, breaking sequencing and pinning. Exemption anchors are
the frame's starting position plus its motion targets; when a target has a
declared PlannerSurfaceContact, only faces aligned with the declared surface
normal are exempted. This also implements "do not check the floor" naturally:
the floor's top face is exempted for the feet (which stand on it) while its
side faces and the floor as seen by other frames stay inflated.

Inflation is by face offset (b_i + r * ||a_i||), which over-approximates the
exact Minkowski dilation at corners -- conservative in the safe direction.

The torso is a capsule, not a sphere, so it is left un-inflated for now; a
fixed-orientation capsule dilation (extend vertically by the half-length,
then face-offset by the capsule radius: b + L*|a_z| + R*||a||) is the natural
extension once its feasibility across scenarios is verified.
"""
from typing import List

import cvxpy as cp
import numpy as np
from pydrake.geometry.optimization import HPolyhedron


def _distance_to_polytope(p: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
    """Exact euclidean distance from point p to the polytope {x : Ax <= b}."""
    x = cp.Variable(3)
    prob = cp.Problem(cp.Minimize(cp.norm2(x - p)), [A @ x <= b])
    prob.solve(solver='CLARABEL')
    return float(prob.value)


def inflate_hpolyhedron(poly: HPolyhedron, radius: float,
                        exempt_rows=None) -> HPolyhedron:
    """Inflate an H-polyhedron by `radius` via per-face offsets, skipping the
    faces listed in exempt_rows."""
    A = poly.A()
    b = np.array(poly.b(), dtype=float).copy()
    growth = radius * np.linalg.norm(A, axis=1)
    if exempt_rows is not None and len(exempt_rows) > 0:
        growth[np.asarray(list(exempt_rows), dtype=int)] = 0.0
    return HPolyhedron(A, b + growth)


def collect_contact_anchors(starting_pose: dict,
                            motion_frames_seq) -> dict:
    """Exemption anchors per frame: (position, declared contact normal or None).

    Anchors are the frame's starting position (supports the initial stance,
    e.g. feet on the floor) and every motion target, paired with the
    PlannerSurfaceContact normal declared for that frame in the same phase
    (None when the target is a free-space position).
    """
    anchors = {fr: [(np.asarray(p, dtype=float), None)]
               for fr, p in starting_pose.items()}
    contact_lst = motion_frames_seq.get_contact_surfaces()
    for phase_idx, motion_frames in enumerate(motion_frames_seq.get_motion_frames()):
        phase_contacts = contact_lst[phase_idx] if phase_idx < len(contact_lst) else []
        normals = {c.contact_frame_name: np.asarray(c.surface_normal, dtype=float)
                   for c in phase_contacts}
        for fr, pos in motion_frames.items():
            anchors.setdefault(fr, []).append(
                (np.asarray(pos, dtype=float), normals.get(fr)))
    return anchors


class EnvironmentInflator:
    """Builds per-frame obstacle sets, inflated by each frame's sphere radius
    with contact-face exemptions.

    Parameters
    ----------
    obstacles : list[HPolyhedron]
        Environment collision polytopes (shared, un-inflated).
    sca_robot_geometry : SCARobotGeometry
        Source of the collision-sphere radii (frames whose primitive is not a
        sphere, e.g. the torso capsule, keep the un-inflated obstacles).
    dist_tol : float
        Extra reach beyond the sphere radius when deciding whether an anchor
        "touches" an obstacle face (accounts for target/seed placement slop).
        Keep small: any anchor within radius + dist_tol of a face exempts that
        face and gives up the sphere margin there, while a non-exempted anchor
        retains dist - radius of slack inside the inflated world (>= ~5 mm is
        enough for the solvers).
    normal_alignment : float
        Minimum cosine between a face's outward normal and a declared contact
        normal for the face to be exempted (only applied when the anchor has
        a declared normal).
    """

    def __init__(self, obstacles: List[HPolyhedron], sca_robot_geometry,
                 dist_tol: float = 0.005, normal_alignment: float = 0.5):
        self._obstacles = list(obstacles)
        self._sca = sca_robot_geometry
        self._dist_tol = dist_tol
        self._normal_alignment = normal_alignment

    def frame_radius(self, frame_name: str):
        """Collision-sphere radius of the frame, or None for non-sphere frames."""
        if self._sca is None:
            return None
        try:
            if (self._sca.is_link_in_sca_list(frame_name)
                    and self._sca.get_primitive_shape_type(frame_name) == 'sphere'):
                U = self._sca.get_sphere_representation(frame_name)['U']
                return 1.0 / float(U[0, 0])     # U = I / r
        except Exception:
            pass
        return None

    def _exempt_rows(self, poly: HPolyhedron, radius: float, anchor_list):
        A = poly.A()
        b = poly.b()
        n = np.linalg.norm(A, axis=1)
        rows = set()
        for p, normal in anchor_list:
            d = (A @ p - b) / n     # signed normalized distance per face
            # max(d) lower-bounds the true distance (cheap reject); near
            # corners it under-estimates, so confirm with the exact distance
            # before exempting anything
            if d.max() > radius + self._dist_tol:
                continue
            if _distance_to_polytope(p, A, b) > radius + self._dist_tol:
                continue
            cand = np.where((d > -self._dist_tol)
                            & (d <= radius + self._dist_tol))[0]
            for i in cand:
                if normal is not None:
                    nrm = np.linalg.norm(normal)
                    if nrm > 0. and float((A[i] / n[i]) @ (normal / nrm)) \
                            < self._normal_alignment:
                        continue
                rows.add(int(i))
        return rows

    def per_frame_obstacles(self, anchors: dict, verbose: bool = True) -> dict:
        """Returns {frame: [inflated HPolyhedron, ...]} for every anchored frame."""
        out = {}
        for fr, anchor_list in anchors.items():
            radius = self.frame_radius(fr)
            if radius is None:
                out[fr] = self._obstacles
                continue
            inflated = []
            exempt_log = []
            for oi, poly in enumerate(self._obstacles):
                rows = self._exempt_rows(poly, radius, anchor_list)
                if rows:
                    exempt_log.append(f"obs{oi}:{len(rows)}f")
                inflated.append(inflate_hpolyhedron(poly, radius, rows))
            out[fr] = inflated
            if verbose:
                print(f"[EnvInflator] {fr}: obstacles inflated by r={radius:.3f}"
                      + (f" (exempted {', '.join(exempt_log)})" if exempt_log else ""))
        return out
