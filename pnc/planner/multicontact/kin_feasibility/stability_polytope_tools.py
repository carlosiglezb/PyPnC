"""
Stability polytope computation and management for kinematic trajectory planning.

Provides StabilityPolytopeManager, a class that encapsulates computing outer
stability polytopes from a contact sequence using the stabilipy package and
exposes them to trajectory optimizers as (A_stab, b_stab) half-space constraints
of the form  A_stab @ p <= b_stab.
"""
from __future__ import annotations

import numpy as np
import external_source.stabilipy.stabilipy as stab
from pydrake.geometry.optimization import HPolyhedron

from util.util import so3_from_vec_to_vec
from pnc.planner.multicontact.kin_feasibility.fpp_sequencer_tools import get_last_defined_point

# ---------------------------------------------------------------------------
# Default G1 contact geometry constants [metres]
# ---------------------------------------------------------------------------
ANKLE_HEEL_DIST  = 0.06
ANKLE_TOE_DIST   = 0.13
HALF_FOOT_WIDTH  = 0.02
FOOT_HEIGHT      = -0.03
HAND_BOX_SIDE    = 0.02

_DEFAULT_NORMALS = {
    'LF': np.array([0., 0., 1.]),
    'RF': np.array([0., 0., 1.]),
    'LH': np.array([0., -1., 0.]),
    'RH': np.array([0.,  1., 0.]),
}

_GRAVITY_SHAPE_DIRS = [
    np.array([[-1., 0, 0]]).T,
    np.array([[ 1., 0, 0]]).T,
    np.array([[0,  1., 0]]).T,
    np.array([[0, -1., 0]]).T,
    np.array([[0,  0., 1]]).T,
    np.array([[0,  0.,-1]]).T,
]


# ---------------------------------------------------------------------------
# Low-level contact-point expansion helper
# ---------------------------------------------------------------------------

def expand_contact_frame_to_points(frame_name: str,
                                    frame_pos: np.ndarray,
                                    normal: np.ndarray,
                                    ankle_heel_dist: float = ANKLE_HEEL_DIST,
                                    ankle_toe_dist: float  = ANKLE_TOE_DIST,
                                    half_foot_width: float = HALF_FOOT_WIDTH,
                                    foot_height: float     = FOOT_HEIGHT,
                                    hand_box_side: float   = HAND_BOX_SIDE):
    """
    Expand a planning frame position into multiple contact points for stabilipy.

    Feet  (LF/RF): 4 corner points of the sole, rotated to align with the
                   contact normal via SO(3).
    Hands (LH/RH): 4 corner points of a small box on a wall surface (offsets
                   in the x-z plane, normal assumed to be in ±y).

    Returns
    -------
    pos_list     : list[np.ndarray(3,1)]
    normals_list : list[np.ndarray(3,1)]
    """
    pos_list, normals_list = [], []
    pos = np.array(frame_pos).reshape(3, 1)

    # Normalise the contact normal
    n = np.array(normal).reshape(3,)
    mag = np.linalg.norm(n)
    n = n / mag if mag > 1e-6 else np.array([0., 0., 1.])
    n_col = n.reshape(3, 1)

    if frame_name in ('LF', 'RF'):
        rot = so3_from_vec_to_vec(np.array([0., 0., 1.]), n)
        if isinstance(rot, tuple):          # degenerate: vecs are antiparallel
            rot = np.eye(3)
        for off in [
            np.array([[ankle_toe_dist],  [ half_foot_width], [foot_height]]),
            np.array([[ankle_toe_dist],  [-half_foot_width], [foot_height]]),
            np.array([[-ankle_heel_dist],[-half_foot_width], [foot_height]]),
            np.array([[-ankle_heel_dist],[ half_foot_width], [foot_height]]),
        ]:
            pos_list.append(pos + rot @ off)
            normals_list.append(n_col)

    elif frame_name in ('LH', 'RH'):
        for off in [
            np.array([[ hand_box_side], [0.], [ hand_box_side]]),
            np.array([[ hand_box_side], [0.], [-hand_box_side]]),
            np.array([[-hand_box_side], [0.], [-hand_box_side]]),
            np.array([[-hand_box_side], [0.], [ hand_box_side]]),
        ]:
            pos_list.append(pos + off)
            normals_list.append(n_col)

    return pos_list, normals_list


# ---------------------------------------------------------------------------
# StabilityPolytopeManager
# ---------------------------------------------------------------------------

class StabilityPolytopeManager:
    """
    Manages stability polytopes for a multi-phase contact sequence.

    Lifecycle
    ---------
    1. Construct with contact-sequence metadata and robot parameters.
    2. Call :meth:`compute` once the contact-frame positions are known
       (i.e., after ``plan_multistage_iris_seq`` has returned ``safe_pnt_lst``).
    3. Pass the instance to trajectory optimisers; they query polytopes via
       :meth:`get_polytope`.

    Parameters
    ----------
    contact_seqs : list[list[str]]
        Contact frame names per phase (from
        ``get_contact_seq_from_fixed_frames_seq``).
    contact_planes : list[dict[str, np.ndarray]]
        Contact-surface normals per phase (from
        ``get_contact_planes_from_motion_frames_seq``).
    robot_mass : float
        Total robot mass [kg] — used by stabilipy.
    n_phases_out : int, optional
        Number of phases that the optimiser expects.  If larger than
        ``len(contact_seqs)``, the last valid polytope is repeated.
        Defaults to ``len(contact_seqs)``.
    mu : float
        Friction coefficient (default 0.7).
    radius : float
        Stability-sphere radius passed to stabilipy (default 1.0).
    margin : float
        Gravity-envelope scaling margin (default 0.0 — no explicit margin).
    epsilon : float
        Stabilipy convergence tolerance (default 1e-2).
    max_iter : int
        Stabilipy maximum iterations (default 20).
    contact_geom_kwargs : dict
        Optional overrides for foot / hand geometry constants
        (``ankle_heel_dist``, ``ankle_toe_dist``, ``half_foot_width``,
        ``foot_height``, ``hand_box_side``).
    """

    def __init__(self,
                 contact_seqs,
                 contact_planes,
                 robot_mass: float,
                 n_phases_out: int = None,
                 mu: float = 0.7,
                 radius: float = 1.0,
                 margin: float = 0.0,
                 epsilon: float = 1e-2,
                 max_iter: int = 30,
                 foot_force_lim: float = 1.5,
                 hand_force_lim: float = 0.25,
                 **contact_geom_kwargs):
        self._contact_seqs   = contact_seqs
        self._contact_planes = contact_planes
        self._robot_mass     = robot_mass
        self._n_phases_out   = n_phases_out if n_phases_out is not None else len(contact_seqs)
        self._mu             = mu
        self._radius         = radius
        self._margin         = margin
        self._epsilon        = epsilon
        self._max_iter       = max_iter
        self._foot_force_lim = foot_force_lim
        self._hand_force_lim = hand_force_lim
        self._geom_kwargs    = contact_geom_kwargs
        self._polytopes: list | None = None   # computed lazily

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_fixed_frames(cls,
                          fixed_frames,
                          motion_frames_seq,
                          robot_mass: float,
                          **kwargs) -> 'StabilityPolytopeManager':
        """
        Build a manager directly from planner inputs (``fixed_frames`` and
        ``motion_frames_seq``), computing contact sequences and normals
        internally.

        Parameters
        ----------
        fixed_frames : list[list[str]]
            One entry per contact phase — the frames that are fixed.
        motion_frames_seq : MotionFrameSequencer
        robot_mass : float
        **kwargs
            Forwarded to :class:`StabilityPolytopeManager`.
        """
        from pnc.planner.multicontact.kin_feasibility.planner_surface_contact import (
            get_contact_seq_from_fixed_frames_seq,
            get_contact_planes_from_motion_frames_seq,
        )
        contact_seqs   = get_contact_seq_from_fixed_frames_seq(fixed_frames)
        contact_planes = get_contact_planes_from_motion_frames_seq(
            contact_seqs, motion_frames_seq)
        kwargs.setdefault('n_phases_out', len(fixed_frames))
        return cls(contact_seqs, contact_planes, robot_mass, **kwargs)

    # ------------------------------------------------------------------
    # Core compute method
    # ------------------------------------------------------------------

    def compute(self, safe_pnt_lst) -> 'StabilityPolytopeManager':
        """
        Compute all stability polytopes given the safe waypoints.

        Must be called before :meth:`get_polytope`.  Returns ``self`` to
        allow chaining.

        Parameters
        ----------
        safe_pnt_lst : list[dict[str, np.ndarray]]
            Safe waypoints from ``plan_multistage_iris_seq``.  Index *k*
            contains the frame positions at the start of contact phase *k*.
        """
        n_cs = len(self._contact_seqs)
        # ForceConstraint.compute() sums forces across ALL gravity-envelope directions,
        # so n_g redundant copies of the same zero-perturbation make the effective limit
        # n_g× tighter than intended.  When margin=0 all directions are identical zero
        # vectors — collapse to one to keep n_g=1 and ForceConstraint correct.
        if self._margin > 0:
            gravity_envelope = [self._margin * s for s in _GRAVITY_SHAPE_DIRS]
        else:
            gravity_envelope = [np.zeros((3, 1))]

        polytopes = []
        last_valid = None

        for k in range(self._n_phases_out):
            cs_k     = min(k, n_cs - 1)
            cf_names = self._contact_seqs[cs_k]
            planes   = (self._contact_planes[cs_k]
                        if cs_k < len(self._contact_planes) else {})

            contacts = []
            per_limb_contacts = {'LF': [], 'RF': [], 'LH': [], 'RH': []}
            for frame in cf_names:
                frame_pos = get_last_defined_point(safe_pnt_lst[:k + 1], frame)
                if (frame_pos is None
                        or (isinstance(frame_pos, (int, float)) and frame_pos == 0)):
                    frame_pos = safe_pnt_lst[0].get(frame, np.zeros(3))
                frame_pos = np.array(frame_pos).reshape(3,)

                normal = planes.get(frame, _DEFAULT_NORMALS.get(frame,
                                                                  np.array([0., 0., 1.])))
                pts, nrms = expand_contact_frame_to_points(
                    frame, frame_pos, normal, **self._geom_kwargs)
                frame_conts = [stab.Contact(self._mu, p, n) for p, n in zip(pts, nrms)]
                contacts.extend(frame_conts)
                if frame in per_limb_contacts:
                    per_limb_contacts[frame].extend(frame_conts)

            if not contacts:
                print(f"[StabilityPolytopeManager] Phase {k}: "
                      "no contact points — skipping.")
                polytopes.append(last_valid)
                continue

            try:
                polyhedron = stab.StabilityPolygon(
                    self._robot_mass, dimension=3, radius=self._radius)
                polyhedron.contacts = contacts
                # Set envelope first so ForceConstraint.compute() sees the correct n_g.
                polyhedron.gravity_envelope = gravity_envelope
                # Per-limb force limits: one ForceConstraint per foot/hand so that
                # ||f_limb|| <= lim*weight for each limb independently.  Grouping
                # feet together would force ||f_LF+f_RF|| <= weight, which equals
                # the equilibrium support force and leaves no slack for friction.
                for limb in ('LF', 'RF'):
                    if per_limb_contacts[limb] and self._foot_force_lim is not None:
                        polyhedron.addForceConstraint(per_limb_contacts[limb],
                                                      self._foot_force_lim)
                for limb in ('LH', 'RH'):
                    if per_limb_contacts[limb] and self._hand_force_lim is not None:
                        polyhedron.addForceConstraint(per_limb_contacts[limb],
                                                      self._hand_force_lim)
                polyhedron.compute(
                    stab.Mode.best,
                    epsilon=self._epsilon,
                    maxIter=self._max_iter,
                    solver='qhull',
                    record_anim=False,
                    plot_init=False,
                    plot_step=False,
                    plot_final=False,
                )

                p_inner = HPolyhedron(
                    polyhedron.inner.equations[:, :3],
                    -polyhedron.inner.equations[:, -1])

                if not p_inner.IsBounded():
                    print(f"[StabilityPolytopeManager] Phase {k}: "
                          "inner polytope unbounded — skipping.")
                    polytopes.append(last_valid)
                    continue

                A_stab = polyhedron.inner.equations[:, :3].copy()
                b_stab = (-polyhedron.inner.equations[:, -1]).copy()
                last_valid = (A_stab, b_stab)
                polytopes.append(last_valid)

            except Exception as exc:
                print(f"[StabilityPolytopeManager] Phase {k}: "
                      f"computation failed — {exc}")
                polytopes.append(last_valid)

        self._polytopes = polytopes
        return self

    # ------------------------------------------------------------------
    # Query interface used by trajectory optimisers
    # ------------------------------------------------------------------

    @property
    def is_computed(self) -> bool:
        return self._polytopes is not None

    @property
    def n_phases(self) -> int:
        return self._n_phases_out

    def get_polytope(self, phase_idx: int):
        """
        Return ``(A_stab, b_stab)`` for phase *phase_idx*, or ``None`` if
        the polytope could not be computed.

        Indices out of range are clamped to the last entry.
        """
        if not self.is_computed:
            return None
        idx = min(phase_idx, len(self._polytopes) - 1)
        return self._polytopes[idx]

    # ------------------------------------------------------------------
    # Dunder helpers (iteration / indexing convenience)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._polytopes) if self._polytopes is not None else 0

    def __iter__(self):
        return iter(self._polytopes) if self._polytopes is not None else iter([])

    def __getitem__(self, idx):
        if self._polytopes is None:
            return None
        return self._polytopes[idx]

    def __repr__(self) -> str:
        state = "computed" if self.is_computed else "not computed"
        return (f"StabilityPolytopeManager("
                f"n_phases={self._n_phases_out}, robot_mass={self._robot_mass:.1f} kg, "
                f"state={state})")
