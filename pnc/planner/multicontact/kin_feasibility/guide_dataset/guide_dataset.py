"""Offline guide dataset generation and GPU-resident runtime Bezier evaluation.

Workflow
--------
Offline (run once, before training):

    from extensions.g1_residual_control.guide_dataset import generate_guide_dataset
    generate_guide_dataset(planner_builder_fn, xy_offset_bounds,
                           save_path="guides.npz")

Online (env reset, zero solve overhead):

    dataset = GuideDataset("guides.npz", device="cuda")
    # at reset:
    ctrl_pts, trans_times, T_per_env = dataset.sample(num_envs, p_torso_actual)
    # each policy step:
    targets = dataset.query_targets(ctrl_pts, trans_times, t_elapsed)

Storage format
--------------
``control_points``: (n_guides, n_frames, n_segments, degree+1, 3) float32
    Bezier control points for each guide, frame, and segment.
``transition_times``: (n_guides, n_segments+1) float32
    Segment breakpoints [t0=0, t1, ..., tN=T_total], shared across frames per guide.

Runtime evaluation
------------------
``query_targets`` finds the correct Bezier segment for the current ``t_elapsed``
via vectorised comparison, computes the Bernstein basis in closed form, and
evaluates the curve — entirely as GPU tensor ops with no CPU round-trip.
This gives exact-time evaluation at arbitrary t rather than nearest-waypoint
quantisation.

Nearest-neighbour selection
---------------------------
For each environment the guide whose nominal XY torso position is closest
to the actual spawn torso is selected.  With a sufficiently dense dataset
the NN guide is close enough that the residual policy handles any remaining
offset, so no rigid shift is applied after selection.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Name mappings
# ---------------------------------------------------------------------------

# Planner frame name → URDF body name (matches plan_to_model_frames in cfree_dyn_y_robustness.py)
FRAME_TO_BODY: dict[str, str] = OrderedDict({
    "torso":  "torso_primitive_shape",
    "LF":     "left_ankle_roll_link",
    "RF":     "right_ankle_roll_link",
    "L_knee": "left_knee_link",
    "R_knee": "right_knee_link",
    "LH":     "left_rubber_hand",
    "RH":     "right_rubber_hand",
})
BODY_TO_FRAME: dict[str, str] = {v: k for k, v in FRAME_TO_BODY.items()}

# Frames in contact with the environment: their guides are never shifted.
_CONTACT_FRAMES: frozenset[str] = frozenset({"LH", "RH"})


# ---------------------------------------------------------------------------
# Offline generation
# ---------------------------------------------------------------------------

def _extract_bezier_control_points(
    paths: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract Bezier control points and transition times from a planner path list.

    Validates that all frames share the same number of segments, the same
    Bezier degree, and the same transition times (standard for joint MFPP).

    Parameters
    ----------
    paths : list[CompositeBezierCurve]
        One composite Bezier curve per frame, as returned by the planner.

    Returns
    -------
    ctrl_pts : np.ndarray, shape (n_frames, n_segments, degree+1, 3)
    transition_times : np.ndarray, shape (n_segments+1,)
    """
    n_frames = len(paths)
    ref = paths[0]
    n_segments = ref.N
    degree = ref.beziers[0].h
    transition_times = np.array(ref.transition_times, dtype=np.float32)

    for i, path in enumerate(paths[1:], start=1):
        if path.N != n_segments:
            raise ValueError(f"Frame {i} has {path.N} segments, expected {n_segments}")

    ctrl_pts = np.zeros((n_frames, n_segments, degree + 1, 3), dtype=np.float32)
    for i, path in enumerate(paths):
        for s, bez in enumerate(path.beziers):
            if bez.h != degree:
                raise ValueError(
                    f"Frame {i}, segment {s}: degree {bez.h} differs from expected {degree}"
                )
            ctrl_pts[i, s] = bez.points.astype(np.float32)

    return ctrl_pts, transition_times


def generate_guide_dataset(
    planner_builder_fn: Callable[[np.ndarray], Any],
    xy_offset_bounds: np.ndarray,
    T_min: float = 2.0,
    T_max: float = 3.0,
    n_alpha: int = 20,
    n_w_rigid: int = 10,
    w_rigid_mid_fixed: float = 0.0,
    seed: int = 42,
    save_path: str = "guide_dataset.npz",
) -> None:
    """Generate and save a dataset of kinematic guides as Bezier control points.

    Samples ``n_alpha`` alpha vectors and ``n_w_rigid`` w_rigid vectors
    independently at random (Latin-Hypercube-like uniform coverage), then
    solves the two-stage SOCP planner for each of the
    ``n_alpha × n_w_rigid`` (alpha, w_rigid) combinations.

    Each guide independently draws:
      - a traversal duration T_i ~ U(T_min, T_max) for speed diversity, and
      - an XY offset (dx, dy) ~ U(-xy_offset_bounds, +xy_offset_bounds).

    The planner's CompositeBezierCurve output is stored directly as control
    points rather than sampled waypoints, enabling exact-time evaluation.

    Parameter ranges
    ----------------
    alpha   : (n_alpha, 3) — entries [0,1] from U(0,1); entry [2] from U(0,0.2)
    w_rigid : (n_w_rigid, 3) — entries [0,2] from U(0,1); entry [1] fixed

    Parameters
    ----------
    planner_builder_fn : Callable[[np.ndarray], Any]
        Called as ``planner_builder_fn(xy_offset)`` where ``xy_offset`` is a
        shape-(2,) array [dx, dy].  Must return a 2-tuple
        ``(ik_cfree_planner, p_init)`` where ``ik_cfree_planner`` is a fully
        initialised :class:`IKCFreePlanner` (with its inner
        ``LocomanipulationFramePlanner`` set) and ``p_init`` is a dict mapping
        planner-frame names to their 3-D starting positions.
    xy_offset_bounds : np.ndarray, shape (2,)
        Symmetric bounds [bound_x, bound_y].
    T_min, T_max : float
        Per-phase traversal duration range (seconds).
    n_alpha, n_w_rigid : int
        Number of alpha / w_rigid samples.
    w_rigid_mid_fixed : float
        Fixed value for w_rigid index 1.
    seed : int
        RNG seed for reproducibility.
    save_path : str
        Destination ``.npz`` file.
    """
    rng = np.random.default_rng(seed)

    # --- Parameter grids --------------------------------------------------
    alpha_grid = np.zeros((n_alpha, 3), dtype=np.float64)
    alpha_grid[:, 0] = rng.uniform(0.0, 1.0, n_alpha)
    alpha_grid[:, 1] = rng.uniform(0.0, 1.0, n_alpha)
    alpha_grid[:, 2] = rng.uniform(0.0, 0.2, n_alpha)

    w_rigid_grid = np.zeros((n_w_rigid, 3), dtype=np.float64)
    w_rigid_grid[:, 0] = rng.uniform(0.0, 1.0, n_w_rigid)
    w_rigid_grid[:, 1] = w_rigid_mid_fixed
    w_rigid_grid[:, 2] = rng.uniform(0.0, 1.0, n_w_rigid)

    # --- Metadata ---------------------------------------------------------
    frame_names: list[str] = list(FRAME_TO_BODY.keys())
    body_names:  list[str] = [FRAME_TO_BODY.get(fn, fn) for fn in frame_names]
    hand_mask = np.array([fn in _CONTACT_FRAMES for fn in frame_names], dtype=bool)

    # --- Generation loop --------------------------------------------------
    ctrl_pts_list:      list[np.ndarray] = []
    trans_times_list:   list[np.ndarray] = []
    alpha_used:         list[np.ndarray] = []
    w_rigid_used:       list[np.ndarray] = []
    T_used:             list[float] = []
    torso_nominal_used: list[np.ndarray] = []

    total = n_alpha * n_w_rigid

    # Pre-flatten the grid into a queue; extend with fresh random samples on
    # failure so the final dataset always contains exactly `total` guides.
    param_queue: list[tuple[np.ndarray, np.ndarray]] = [
        (alpha_grid[i], w_rigid_grid[j])
        for i in range(n_alpha)
        for j in range(n_w_rigid)
    ]
    attempt = 0
    skipped = 0
    while len(ctrl_pts_list) < total:
        if attempt < len(param_queue):
            alpha, w_rigid = param_queue[attempt]
        else:
            # Draw a fresh random replacement for a skipped sample.
            alpha = np.zeros(3, dtype=np.float64)
            alpha[0] = rng.uniform(0.0, 1.0)
            alpha[1] = rng.uniform(0.0, 1.0)
            alpha[2] = rng.uniform(0.0, 0.2)
            w_rigid = np.zeros(3, dtype=np.float64)
            w_rigid[0] = rng.uniform(0.0, 1.0)
            w_rigid[1] = w_rigid_mid_fixed
            w_rigid[2] = rng.uniform(0.0, 1.0)
        attempt += 1

        count = len(ctrl_pts_list) + 1
        T_i = float(rng.uniform(T_min, T_max))
        xy_offset = rng.uniform(-xy_offset_bounds, xy_offset_bounds)
        print(f"[{count:3d}/{total}]  alpha={np.round(alpha, 3)}  "
              f"w_rigid={np.round(w_rigid, 3)}  T={T_i:.2f}s  "
              f"offset=[{xy_offset[0]:+.3f}, {xy_offset[1]:+.3f}]", end="  ")
        try:
            ik_cfree_planner, p_init = planner_builder_fn(xy_offset)
            ik_cfree_planner.planner.plan_iris(
                p_init, T_i, alpha.tolist(), w_rigid, w_rigid,
                b_final_vel_constraint=True, verbose=False,
            )
            ctrl_pts, trans_times = _extract_bezier_control_points(ik_cfree_planner.planner.path)
            T_i_total = float(trans_times[-1])
            ctrl_pts_list.append(ctrl_pts)
            trans_times_list.append(trans_times)
            alpha_used.append(alpha.copy())
            w_rigid_used.append(w_rigid.copy())
            T_used.append(T_i_total)
            torso_nominal_used.append(p_init['torso'].astype(np.float32))
            print("ok")
        except Exception as exc:
            skipped += 1
            print(f"SKIPPED ({exc}) — drawing replacement [{skipped} skipped so far]")

    if not ctrl_pts_list:
        raise RuntimeError("Every guide generation attempt failed. "
                           "Check planner, IRIS regions, and robot model paths.")

    ctrl_pts_arr    = np.stack(ctrl_pts_list,    axis=0)  # (n_guides, F, S, D+1, 3)
    trans_times_arr = np.stack(trans_times_list, axis=0)  # (n_guides, S+1)
    T_plan_arr      = np.array(T_used, dtype=np.float32)

    np.savez(
        save_path,
        control_points=ctrl_pts_arr.astype(np.float32),
        transition_times=trans_times_arr.astype(np.float32),
        alpha_grid=np.array(alpha_used, dtype=np.float32),
        w_rigid_grid=np.array(w_rigid_used, dtype=np.float32),
        p_init_nominal_torso=np.array(torso_nominal_used, dtype=np.float32),
        T_plan=T_plan_arr.mean(),
        T_plan_arr=T_plan_arr,
        frame_names=np.array(frame_names),
        body_names=np.array(body_names),
        hand_mask=hand_mask,
    )
    n = len(ctrl_pts_list)
    n_frames, n_segments, degree_plus_1, _ = ctrl_pts_arr.shape[1:]
    mb = ctrl_pts_arr.nbytes / 1e6
    print(f"\nSaved {n}/{total} guides → {save_path}  "
          f"({skipped} infeasible skipped, {attempt} attempts total, "
          f"n_frames={n_frames}, n_segments={n_segments}, "
          f"degree={degree_plus_1 - 1}, {mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Runtime dataset
# ---------------------------------------------------------------------------

class GuideDataset:
    """GPU-resident guide library with O(1) nearest-neighbour sampling at env reset.

    The full guide tensor is loaded onto the GPU once and stays there for the
    entire training run.  All operations (nearest-neighbour selection,
    Bezier evaluation) are vectorised GPU tensor operations with no CPU
    round-trips.

    Guide trajectories are stored as Bezier control points and evaluated at
    exact episode times via vectorised Bernstein polynomials, avoiding the
    temporal quantisation of nearest-waypoint lookup.

    Attributes
    ----------
    ctrl_pts : torch.Tensor, shape (n_guides, n_frames, n_segments, degree+1, 3)
        Bezier control points for all guides.
    transition_times : torch.Tensor, shape (n_guides, n_segments+1)
        Segment breakpoints [0, t1, ..., T_total] per guide.
    p_init_nominal_torso_arr : torch.Tensor, shape (n_guides, 3)
        Per-guide nominal torso position for nearest-neighbour selection.
    q_phase_boundary : torch.Tensor or None, shape (n_guides, n_phases, n_joints)
        Per-guide IK joint angles (excluding floating-base, i.e. q[7:]) at the
        start of each phase.  ``n_phases == n_segments``.  ``None`` when the
        dataset was generated without the augmentation step
        (run ``augment_guide_dataset_ik.py`` to populate).
        Usage in ``reset_guide_assignment``::
            q_init = dataset.q_phase_boundary[env_ids, phase_idx]  # (E, n_joints)
            robot.write_joint_pos_to_sim(q_init, env_ids=env_ids)
    T_plan : float
        Mean guide duration in seconds.
    frame_names : list[str]
        Planner frame names in dataset index order.
    body_names : list[str]
        Corresponding URDF body names.
    body_name_to_idx : dict[str, int]
        Lookup from URDF body name to frame axis index.
    """

    def __init__(self, path: str, device: str = "cuda"):
        data = np.load(path, allow_pickle=True)

        # Control points: (n_guides, n_frames, n_segments, degree+1, 3)
        self.ctrl_pts = torch.from_numpy(data["control_points"]).float().to(device)
        self.n_guides, self.n_frames, self.n_segments, self.degree_plus_1, _ = (
            self.ctrl_pts.shape
        )
        self.degree = self.degree_plus_1 - 1

        # Transition times: (n_guides, n_segments+1)
        self.transition_times = (
            torch.from_numpy(data["transition_times"]).float().to(device)
        )

        # Precompute Bernstein basis coefficients once on the target device.
        # B_n(t) = binom(D, n) * t^n * (1-t)^(D-n)
        n_vec = torch.arange(self.degree + 1, dtype=torch.float32, device=device)
        log_binom = (
            torch.lgamma(torch.tensor(float(self.degree + 1), device=device))
            - torch.lgamma(n_vec + 1.0)
            - torch.lgamma(torch.tensor(float(self.degree + 1), device=device) - n_vec)
        )
        self._binom_coeffs = log_binom.exp()   # (D+1,)
        self._n_vec        = n_vec             # (D+1,)
        self._D_minus_n    = (self.degree - n_vec)  # (D+1,)

        # Precompute degree-(D-1) Bernstein basis coefficients for velocity
        # (analytical derivative of the position Bezier curve).
        D_d = self.degree - 1  # degree of derivative curve
        n_vec_d = torch.arange(D_d + 1, dtype=torch.float32, device=device)
        log_binom_d = (
            torch.lgamma(torch.tensor(float(D_d + 1), device=device))
            - torch.lgamma(n_vec_d + 1.0)
            - torch.lgamma(torch.tensor(float(D_d + 1), device=device) - n_vec_d)
        )
        self._binom_coeffs_d = log_binom_d.exp()   # (D,)
        self._n_vec_d        = n_vec_d             # (D,)
        self._D_minus_n_d    = D_d - n_vec_d       # (D,)

        # Per-guide nominal torso for NN selection
        torso_np = data["p_init_nominal_torso"]
        if torso_np.ndim == 1:
            torso_np = np.broadcast_to(torso_np, (self.n_guides, 3)).copy()
        self.p_init_nominal_torso_arr = (
            torch.from_numpy(torso_np).float().to(device)
        )

        self.T_plan = float(data["T_plan"])
        if "T_plan_arr" in data:
            self.T_plan_arr = torch.from_numpy(data["T_plan_arr"]).float().to(device)
        else:
            # Derive from transition times when not explicitly stored
            self.T_plan_arr = self.transition_times[:, -1]

        frame_names: list[str] = data["frame_names"].tolist()
        body_names:  list[str] = data["body_names"].tolist()

        self.frame_names = frame_names
        self.body_names  = body_names
        self.frame_name_to_idx: dict[str, int] = {n: i for i, n in enumerate(frame_names)}
        self.body_name_to_idx:  dict[str, int] = {n: i for i, n in enumerate(body_names)}

        if "hand_mask" in data:
            self.hand_mask = torch.from_numpy(data["hand_mask"]).to(device)
        else:
            self.hand_mask = None

        # Per-phase IK joint angles: (n_guides, n_phases, n_joints).
        # Populated by augment_guide_dataset_ik.py; None if the dataset predates
        # the augmentation step.
        if "q_phase_boundary" in data:
            self.q_phase_boundary = (
                torch.from_numpy(data["q_phase_boundary"]).float().to(device)
            )
        else:
            self.q_phase_boundary = None

        self.device = device

    # ------------------------------------------------------------------
    # Env-reset: nearest-neighbour sample
    # ------------------------------------------------------------------

    def sample(
        self,
        num_envs: int,
        p_init_torso_actual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Select guides for a batch of environments via nearest-neighbour lookup.

        For each environment the guide whose nominal XY torso position is
        closest to the actual spawn torso (L2 in XY) is selected.  The
        search is computed entirely on GPU as a batched squared-distance
        matrix followed by argmin:

            dist_sq[e, i] = ||p_actual_xy[e] - p_nominal_xy[i]||²
            idx[e]        = argmin_i  dist_sq[e, i]

        Parameters
        ----------
        num_envs : int
            Number of environments to sample for.
        p_init_torso_actual : torch.Tensor, shape (num_envs, 3)
            Actual world-frame torso position at env spawn, on GPU.

        Returns
        -------
        ctrl_pts : torch.Tensor, shape (num_envs, n_frames, n_segments, degree+1, 3)
        transition_times : torch.Tensor, shape (num_envs, n_segments+1)
        T_per_env : torch.Tensor, shape (num_envs,)
        q_phase_boundary : torch.Tensor or None, shape (num_envs, n_phases, n_joints)
            Per-phase IK joint angles for warm-starting ``reset_guide_assignment``.
            ``None`` when the dataset was not augmented with the IK step.
            Use as::
                q_init = q_phase_boundary[env_ids, phase_idx]
                robot.write_joint_pos_to_sim(q_init, env_ids=env_ids)
        """
        p_actual_xy  = p_init_torso_actual[:, :2]               # (E, 2)
        p_nominal_xy = self.p_init_nominal_torso_arr[:, :2]     # (G, 2)
        dist_sq = (
            p_actual_xy[:, None, :] - p_nominal_xy[None, :, :]  # (E, G, 2)
        ).pow(2).sum(-1)                                         # (E, G)
        idx = dist_sq.argmin(dim=1)                              # (E,)

        ctrl_pts    = self.ctrl_pts[idx].clone()           # (E, F, S, D+1, 3)
        trans_times = self.transition_times[idx].clone()   # (E, S+1)
        T_per_env   = self.T_plan_arr[idx]                 # (E,)
        q_phase_bnd = (
            self.q_phase_boundary[idx].clone()             # (E, N_phases, n_joints)
            if self.q_phase_boundary is not None else None
        )

        return ctrl_pts, trans_times, T_per_env, q_phase_bnd

    # ------------------------------------------------------------------
    # Per-step: Bezier evaluation at exact episode time
    # ------------------------------------------------------------------

    def query_targets(
        self,
        ctrl_pts: torch.Tensor,
        transition_times: torch.Tensor,
        t_elapsed: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the Bezier guide at the current episode time.

        Finds the correct segment via vectorised comparison, computes the
        Bernstein basis, and evaluates the curve in closed form.  All ops
        stay on the GPU — no CPU round-trips.

        Parameters
        ----------
        ctrl_pts : torch.Tensor, shape (num_envs, n_frames, n_segments, degree+1, 3)
        transition_times : torch.Tensor, shape (num_envs, n_segments+1)
            Segment breakpoints [t0=0, t1, ..., tN=T_total] per env.
        t_elapsed : torch.Tensor, shape (num_envs,)
            Seconds since episode start.

        Returns
        -------
        torch.Tensor, shape (num_envs, n_frames, 3)
        """
        E  = ctrl_pts.shape[0]
        S  = self.n_segments
        D1 = self.degree_plus_1
        F  = self.n_frames
        device = ctrl_pts.device

        # Clamp t to the valid curve domain
        t_clamped = torch.max(
            torch.min(t_elapsed, transition_times[:, -1]),
            transition_times[:, 0],
        )  # (E,)

        # 1. Segment selection: last segment start time ≤ t_clamped
        #    transition_times[:, :-1] are the S segment start times.
        t_exp   = t_clamped.unsqueeze(1)                                 # (E, 1)
        seg_idx = (t_exp >= transition_times[:, :-1]).sum(dim=1) - 1    # (E,)
        seg_idx = seg_idx.clamp(0, S - 1)                               # (E,)

        # 2. Local parameter u ∈ [0, 1] within the selected segment
        arange_e = torch.arange(E, device=device)
        t_start  = transition_times[arange_e, seg_idx]                  # (E,)
        t_end    = transition_times[arange_e, seg_idx + 1]              # (E,)
        u = (t_clamped - t_start) / (t_end - t_start).clamp(min=1e-8)  # (E,)
        u = u.clamp(0.0, 1.0)                                           # (E,)

        # 3. Bernstein basis: B_n(u) = C(D,n) · u^n · (1-u)^(D-n)
        u_col = u.unsqueeze(1)                                           # (E, 1)
        n_row = self._n_vec.unsqueeze(0)                                 # (1, D+1)
        m_row = self._D_minus_n.unsqueeze(0)                             # (1, D+1)
        basis = self._binom_coeffs * u_col.pow(n_row) * (1.0 - u_col).pow(m_row)
        # basis: (E, D+1)

        # 4. Gather control points for the active segment
        #    ctrl_pts: (E, F, S, D+1, 3) → gather dim 2 → (E, F, 1, D+1, 3)
        seg_exp = seg_idx[:, None, None, None, None].expand(E, F, 1, D1, 3)
        pts = ctrl_pts.gather(2, seg_exp).squeeze(2)                     # (E, F, D+1, 3)

        # 5. Evaluate: sum_n B_n(u) · pts[:, :, n, :]
        basis_exp = basis[:, None, :, None].expand(E, F, D1, 3)         # (E, F, D+1, 3)
        return (basis_exp * pts).sum(dim=2)                              # (E, F, 3)

    def query_velocities(
        self,
        ctrl_pts: torch.Tensor,
        transition_times: torch.Tensor,
        t_elapsed: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the analytical Bezier velocity (time derivative) at the current episode time.

        Uses the degree-reduction identity for Bezier derivatives:
            dB/du = D * Σ_{n=0}^{D-1} (P_{n+1} - P_n) * B_n^{D-1}(u)
            dp/dt = dB/du / (t_end - t_start)

        Past the end of each guide trajectory the velocity is zeroed out so
        that the base policy sees a stop command once the crossing is complete.

        Parameters
        ----------
        ctrl_pts : torch.Tensor, shape (num_envs, n_frames, n_segments, degree+1, 3)
        transition_times : torch.Tensor, shape (num_envs, n_segments+1)
        t_elapsed : torch.Tensor, shape (num_envs,)

        Returns
        -------
        torch.Tensor, shape (num_envs, n_frames, 3)
            World-frame body velocity in m/s.
        """
        E  = ctrl_pts.shape[0]
        S  = self.n_segments
        D  = self.degree
        D1 = self.degree_plus_1
        F  = self.n_frames
        device = ctrl_pts.device

        # Flag environments that have elapsed past the end of their guide.
        past_end = t_elapsed >= transition_times[:, -1]   # (E,) bool

        # Clamp t to the valid curve domain (same as query_targets).
        t_clamped = torch.max(
            torch.min(t_elapsed, transition_times[:, -1]),
            transition_times[:, 0],
        )  # (E,)

        # 1. Segment selection.
        t_exp   = t_clamped.unsqueeze(1)                                 # (E, 1)
        seg_idx = (t_exp >= transition_times[:, :-1]).sum(dim=1) - 1    # (E,)
        seg_idx = seg_idx.clamp(0, S - 1)

        # 2. Local parameter u and segment duration.
        arange_e = torch.arange(E, device=device)
        t_start  = transition_times[arange_e, seg_idx]                  # (E,)
        t_end    = transition_times[arange_e, seg_idx + 1]              # (E,)
        seg_dur  = (t_end - t_start).clamp(min=1e-8)                    # (E,)
        u = ((t_clamped - t_start) / seg_dur).clamp(0.0, 1.0)          # (E,)

        # 3. Gather active segment control points: (E, F, D+1, 3).
        seg_exp = seg_idx[:, None, None, None, None].expand(E, F, 1, D1, 3)
        pts = ctrl_pts.gather(2, seg_exp).squeeze(2)                    # (E, F, D+1, 3)

        # 4. First-order finite differences of control points: ΔP_n = P_{n+1} - P_n.
        delta_pts = pts[:, :, 1:, :] - pts[:, :, :-1, :]               # (E, F, D, 3)

        # 5. Degree-(D-1) Bernstein basis for derivative curve.
        u_col   = u.unsqueeze(1)                                        # (E, 1)
        n_row_d = self._n_vec_d.unsqueeze(0)                            # (1, D)
        m_row_d = self._D_minus_n_d.unsqueeze(0)                        # (1, D)
        basis_d = self._binom_coeffs_d * u_col.pow(n_row_d) * (1.0 - u_col).pow(m_row_d)
        # basis_d: (E, D)

        # 6. dB/du = D * Σ_n ΔP_n * B_n^{D-1}(u), shape (E, F, 3).
        basis_d_exp = basis_d[:, None, :, None].expand(E, F, D, 3)
        dBdu = D * (basis_d_exp * delta_pts).sum(dim=2)                 # (E, F, 3)

        # 7. dp/dt = dB/du / seg_dur.
        seg_dur_exp = seg_dur[:, None, None].expand(E, F, 3)
        vel = dBdu / seg_dur_exp                                        # (E, F, 3)

        # 8. Zero velocity for environments that have passed the end of the guide.
        vel[past_end] = 0.0

        return vel
