"""Load a guide dataset .npz and visualize a sampled subset of trajectories.

Handles two dataset formats automatically:
  * **Kinematic** (``guide_dataset.npz``) — Bezier control-point format produced
    by ``generate_guide_dataset.py``.  Uses ``GuideDataset.query_targets`` for the
    matplotlib plot and animates collision-body targets in meshcat.
  * **Dynamic** (``guide_dataset_dyn.npz``) — discrete joint-trajectory format
    produced by the same script with ``--save_dyn_plan``.  Uses Pinocchio FK to
    reconstruct frame positions for matplotlib, and ``animate_frame`` for a full
    robot visual-mesh animation in meshcat.

Usage (from repo root):
    python pnc/planner/multicontact/kin_feasibility/guide_dataset/visualize_guide_dataset.py \\
        --dataset guide_dataset.npz     --n_guides 4 --n_queries 150 --seed 0
    python pnc/planner/multicontact/kin_feasibility/guide_dataset/visualize_guide_dataset.py \\
        --dataset guide_dataset_dyn.npz --n_guides 4 --n_queries 150 --seed 0
"""
from __future__ import annotations

import argparse
import sys
import os
from collections import OrderedDict

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import pinocchio as pin

from pnc.planner.multicontact.kin_feasibility.guide_dataset.guide_dataset import GuideDataset
import plot.meshcat_utils as vis_tools
from util.environment_creator import HoleInWallObstructed

# Planner frame name → Pinocchio collision-geometry base name (suffix '_0' added at call sites)
_PLAN_TO_COLLISION: dict[str, str] = OrderedDict([
    ('torso',   'torso_primitive_shape'),
    ('LF',      'left_ankle_roll_link'),
    ('RF',      'right_ankle_roll_link'),
    ('L_knee',  'left_knee_link'),
    ('R_knee',  'right_knee_link'),
    ('LH',      'left_rubber_hand'),
    ('RH',      'right_rubber_hand'),
])
_FRAME_NAMES = list(_PLAN_TO_COLLISION.keys())


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def _detect_dataset_type(path: str) -> str:
    """Return ``'kin'`` or ``'dyn'`` based on the keys present in the .npz."""
    data = np.load(path, allow_pickle=True)
    if 'control_points' in data:
        return 'kin'
    if 'joint_pos' in data:
        return 'dyn'
    raise ValueError(
        f"{path} is not a recognised guide dataset "
        "(expected 'control_points' or 'joint_pos' key)."
    )


# ---------------------------------------------------------------------------
# Shared robot model loader
# ---------------------------------------------------------------------------

def _load_robot_from_urdf(urdf_file: str):
    """Shared loader: build Pinocchio model bundle + torso offset + frame IDs."""
    package_dir = cwd + "/robot_model/g1_description"
    rob_model, col_model, vis_model = pin.buildModelsFromUrdf(
        urdf_file, package_dir, pin.JointModelFreeFlyer())
    rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)

    q_zero = pin.neutral(rob_model)
    torso_frame_id = rob_model.getFrameId('torso_primitive_shape')
    pin.forwardKinematics(rob_model, rob_data, q_zero)
    pin.updateFramePlacements(rob_model, rob_data)
    torso_offset = rob_data.oMf[torso_frame_id].translation.copy()

    frame_ids = {fname: rob_model.getFrameId(body)
                 for fname, body in _PLAN_TO_COLLISION.items()}

    return rob_model, col_model, vis_model, rob_data, col_data, vis_data, torso_offset, frame_ids


def _load_robot():
    """Load the kinematic G1 model (used for kin-dataset visualisation)."""
    package_dir = cwd + "/robot_model/g1_description"
    return _load_robot_from_urdf(package_dir + "/g1_29dof_lock_waist_modified.urdf")


def _load_dyn_robot():
    """Load the dynamic G1 model — must match the URDF used during trajectory optimisation."""
    package_dir = cwd + "/robot_model/g1_description"
    return _load_robot_from_urdf(package_dir + "/g1_29dof_simple_collisions.urdf")


def _build_q_full(q_base_t: np.ndarray, joints_t: np.ndarray) -> np.ndarray:
    """Assemble full Pinocchio configuration from floating-base state + joint angles."""
    return np.concatenate([q_base_t, joints_t])


# ---------------------------------------------------------------------------
# Dataset summaries
# ---------------------------------------------------------------------------

def _print_kin_summary(dataset: GuideDataset) -> None:
    print("\n=== Kinematic GuideDataset summary ===")
    s = dataset.ctrl_pts.shape
    print(f"  ctrl_pts shape : {tuple(s)}  "
          f"(n_guides={s[0]} × n_frames={s[1]} × n_segments={s[2]} "
          f"× (degree+1)={s[3]} × 3)")
    print(f"  Bezier degree  : {dataset.degree}")
    print(f"  frames         : {dataset.frame_names}")
    print(f"  T_plan (mean)  : {dataset.T_plan:.3f} s")
    t = dataset.T_plan_arr.numpy()
    print(f"  T_plan range   : [{t.min():.3f}, {t.max():.3f}] s")
    if dataset.hand_mask is not None:
        contact = [n for n, m in zip(dataset.frame_names, dataset.hand_mask.tolist()) if m]
        print(f"  contact frames : {contact}")
    torso_xy = dataset.p_init_nominal_torso_arr[:, :2].numpy()
    print(f"  torso XY range : "
          f"x=[{torso_xy[:, 0].min():+.3f}, {torso_xy[:, 0].max():+.3f}]  "
          f"y=[{torso_xy[:, 1].min():+.3f}, {torso_xy[:, 1].max():+.3f}]")
    has_ik = dataset.q_phase_boundary is not None
    print(f"  q_phase_boundary: "
          f"{'present  shape=' + str(tuple(dataset.q_phase_boundary.shape)) if has_ik else 'absent'}")


def _print_dyn_summary(data: np.lib.npyio.NpzFile) -> None:
    jp = data['joint_pos']
    print("\n=== Dynamic GuideDataset summary ===")
    print(f"  joint_pos shape : {jp.shape}  "
          f"(n_guides={jp.shape[0]}, n_steps={jp.shape[1]}, n_joints={jp.shape[2]})")
    T_arr = data['T_plan_arr']
    print(f"  T_plan range    : [{T_arr.min():.3f}, {T_arr.max():.3f}] s")
    print(f"  dt (mean)       : {float(data['dt']):.5f} s")
    if 'joint_names' in data and len(data['joint_names']) > 0:
        print(f"  joint_names     : {data['joint_names'].tolist()}")
    torso_xy = data['p_init_nominal_torso'][:, :2]
    print(f"  torso XY range  : "
          f"x=[{torso_xy[:, 0].min():+.3f}, {torso_xy[:, 0].max():+.3f}]  "
          f"y=[{torso_xy[:, 1].min():+.3f}, {torso_xy[:, 1].max():+.3f}]")
    if 'q_base' in data:
        qb = data['q_base']              # (N, n_steps, 7): xyz + quat xyzw
        quat = qb[:, :, 3:]              # (N, n_steps, 4): qx qy qz qw
        # deviation of qx,qy,qz from zero — zero means perfectly upright
        max_tilt = np.abs(quat[:, :, :3]).max()
        print(f"  q_base          : present  shape={qb.shape}  "
              f"max |qx,qy,qz| = {max_tilt:.4f} "
              f"({'non-trivial orientation' if max_tilt > 1e-3 else 'WARNING: always upright — regenerate dataset'})")
    else:
        print("  q_base          : ABSENT — base orientation will be approximated as "
              "upright (roll/pitch will be missing). Regenerate the dataset.")


# ---------------------------------------------------------------------------
# Kinematic dataset — matplotlib + meshcat
# ---------------------------------------------------------------------------

def _kin_visualize_matplotlib(
    dataset: GuideDataset,
    idx: torch.Tensor,
    ctrl_pts: torch.Tensor,
    trans_times: torch.Tensor,
    T_per_env: torch.Tensor,
    torso_nominals: torch.Tensor,
    base_torso_xy: np.ndarray,
    n_queries: int,
    save_path: str | None,
    dataset_path: str,
) -> None:
    n_guides    = len(idx)
    frame_names = dataset.frame_names
    colors      = plt.cm.tab10(np.linspace(0, 1, dataset.n_frames))
    fig = plt.figure(figsize=(10, 5 * n_guides))

    for g in range(n_guides):
        ax = fig.add_subplot(n_guides, 1, g + 1, projection='3d')
        T_g        = float(T_per_env[g])
        t_vals     = torch.linspace(0.0, T_g, n_queries)
        ctrl_pts_g = ctrl_pts[g:g+1].expand(n_queries, -1, -1, -1, -1)
        trans_g    = trans_times[g:g+1].expand(n_queries, -1)
        targets    = dataset.query_targets(ctrl_pts_g, trans_g, t_vals)
        pts_np     = targets.numpy()

        for f_idx, fname in enumerate(frame_names):
            traj = pts_np[:, f_idx, :]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    '-o', color=colors[f_idx], markersize=2, linewidth=1.5,
                    label=fname if g == 0 else None)
            ax.scatter(*traj[0],  marker='>', s=40, color=colors[f_idx], zorder=6)
            ax.scatter(*traj[-1], marker='s', s=40, color=colors[f_idx], zorder=6)

        p_nom    = torso_nominals[g].numpy()
        xy_off_g = p_nom[:2] - base_torso_xy
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
        ax.set_title(
            f"Guide {g}  (idx {idx[g].item()})  |  T = {T_g:.2f} s  |  "
            f"torso offset = [{xy_off_g[0]:+.3f}, {xy_off_g[1]:+.3f}] m", fontsize=10)
        if g == 0:
            ax.legend(loc='upper left', fontsize=7, ncol=3)

    plt.suptitle(f"{dataset_path}  [kinematic]  —  {n_guides} guides", fontsize=11)
    plt.tight_layout()
    out = save_path or os.path.splitext(dataset_path)[0] + "_preview.png"
    plt.savefig(out, dpi=130)
    print(f"\nSaved → {out}")
    plt.show()


def _kin_visualize_meshcat(
    dataset: GuideDataset,
    idx: torch.Tensor,
    ctrl_pts: torch.Tensor,
    trans_times: torch.Tensor,
    T_per_env: torch.Tensor,
    n_queries: int,
    rob_model, col_model, vis_model, rob_data, col_data, vis_data,
    torso_offset: np.ndarray,
) -> None:
    """Animate collision-body targets; overlay full robot mesh when q_phase_boundary present."""
    door_pos        = np.array([0.32, 0.0, 0.])
    obstructed_hole = HoleInWallObstructed(door_pos)

    frame_names = dataset.frame_names
    has_ik      = dataset.q_phase_boundary is not None
    n_joints    = rob_model.nq - 7

    kin_display = vis_tools.MeshcatPinocchioAnimation(
        rob_model, col_model, vis_model,
        rob_data, vis_data, col_data,
        ctrl_freq=n_queries, save_freq=1)
    kin_display.add_shapes_from(obstructed_hole.obstacles)
    kin_display.start_animation()

    q_phase_np  = dataset.q_phase_boundary.numpy() if has_ik else None

    for g in range(len(idx)):
        g_global   = idx[g].item()
        T_g        = float(T_per_env[g])
        t_vals     = torch.linspace(0.0, T_g, n_queries)
        ctrl_pts_g = ctrl_pts[g:g+1].expand(n_queries, -1, -1, -1, -1)
        trans_g    = trans_times[g:g+1].expand(n_queries, -1)
        targets    = dataset.query_targets(ctrl_pts_g, trans_g, t_vals)
        pts_np     = targets.numpy()
        trans_g_np = trans_times[g].numpy()

        for t_idx in range(n_queries):
            for f_idx, fname in enumerate(frame_names):
                if fname in _PLAN_TO_COLLISION:
                    kin_display.animate_single_collision(
                        _PLAN_TO_COLLISION[fname] + '_0', pts_np[t_idx, f_idx])

            if has_ik:
                t_now = float(t_vals[t_idx])
                phase = max(0, int(np.sum(trans_g_np[:-1] <= t_now)) - 1)
                phase = min(phase, q_phase_np.shape[1] - 1)
                joints = q_phase_np[g_global, phase, :n_joints]
                torso_target = pts_np[t_idx, frame_names.index('torso')]
                base_pos = torso_target - torso_offset
                q_full = np.concatenate([base_pos, [0., 0., 0., 1.], joints])
                kin_display.animate_frame(q_full)

            kin_display.animation_step()

    kin_display.finish_animation()
    print("Meshcat animation ready — open the meshcat viewer URL shown above.")


# ---------------------------------------------------------------------------
# Dynamic dataset — matplotlib + meshcat
# ---------------------------------------------------------------------------

def _dyn_visualize_matplotlib(
    data: np.lib.npyio.NpzFile,
    q_base: np.ndarray,
    g_indices: np.ndarray,
    n_queries: int,
    rob_model, rob_data,
    frame_ids: dict[str, int],
    save_path: str | None,
    dataset_path: str,
) -> None:
    joint_pos    = data['joint_pos']              # (N, n_steps, n_joints)
    # q_base passed in: (N, n_steps, 7) — exact from dataset or approximated by caller
    T_plan_arr   = data['T_plan_arr']             # (N,)
    p_init_torso = data['p_init_nominal_torso']   # (N, 3)

    n_steps   = joint_pos.shape[1]
    step_ids  = np.linspace(0, n_steps - 1, n_queries, dtype=int)
    n_guides  = len(g_indices)
    colors    = plt.cm.tab10(np.linspace(0, 1, len(_FRAME_NAMES)))
    base_torso_xy = p_init_torso[:, :2].mean(0)

    fig = plt.figure(figsize=(10, 5 * n_guides))

    for plot_g, g in enumerate(g_indices):
        ax = fig.add_subplot(n_guides, 1, plot_g + 1, projection='3d')

        # FK over downsampled steps using exact floating-base pose from the optimizer
        traj = {fname: np.zeros((n_queries, 3)) for fname in _FRAME_NAMES}
        for q_idx, t_idx in enumerate(step_ids):
            q_full = _build_q_full(q_base[g, t_idx], joint_pos[g, t_idx])
            pin.forwardKinematics(rob_model, rob_data, q_full)
            pin.updateFramePlacements(rob_model, rob_data)
            for fname, fid in frame_ids.items():
                traj[fname][q_idx] = rob_data.oMf[fid].translation

        for f_idx, fname in enumerate(_FRAME_NAMES):
            t = traj[fname]
            ax.plot(t[:, 0], t[:, 1], t[:, 2],
                    '-o', color=colors[f_idx], markersize=2, linewidth=1.5,
                    label=fname if plot_g == 0 else None)
            ax.scatter(*t[0],  marker='>', s=40, color=colors[f_idx], zorder=6)
            ax.scatter(*t[-1], marker='s', s=40, color=colors[f_idx], zorder=6)

        T_g   = float(T_plan_arr[g])
        p_nom = p_init_torso[g]
        xy_off = p_nom[:2] - base_torso_xy
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
        ax.set_title(
            f"Guide {plot_g}  (idx {g})  |  T = {T_g:.2f} s  |  "
            f"torso offset = [{xy_off[0]:+.3f}, {xy_off[1]:+.3f}] m", fontsize=10)
        if plot_g == 0:
            ax.legend(loc='upper left', fontsize=7, ncol=3)

    plt.suptitle(f"{dataset_path}  [dynamic]  —  {n_guides} guides", fontsize=11)
    plt.tight_layout()
    out = save_path or os.path.splitext(dataset_path)[0] + "_preview.png"
    plt.savefig(out, dpi=130)
    print(f"\nSaved → {out}")
    plt.show()


def _dyn_visualize_meshcat(
    data: np.lib.npyio.NpzFile,
    q_base: np.ndarray,
    g_indices: np.ndarray,
    n_queries: int,
    rob_model, col_model, vis_model, rob_data, col_data, vis_data,
) -> None:
    """Animate the full robot visual mesh from joint-level trajectories."""
    joint_pos = data['joint_pos']   # (N, n_steps, n_joints)
    # q_base passed in: (N, n_steps, 7) — exact from dataset or approximated by caller

    n_steps  = joint_pos.shape[1]
    step_ids = np.linspace(0, n_steps - 1, n_queries, dtype=int)

    # ctrl_freq sets the animation framerate (fps = ctrl_freq / save_freq).
    # We display n_queries frames spanning the full trajectory duration so the
    # playback speed matches real time.
    dt_mean    = float(data['dt'])
    T_total    = dt_mean * (n_steps - 1)
    ctrl_freq  = (n_queries - 1) / T_total   # frames / second

    door_pos        = np.array([0.32, 0.0, 0.])
    obstructed_hole = HoleInWallObstructed(door_pos)

    kin_display = vis_tools.MeshcatPinocchioAnimation(
        rob_model, col_model, vis_model,
        rob_data, vis_data, col_data,
        ctrl_freq=ctrl_freq, save_freq=1)
    kin_display.add_shapes_from(obstructed_hole.obstacles)
    kin_display.start_animation()

    for g in g_indices:
        for t_idx in step_ids:
            q_full = _build_q_full(q_base[g, t_idx], joint_pos[g, t_idx])
            kin_display.animate_frame(q_full)
            kin_display.animation_step()

    kin_display.finish_animation()
    print("Meshcat animation ready — open the meshcat viewer URL shown above.")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def visualize(
    dataset_path: str,
    n_guides: int,
    n_queries: int,
    seed: int,
    save_path: str | None,
    no_meshcat: bool,
) -> None:
    dtype = _detect_dataset_type(dataset_path)
    print(f"Detected dataset type: {dtype}")

    if dtype == 'kin':
        rob_model, col_model, vis_model, rob_data, col_data, vis_data, torso_offset, frame_ids = (
            _load_robot()
        )
    else:
        # Must match the URDF used during trajectory optimisation so q_base is interpreted correctly.
        rob_model, col_model, vis_model, rob_data, col_data, vis_data, torso_offset, frame_ids = (
            _load_dyn_robot()
        )

    if dtype == 'kin':
        dataset = GuideDataset(dataset_path, device="cpu")
        _print_kin_summary(dataset)

        torch.manual_seed(seed)
        idx            = torch.randperm(dataset.n_guides)[:n_guides]
        ctrl_pts       = dataset.ctrl_pts[idx].clone()
        trans_times    = dataset.transition_times[idx].clone()
        T_per_env      = dataset.T_plan_arr[idx].clone()
        torso_nominals = dataset.p_init_nominal_torso_arr[idx]
        base_torso_xy  = dataset.p_init_nominal_torso_arr[:, :2].mean(0).numpy()

        print(f"\nSampled {n_guides} guides  |  T: {T_per_env.numpy().round(3).tolist()}")
        for g_i, (g_idx, p_nom) in enumerate(zip(idx.tolist(), torso_nominals.numpy())):
            xy_off = p_nom[:2] - base_torso_xy
            print(f"  guide {g_i} (idx {g_idx:3d}): torso xy = [{p_nom[0]:+.3f}, {p_nom[1]:+.3f}]"
                  f"  offset = [{xy_off[0]:+.3f}, {xy_off[1]:+.3f}]")

        _kin_visualize_matplotlib(
            dataset, idx, ctrl_pts, trans_times, T_per_env,
            torso_nominals, base_torso_xy, n_queries, save_path, dataset_path)

        if not no_meshcat:
            _kin_visualize_meshcat(
                dataset, idx, ctrl_pts, trans_times, T_per_env, n_queries,
                rob_model, col_model, vis_model, rob_data, col_data, vis_data, torso_offset)

    else:  # dyn
        data = np.load(dataset_path, allow_pickle=True)
        _print_dyn_summary(data)

        # Resolve floating-base pose array: exact if saved, approximated otherwise.
        if 'q_base' in data:
            q_base = data['q_base']          # (N, n_steps, 7)
        else:
            print("Warning: 'q_base' not found in dataset — "
                  "falling back to upright-base approximation from torso_pos. "
                  "Regenerate the dataset to get exact base orientation.")
            torso_pos = data['torso_pos']    # (N, n_steps, 3)
            N, n_steps, _ = torso_pos.shape
            base_xyz = torso_pos - torso_offset   # (N, n_steps, 3)
            upright_quat = np.tile([0., 0., 0., 1.], (N, n_steps, 1))
            q_base = np.concatenate([base_xyz, upright_quat], axis=-1).astype(np.float32)

        n_total   = data['joint_pos'].shape[0]
        rng       = np.random.default_rng(seed)
        g_indices = rng.choice(n_total, size=min(n_guides, n_total), replace=False)

        p_init    = data['p_init_nominal_torso']
        base_xy   = p_init[:, :2].mean(0)
        print(f"\nSampled {len(g_indices)} guides:")
        for plot_g, g in enumerate(g_indices):
            p_nom  = p_init[g]
            xy_off = p_nom[:2] - base_xy
            print(f"  guide {plot_g} (idx {g:3d}): torso xy = [{p_nom[0]:+.3f}, {p_nom[1]:+.3f}]"
                  f"  offset = [{xy_off[0]:+.3f}, {xy_off[1]:+.3f}]")

        _dyn_visualize_matplotlib(
            data, q_base, g_indices, n_queries,
            rob_model, rob_data, frame_ids,
            save_path, dataset_path)

        if not no_meshcat:
            _dyn_visualize_meshcat(
                data, q_base, g_indices, n_queries,
                rob_model, col_model, vis_model, rob_data, col_data, vis_data)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize kinematic or dynamic guide datasets"
    )
    parser.add_argument("--dataset", default="guide_dataset_dyn.npz",
                        help="Path to the .npz dataset (default: guide_dataset.npz)")
    parser.add_argument("--n_guides", type=int, default=2,
                        help="Number of guides to sample and plot (default: 2)")
    parser.add_argument("--n_queries", type=int, default=150,
                        help="Time steps to display per guide (default: 150)")
    parser.add_argument("--seed", type=int, default=10,
                        help="RNG seed for guide sampling (default: 10)")
    parser.add_argument("--save_path", default=None,
                        help="Output PNG path (default: <dataset>_preview.png)")
    parser.add_argument("--no_meshcat", action="store_true",
                        help="Skip meshcat animation (matplotlib only)")
    args = parser.parse_args()

    visualize(
        dataset_path=args.dataset,
        n_guides=args.n_guides,
        n_queries=args.n_queries,
        seed=args.seed,
        save_path=args.save_path,
        no_meshcat=args.no_meshcat,
    )


if __name__ == "__main__":
    main()
