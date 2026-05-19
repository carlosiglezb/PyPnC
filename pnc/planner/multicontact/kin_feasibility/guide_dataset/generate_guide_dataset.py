"""Generate the guide dataset .npz file for the knee-knocker residual-policy env.

This script is standalone (no Isaac Sim required).  It:
  1. Loads shared robot/environment assets once (URDF, SCA geometry, nav-door
     collision model).
  2. Computes IRIS regions once for the default starting pose (dx=0, dy=0) and
     reuses them across all per-guide builds.
  3. For each sampled (alpha, w_rigid, T, xy_offset) combination, builds an
     IKCFreePlanner (wrapping LocomanipulationFramePlanner) with the shifted
     initial configuration, solves the kinematic SOCP, and stores the Bezier
     control points.

Usage (from repo root):
    python pnc/planner/multicontact/kin_feasibility/guide_dataset/generate_guide_dataset.py \\
        --save_path guide_dataset.npz \\
        --T_min 2.5 --T_max 3.0 --n_alpha 5 --n_w_rigid 5
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import pinocchio as pin

from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
from util.environment_creator import HoleInWallObstructed
from util.pydrake_meshcat_interface import hpoly_to_fcl_collision, create_convex_geom_from_copy
import pnc.planner.multicontact.contact_sequence_plans.hole_in_wall_plans as hole_plan
import config.multicontact.g1_planner_config as g1_params
from pnc.planner.multicontact.kin_feasibility import SCARobotGeometry
from pnc.planner.multicontact.kin_feasibility.self_collision_avoidance.SCAHPolyhedronGeometry import (
    SCAHPolyhedronGeometry,
)
from pnc.planner.multicontact.kin_feasibility.frame_traversable_region import FrameTraversableRegion
from pnc.planner.multicontact.kin_feasibility.planner_surface_contact import (
    PlannerSurfaceContact,
    MotionFrameSequencer,
)
from pnc.planner.multicontact.kin_feasibility.ik_cfree_planner import IKCFreePlanner
from pnc.planner.multicontact.kin_feasibility.locomanipulation_frame_planner import (
    LocomanipulationFramePlanner,
)
from pnc.planner.multicontact.kin_feasibility.guide_dataset.guide_dataset import (
    generate_guide_dataset,
)

# ---------------------------------------------------------------------------
# Planner options
# ---------------------------------------------------------------------------
B_USE_KNEES                    = True
B_USE_SELF_COLLISION_AVOIDANCE = True
B_USE_KNEES_IN_SMOOTH_PLAN     = False

# ---------------------------------------------------------------------------
# Contact geometry constants (matching cfree_dyn_y_robustness.py)
# ---------------------------------------------------------------------------
KNOCKER_X    = 0.35
KNOCKER_Z    = 0.44
FT_KN_OFFSET = np.array([0.15, 0., 0.28])
STEP_LENGTH  = 0.44
DOOR_L_INNER = np.array([0.34,  0.32, 1.0])
DOOR_R_INNER = np.array([0.34, -0.32, 1.0])

env_urdf_path = cwd + "/robot_model/ground/navy_door_fixed.urdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_robot_model(package_dir, urdf_file):
    rob_model, col_model, vis_model = pin.buildModelsFromUrdf(
        urdf_file, package_dir, pin.JointModelFreeFlyer())
    rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)
    return rob_model, col_model, vis_model, rob_data, col_data, vis_data


def _get_root_to_torso_offset(geom_model):
    for gm in geom_model.geometryObjects:
        if 'torso_primitive_shape' in gm.name:
            return gm.placement.translation
    raise ValueError("Could not find torso_primitive_shape in geometry model.")


def get_g1_pose_with_xy(n_joints: int, dx: float, y_pos: float) -> np.ndarray:
    """Default G1 'hole' joint configuration with floating-base x shifted by dx and y set to y_pos."""
    q0 = np.zeros(n_joints)
    q0[0]  = -0.697   # left_hip_pitch_joint
    q0[3]  =  1.23    # left_knee_joint
    q0[4]  = -0.53    # left_ankle_pitch_joint
    q0[6]  = -0.697   # right_hip_pitch_joint
    q0[9]  =  1.23    # right_knee_joint
    q0[10] = -0.53    # right_ankle_pitch_joint
    floating_base = np.array([-0.03 + dx, y_pos, 0.68, 0., 0., 0., 1.])
    return np.concatenate((floating_base, q0))


def _fk_starting_pose(robot_urdf_file, package_dir, q0, plan_to_model_frames):
    """Run FK and return a dict of 3-D frame positions keyed by planner frame name."""
    robot_fwdk = PinocchioRobotSystem(robot_urdf_file, package_dir, False, False)
    cmd = robot_fwdk.create_cmd_ordered_dict(
        q0[7:], np.zeros(len(q0[7:])), np.zeros(len(q0[7:])))
    robot_fwdk.update_system(
        None, None, None, None,
        q0[:3], q0[3:7], np.zeros(3), np.zeros(3),
        cmd["joint_pos"], cmd["joint_vel"])
    return {fr: robot_fwdk.get_link_iso(plan_to_model_frames[fr])[:3, 3]
            for fr in plan_to_model_frames.keys()}


def build_knocker_contact_seq(starting_pose: dict, b_use_knees: bool = True):
    """Five-phase contact sequence for the obstructed-hole environment.

    Final position is centred at y=0; the mid-step on the knee-knocker is at
    half the y-displacement from the initial torso y (matching
    cfree_dyn_y_robustness.py).
    """
    torso_y0   = starting_pose['torso'][1]
    dy_knocker = -torso_y0 / 2
    dy_final   = -torso_y0

    rf_on_knocker  = np.array([KNOCKER_X, starting_pose['RF'][1] + dy_knocker, KNOCKER_Z])
    rkn_on_knocker = rf_on_knocker + FT_KN_OFFSET

    lf_final    = starting_pose['LF']    + np.array([STEP_LENGTH, dy_final, 0.])
    rf_final    = starting_pose['RF']    + np.array([STEP_LENGTH, dy_final, 0.])
    lkn_final   = lf_final + FT_KN_OFFSET
    rkn_final   = rf_final + FT_KN_OFFSET
    torso_final = starting_pose['torso'] + np.array([STEP_LENGTH, dy_final, 0.])
    rh_final    = starting_pose['RH']    + np.array([STEP_LENGTH, dy_final, 0.])
    lh_final    = starting_pose['LH']    + np.array([STEP_LENGTH, dy_final, 0.])

    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Phase 1: hands to door frame ----
    fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'] if b_use_knees else ['LF', 'RF'])
    motion_frames_seq.add_motion_frame({'LH': DOOR_L_INNER, 'RH': DOOR_R_INNER})
    motion_frames_seq.add_contact_surfaces([
        PlannerSurfaceContact('LH', np.array([0, -1, 0])),
        PlannerSurfaceContact('RH', np.array([0,  1, 0])),
    ])

    # ---- Phase 2: RF + R_knee onto knocker ----
    if b_use_knees:
        fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({'RF': rf_on_knocker, 'R_knee': rkn_on_knocker})
    else:
        fixed_frames.append(['LF', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({'RF': rf_on_knocker})
    motion_frames_seq.add_contact_surfaces([PlannerSurfaceContact('RF', np.array([0, 0, 1]))])

    # ---- Phase 3: LF + L_knee onto knocker ----
    if b_use_knees:
        fixed_frames.append(['RF', 'R_knee', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({'LF': lf_final, 'L_knee': lkn_final})
    else:
        fixed_frames.append(['RF', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({'LF': lf_final})
    motion_frames_seq.add_contact_surfaces([PlannerSurfaceContact('LF', np.array([0, 0, 1]))])

    # ---- Phase 4: torso stabilisation + LH release ----
    if b_use_knees:
        fixed_frames.append(['LF', 'L_knee', 'RH'])
        motion_frames_seq.add_motion_frame({
            'torso': torso_final, 'RF': rf_final, 'R_knee': rkn_final, 'LH': lh_final,
        })
    else:
        fixed_frames.append(['LF', 'RH'])
        motion_frames_seq.add_motion_frame({
            'torso': torso_final, 'RF': rf_final, 'LH': lh_final,
        })
    motion_frames_seq.add_contact_surfaces([PlannerSurfaceContact('RF', np.array([0, 0, 1]))])

    # ---- Phase 5: final balance ----
    if b_use_knees:
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH'])
    else:
        fixed_frames.append(['torso', 'LF', 'RF', 'LH'])
    motion_frames_seq.add_motion_frame({'RH': rh_final})

    return fixed_frames, motion_frames_seq


# ---------------------------------------------------------------------------
# One-time setup (robot model, environment, SCA geometry, IRIS regions)
# ---------------------------------------------------------------------------

def setup_g1_obstructed_hole() -> dict:
    """Load all shared assets and precompute IRIS regions for the default pose."""
    robot_name        = 'g1'
    package_dir       = cwd + "/robot_model/g1_description"
    robot_urdf_file   = package_dir + "/g1_29dof_lock_waist_modified.urdf"

    plan_to_model_frames = OrderedDict([
        ('torso',   'torso_primitive_shape'),
        ('LF',      'left_ankle_roll_link'),
        ('RF',      'right_ankle_roll_link'),
    ])
    if B_USE_KNEES:
        plan_to_model_frames['L_knee'] = 'left_knee_link'
        plan_to_model_frames['R_knee'] = 'right_knee_link'
    plan_to_model_frames['LH'] = 'left_rubber_hand'
    plan_to_model_frames['RH'] = 'right_rubber_hand'

    aux_frames_path = (
        cwd + f'/pnc/reachability_map/output/{robot_name}/{robot_name}_aux_frames.yaml'
        if B_USE_KNEES else None
    )

    reach_path = cwd + f'/pnc/reachability_map/output/{robot_name}/{robot_name}'
    ee_halfspace_params = OrderedDict(
        {fr: reach_path + '_' + fr + '.yaml' for fr in plan_to_model_frames.keys()}
    )

    rob_model, col_model, vis_model, _, _, _ = _load_robot_model(package_dir, robot_urdf_file)

    geom_model = pin.buildGeomFromUrdf(rob_model, robot_urdf_file, pin.GeometryType.COLLISION)
    root_to_torso_offset = _get_root_to_torso_offset(geom_model)

    door_pos        = np.array([0.32, 0.0, 0.])
    obstructed_hole = HoleInWallObstructed(door_pos)

    sca_geometry = (
        SCARobotGeometry(package_dir, robot_urdf_file, plan_to_model_frames)
        if B_USE_SELF_COLLISION_AVOIDANCE else None
    )

    planner_params = g1_params.MultiContactDoorConfig()

    # Compute IRIS once for the default pose (dx=0, dy=0) and reuse across all guides.
    q0_default = get_g1_pose_with_xy(rob_model.nq - 7, 0.0, 0.0)
    starting_pose_default = _fk_starting_pose(
        robot_urdf_file, package_dir, q0_default, plan_to_model_frames)
    _, motion_frames_seq_default = build_knocker_contact_seq(
        starting_pose_default, B_USE_KNEES)
    iris_regions_default = hole_plan.compute_iris_regions_mgr(
        obstructed_hole, starting_pose_default, motion_frames_seq_default,
        b_use_knees=B_USE_KNEES)

    return dict(
        package_dir          = package_dir,
        robot_urdf_file      = robot_urdf_file,
        plan_to_model_frames = plan_to_model_frames,
        aux_frames_path      = aux_frames_path,
        ee_halfspace_params  = ee_halfspace_params,
        rob_model            = rob_model,
        col_model            = col_model,
        vis_model            = vis_model,
        root_to_torso_offset = root_to_torso_offset,
        obstructed_hole      = obstructed_hole,
        sca_geometry         = sca_geometry,
        planner_params       = planner_params,
        iris_regions_default = iris_regions_default,
    )


# ---------------------------------------------------------------------------
# Per-guide planner builder
# ---------------------------------------------------------------------------

def build_planner(
    xy_offset: np.ndarray,
    shared: dict,
) -> tuple[IKCFreePlanner, dict]:
    """Assemble an IKCFreePlanner for the given xy_offset = [dx, dy].

    The initial configuration is obtained from FK with a floating-base shifted
    by (dx, dy) from the default pose.  IRIS regions are the precomputed
    default-pose regions stored in ``shared``.

    Parameters
    ----------
    xy_offset : np.ndarray, shape (2,)
        [dx, dy] applied to the floating-base x and y coordinates.
    shared : dict
        Output of :func:`setup_g1_obstructed_hole`.

    Returns
    -------
    ik_cfree_planner : IKCFreePlanner
    p_init : dict[str, np.ndarray]
        Starting 3-D position of each planner frame.
    """
    dx    = float(xy_offset[0])
    y_pos = float(xy_offset[1])

    rob_model            = shared['rob_model']
    col_model            = shared['col_model']
    vis_model            = shared['vis_model']
    plan_to_model_frames = shared['plan_to_model_frames']
    aux_frames_path      = shared['aux_frames_path']
    ee_halfspace_params  = shared['ee_halfspace_params']
    root_to_torso_offset = shared['root_to_torso_offset']
    package_dir          = shared['package_dir']
    robot_urdf_file      = shared['robot_urdf_file']
    obstructed_hole      = shared['obstructed_hole']
    sca_geometry         = shared['sca_geometry']
    planner_params       = shared['planner_params']
    iris_regions         = shared['iris_regions_default']

    q0            = get_g1_pose_with_xy(rob_model.nq - 7, dx, y_pos)
    starting_pose = _fk_starting_pose(robot_urdf_file, package_dir, q0, plan_to_model_frames)

    fixed_frames_seq, motion_frames_seq = build_knocker_contact_seq(
        starting_pose, B_USE_KNEES)

    rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)
    ik_cfree_planner = IKCFreePlanner(
        rob_model, rob_data, plan_to_model_frames, q0, planner_params)

    standing_pos = q0[:3]
    traversable_regions_dict = OrderedDict()
    for fr in plan_to_model_frames.keys():
        if fr == 'torso':
            traversable_regions_dict[fr] = FrameTraversableRegion(
                fr, b_visualize_reach=False, b_visualize_safe=False)
        else:
            traversable_regions_dict[fr] = FrameTraversableRegion(
                fr, ee_halfspace_params[fr],
                b_visualize_reach=False, b_visualize_safe=False,
                root_to_torso_pos=root_to_torso_offset)
            traversable_regions_dict[fr].update_origin_pose(standing_pos)
        traversable_regions_dict[fr].load_iris_regions(iris_regions[fr])

    frame_order = (
        ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']
        if B_USE_KNEES else
        ['torso', 'LF', 'RF', 'LH', 'RH']
    )
    traversable_regions = [traversable_regions_dict[fr] for fr in frame_order]

    frame_planner = LocomanipulationFramePlanner(
        traversable_regions,
        aux_frames_path=aux_frames_path,
        fixed_frames=fixed_frames_seq,
        motion_frames_seq=motion_frames_seq,
        sca_robot_geom=sca_geometry,
        b_use_knees_in_smooth_plan=B_USE_KNEES_IN_SMOOTH_PLAN)

    ik_cfree_planner.set_planner(frame_planner)
    ik_cfree_planner.set_plan_to_model_frames(plan_to_model_frames)

    env_geom = SCAHPolyhedronGeometry.from_wall_scene(obstructed_hole)
    ik_cfree_planner.set_env_geometry(env_geom)

    p_init = {fr: starting_pose[fr] for fr in plan_to_model_frames.keys()}
    return ik_cfree_planner, p_init


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate guide-dataset .npz for the knee-knocker residual env"
    )
    parser.add_argument("--save_path", default="guide_dataset.npz",
                        help="Output .npz file path (default: guide_dataset.npz)")
    parser.add_argument("--T_min", type=float, default=2.5,
                        help="Minimum traversal duration in seconds (default: 2.5)")
    parser.add_argument("--T_max", type=float, default=3.0,
                        help="Maximum traversal duration in seconds (default: 3.0)")
    parser.add_argument("--n_alpha", type=int, default=5,
                        help="Number of alpha samples (default: 5)")
    parser.add_argument("--n_w_rigid", type=int, default=5,
                        help="Number of w_rigid samples (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--xy_offset_bounds", type=float, nargs=2, default=[0.025, 0.15],
                        metavar=("BX", "BY"),
                        help="Symmetric XY bounds for per-guide offset sampling: "
                             "dx ~ U(-BX, +BX), dy ~ U(-BY, +BY) (default: 0.025 0.15)")
    args = parser.parse_args()

    xy_offset_bounds = np.array(args.xy_offset_bounds, dtype=np.float64)
    total = args.n_alpha * args.n_w_rigid

    print(f"Generating {args.n_alpha} × {args.n_w_rigid} = {total} guides  "
          f"(T ~ U({args.T_min}, {args.T_max})s, "
          f"offset bounds=[±{xy_offset_bounds[0]:.3f}, ±{xy_offset_bounds[1]:.3f}]) ...")

    print("Loading robot model and computing IRIS regions for default pose ...")
    shared = setup_g1_obstructed_hole()

    def planner_builder_fn(xy_offset: np.ndarray):
        return build_planner(xy_offset, shared)

    generate_guide_dataset(
        planner_builder_fn=planner_builder_fn,
        xy_offset_bounds=xy_offset_bounds,
        T_min=args.T_min,
        T_max=args.T_max,
        n_alpha=args.n_alpha,
        n_w_rigid=args.n_w_rigid,
        seed=args.seed,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
