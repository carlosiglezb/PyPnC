"""
Robustness test for the G1 obstructed-hole planner under lateral (y) starting-position
variations.  Samples N_TRIALS starting configurations from U(Y_LB, Y_UB) in the
floating-base y-coordinate, keeping the joint angles identical to the default
'hole' pose.  The contact sequence is simplified: both feet and knees land directly
on the knee-knocker (x ≈ 0.35, z = 0.44) while the hand targets remain at the
standard door-frame locations.  At the end, a summary table reports which trials
returned an optimal (feasible) solution from the HumanoidMulticontactPlanner solver.
"""

import os
import sys
import time
import traceback
from collections import OrderedDict

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np

from pnc.planner.multicontact.kin_feasibility.self_collision_avoidance.SCAHPolyhedronGeometry import \
    SCAHPolyhedronGeometry
from util.pydrake_meshcat_interface import hpoly_to_fcl_collision, create_convex_geom_from_copy
import config.multicontact.g1_planner_config as g1_params
from pnc.planner.multicontact.kin_feasibility import SCARobotGeometry

from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
from util.environment_creator import HoleInWallObstructed
import pnc.planner.multicontact.contact_sequence_plans.hole_in_wall_plans as hole_plan

import crocoddyl

from pydrake.geometry.optimization import HPolyhedron

from pnc.planner.multicontact.kin_feasibility.frame_traversable_region import FrameTraversableRegion
from pnc.planner.multicontact.kin_feasibility.planner_surface_contact import (
    PlannerSurfaceContact, MotionFrameSequencer,
    get_contact_seq_from_fixed_frames_seq, get_contact_planes_from_motion_frames_seq,
)
from pnc.planner.multicontact.kin_feasibility.ik_cfree_planner import *
from humanoid_action_models import *
from pnc.planner.multicontact.dyn_feasibility.G1MulticontactPlanner import G1MulticontactPlanner
from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import ContactSequence
from vision.iris.iris_regions_manager import IrisRegionsManager, IrisGeomInterface

import pinocchio as pin
import plot.meshcat_utils as vis_tools
from pinocchio.visualize import MeshcatVisualizer

# ---------------------------------------------------------------------------
# Trial parameters
# ---------------------------------------------------------------------------
N_TRIALS    = 5
Y_LB, Y_UB = -0.15, 0.15
RNG_SEED    = 42          # set to None for non-reproducible draws

# ---------------------------------------------------------------------------
# Planner options (mirror cfree_dyn_planner.py defaults for the hole env)
# ---------------------------------------------------------------------------
SOLVE_BY_SECTIONS           = 'single'
SOLVER_TYPE                 = 'SQP'
B_SOLVE_HYBRID              = False
B_SCA_REFINEMENT            = True
B_VERBOSE                   = False
B_USE_KNEES                 = True
B_USE_SELF_COLLISION_AVOIDANCE = True
B_USE_KNEES_IN_SMOOTH_PLAN  = False
B_VISUALIZE_KIN             = True

# ---------------------------------------------------------------------------
# Contact geometry constants for the obstructed-hole environment (G1)
# ---------------------------------------------------------------------------
# Knee-knocker landing height (top of base + foot thickness)
KNOCKER_X   = 0.35
KNOCKER_Z   = 0.44
FT_KN_OFFSET = np.array([0.15, 0., 0.28])   # foot → knee offset used in hole plans
STEP_LENGTH = 0.44

# Door-frame hand targets (matching get_on_knocker_balanced_contact_sequence in cfree_dyn_planner.py)
DOOR_L_INNER = np.array([0.34,  0.32, 1.0])
DOOR_R_INNER = np.array([0.34, -0.32, 1.0])

env_urdf_path = cwd + "/robot_model/ground/navy_door_fixed.urdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_robot_model(package_dir, urdf_file):
    rob_model, col_model, vis_model = pin.buildModelsFromUrdf(
        urdf_file, package_dir, pin.JointModelFreeFlyer())
    rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)
    return rob_model, col_model, vis_model, rob_data, col_data, vis_data


def load_navy_door_models():
    return pin.buildModelsFromUrdf(
        cwd + "/robot_model/ground/navy_door_fixed.urdf",
        cwd + "/robot_model/ground")


def get_root_to_torso_offset(geom_model):
    for i, gm in enumerate(geom_model.geometryObjects):
        if 'torso_primitive_shape' in gm.name:
            return gm.placement.translation
        if i == len(geom_model.geometryObjects) - 1:
            raise ValueError("Could not find torso_primitive_shape in geometry model.")


def get_g1_pose_with_y(n_joints: int, y_pos: float) -> np.ndarray:
    """Default G1 'hole' joint configuration with the floating-base y replaced."""
    q0 = np.zeros(n_joints)
    q0[0]  = -0.697   # left_hip_pitch_joint
    q0[3]  =  1.23    # left_knee_joint
    q0[4]  = -0.53    # left_ankle_pitch_joint
    q0[6]  = -0.697   # right_hip_pitch_joint
    q0[9]  =  1.23    # right_knee_joint
    q0[10] = -0.53    # right_ankle_pitch_joint
    floating_base = np.array([-0.03, y_pos, 0.68, 0., 0., 0., 1.])
    return np.concatenate((floating_base, q0))


def setup_visualizer(rob_model, col_model, vis_model, q0):
    """Create and initialise a MeshcatVisualizer without blocking (no viewer.wait())."""
    visualizer = MeshcatVisualizer(rob_model, col_model, vis_model)
    try:
        visualizer.initViewer(open=True)
    except ImportError as err:
        print("Error initializing meshcat viewer:", err)
        return None
    visualizer.loadViewerModel(rootNodeName=rob_model.name)
    visualizer.display(q0)
    return visualizer


def build_knocker_contact_seq(starting_pose: dict, b_use_knees: bool = True):
    """
    Five-phase contact sequence for the obstructed-hole environment where both
    feet and knees land on the knee-knocker.

    Phase 1  – reach for door handles (LH, RH)
    Phase 2  – RF + R_knee step onto the knocker
    Phase 3  – LF + L_knee step onto the knocker
    Phase 4  – torso stabilises over the knocker; LH adjusts
    Phase 5  – final balance; RH adjusts

    The robot's final standing position is centred at y=0.  RF lands on the
    knocker at the y mid-point between its initial y and 0 (dy_knocker), then
    all frames step to their final y-centred positions (dy_final).
    """
    torso_y0   = starting_pose['torso'][1]  # initial torso y ≈ y_pos
    dy_knocker = -torso_y0 / 2              # half y-shift: knocker step mid-point
    dy_final   = -torso_y0                  # full y-shift: centre torso at y=0

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
    if b_use_knees:
        fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])
    else:
        fixed_frames.append(['LF', 'RF'])
    motion_frames_seq.add_motion_frame({'LH': DOOR_L_INNER, 'RH': DOOR_R_INNER})
    lh_contact = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    rh_contact = PlannerSurfaceContact('RH', np.array([0,  1, 0]))
    motion_frames_seq.add_contact_surfaces([lh_contact, rh_contact])

    # ---- Phase 2: RF + R_knee onto knocker ----
    if b_use_knees:
        fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({
            'RF': rf_on_knocker,
            'R_knee': rkn_on_knocker,
        })
    else:
        fixed_frames.append(['LF', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({'RF': rf_on_knocker})
    rf_contact = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_contact])

    # ---- Phase 3: LF + L_knee onto knocker ----
    if b_use_knees:
        fixed_frames.append(['RF', 'R_knee', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({
            'LF': lf_final,
            'L_knee': lkn_final,
        })
    else:
        fixed_frames.append(['RF', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({'LF': lf_final})
    lf_contact = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([lf_contact])

    # ---- Phase 4: torso stabilisation + LH release ----
    # Mirrors "Step 6" of get_on_knocker_balanced_contact_sequence: LF, L_knee, RH fixed.
    # RF is already at its final position (knocker), so it stays fixed with a re-declared
    # floor contact rather than being moved further as in the door-traversal original.
    if b_use_knees:
        fixed_frames.append(['LF', 'L_knee', 'RH'])
        motion_frames_seq.add_motion_frame({
            'torso': torso_final,
            'RF': rf_final,
            'R_knee': rkn_final,
            'LH': lh_final,
        })
    else:
        fixed_frames.append(['LF', 'RH'])
        motion_frames_seq.add_motion_frame({
            'torso': torso_final,
            'RF': rf_final,
            'LH': lh_final,
        })
    rf_stable = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_stable])

    # ---- Phase 5: final balance ----
    # Mirrors "Step 7": all feet + torso + LH fixed; RH finishes.
    if b_use_knees:
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH'])
    else:
        fixed_frames.append(['torso', 'LF', 'RF', 'LH'])
    motion_frames_seq.add_motion_frame({'RH': rh_final})

    return fixed_frames, motion_frames_seq


# ---------------------------------------------------------------------------
# One-time robot / environment setup
# ---------------------------------------------------------------------------

def setup_g1_obstructed_hole():
    """
    Load all static assets (robot model, env, collision model) that are shared
    across all trials.  Returns a dict of shared objects.
    """
    robot_name = 'g1'
    package_dir = cwd + "/robot_model/g1_description"
    robot_urdf_file      = package_dir + "/g1_29dof_lock_waist_modified.urdf"
    refined_urdf_file    = package_dir + "/g1_29dof_simple_collisions.urdf"

    # Planner frame mappings
    plan_to_model_frames = OrderedDict()
    plan_to_model_frames['torso']   = 'torso_primitive_shape'
    plan_to_model_frames['LF']      = 'left_ankle_roll_link'
    plan_to_model_frames['RF']      = 'right_ankle_roll_link'
    if B_USE_KNEES:
        plan_to_model_frames['L_knee'] = 'left_knee_link'
        plan_to_model_frames['R_knee'] = 'right_knee_link'
    plan_to_model_frames['LH']      = 'left_rubber_hand'
    plan_to_model_frames['RH']      = 'right_rubber_hand'

    force_joint_frames = {
        'LF': "left_ankle_roll_joint",
        'RF': "right_ankle_roll_joint",
        'LH': "left_wrist_yaw_joint",
        'RH': "right_wrist_yaw_joint",
    }

    # Auxiliary / reachability paths
    if B_USE_KNEES:
        aux_frames_path = (cwd + '/pnc/reachability_map/output/' + robot_name + '/' +
                           robot_name + '_aux_frames.yaml')
    else:
        aux_frames_path = None

    reach_path = cwd + '/pnc/reachability_map/output/' + robot_name + '/' + robot_name
    ee_halfspace_params = OrderedDict()
    for fr in plan_to_model_frames.keys():
        ee_halfspace_params[fr] = reach_path + '_' + fr + '.yaml'

    # Load robot model (kinematics pass)
    rob_model, col_model, vis_model, rob_data, col_data, vis_data = \
        load_robot_model(package_dir, robot_urdf_file)

    # Root → torso offset (needed for traversable regions)
    geom_model = pin.buildGeomFromUrdf(rob_model, robot_urdf_file, pin.GeometryType.COLLISION)
    geom_model.addAllCollisionPairs()
    root_to_torso_offset = get_root_to_torso_offset(geom_model)  # defined above

    # Frame IDs
    plan_to_model_ids = {}
    for key, frame_name in plan_to_model_frames.items():
        plan_to_model_ids[key] = rob_model.getFrameId(frame_name)

    # Environment
    door_pos = np.array([0.32, 0.0, 0.])
    obstructed_hole = HoleInWallObstructed(door_pos)

    # Refined collision model for dynamics + merge with hole obstacles
    dyn_rob_model, dyn_col_model, dyn_vis_model, dyn_rob_data, dyn_col_data, dyn_vis_data = \
        load_robot_model(package_dir, refined_urdf_file)
    refined_geom_model = pin.buildGeomFromUrdf(
        dyn_rob_model, refined_urdf_file, pin.GeometryType.COLLISION)
    refined_geom_model.addAllCollisionPairs()

    door_model, _, __ = load_navy_door_models()
    door_col_model = pin.buildGeomFromUrdf(door_model, env_urdf_path, pin.GeometryType.COLLISION)
    for i_col, col_obj in enumerate(obstructed_hole.obstacles[1:]):
        obstacle_geom = hpoly_to_fcl_collision(col_obj)
        new_geom = create_convex_geom_from_copy(
            door_col_model.geometryObjects[0],
            obstacle_geom,
            'hole_obstacle_' + str(i_col))
        refined_geom_model.addGeometryObject(new_geom)
    refined_geom_model.addAllCollisionPairs()

    # SCA geometry (loaded once, shared across trials)
    sca_geometry = None
    if B_USE_SELF_COLLISION_AVOIDANCE:
        sca_geometry = SCARobotGeometry(package_dir, robot_urdf_file, plan_to_model_frames)

    planner_params = g1_params.MultiContactDoorConfig()

    return dict(
        package_dir=package_dir,
        robot_urdf_file=robot_urdf_file,
        refined_urdf_file=refined_urdf_file,
        plan_to_model_frames=plan_to_model_frames,
        plan_to_model_ids=plan_to_model_ids,
        force_joint_frames=force_joint_frames,
        aux_frames_path=aux_frames_path,
        ee_halfspace_params=ee_halfspace_params,
        rob_model=rob_model,
        col_model=col_model,
        vis_model=vis_model,
        root_to_torso_offset=root_to_torso_offset,
        dyn_rob_model=dyn_rob_model,
        refined_geom_model=refined_geom_model,
        obstructed_hole=obstructed_hole,
        sca_geometry=sca_geometry,
        planner_params=planner_params,
    )


# ---------------------------------------------------------------------------
# Per-trial planner run
# ---------------------------------------------------------------------------

def run_trial(y_pos: float, shared: dict) -> dict:
    """
    Run a single KIN + DYN feasibility trial for the given floating-base y position.

    Returns a dict with:
      - 'y_pos'       : the sampled y starting position
      - 'is_optimal'  : bool – True if the HumanoidMulticontactPlanner solver
                        reported a feasible (optimal) solution (fddp.isFeasible)
      - 'solver_type' : which solver path was taken ('single', 'sca', ...)
      - 'solve_time'  : total dynamic-planning wall-clock time (seconds)
      - 'error'       : None on success, exception string on failure
    """
    result = {'y_pos': y_pos, 'is_optimal': False, 'solver_type': None,
              'solve_time': None, 'error': None}
    try:
        rob_model         = shared['rob_model']
        col_model            = shared['col_model']
        vis_model            = shared['vis_model']
        plan_to_model_frames = shared['plan_to_model_frames']
        plan_to_model_ids    = shared['plan_to_model_ids']
        aux_frames_path      = shared['aux_frames_path']
        ee_halfspace_params  = shared['ee_halfspace_params']
        root_to_torso_offset = shared['root_to_torso_offset']
        package_dir          = shared['package_dir']
        robot_urdf_file      = shared['robot_urdf_file']
        refined_urdf_file    = shared['refined_urdf_file']
        dyn_rob_model        = shared['dyn_rob_model']
        refined_geom_model   = shared['refined_geom_model']
        obstructed_hole      = shared['obstructed_hole']
        sca_geometry         = shared['sca_geometry']
        planner_params       = shared['planner_params']
        visualizer           = shared.get('visualizer')

        # ---- Initial configuration ----
        q0 = get_g1_pose_with_y(rob_model.nq - 7, y_pos)
        v0 = np.zeros(rob_model.nv)
        x0 = np.concatenate([q0, v0])

        # ---- Forward kinematics (fresh robot system per trial) ----
        robot_fwdk = PinocchioRobotSystem(robot_urdf_file, package_dir, False, False)
        cmd = robot_fwdk.create_cmd_ordered_dict(
            q0[7:], np.zeros(len(q0[7:])), np.zeros(len(q0[7:])))
        robot_fwdk.update_system(
            None, None, None, None,
            q0[:3], q0[3:7], np.zeros(3), np.zeros(3),
            cmd["joint_pos"], cmd["joint_vel"])

        starting_pose = {}
        for fr in plan_to_model_frames.keys():
            starting_pose[fr] = robot_fwdk.get_link_iso(plan_to_model_frames[fr])[:3, 3]

        # ---- Build contact sequence ----
        fixed_frames_seq, motion_frames_seq = build_knocker_contact_seq(
            starting_pose, B_USE_KNEES)
        contact_seqs = get_contact_seq_from_fixed_frames_seq(fixed_frames_seq)
        contact_seq_planes = get_contact_planes_from_motion_frames_seq(
            contact_seqs, motion_frames_seq)

        # ---- IRIS regions ----
        safe_regions_mgr_dict = hole_plan.compute_iris_regions_mgr(
            obstructed_hole, starting_pose, motion_frames_seq, b_use_knees=B_USE_KNEES)

        # ---- IK frame planner ----
        standing_pos = q0[:3]
        rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)
        ik_cfree_planner = IKCFreePlanner(
            rob_model, rob_data, plan_to_model_frames, q0, planner_params)

        traversable_regions_dict = OrderedDict()
        for fr in plan_to_model_frames.keys():
            if fr == 'torso':
                traversable_regions_dict[fr] = FrameTraversableRegion(
                    fr, b_visualize_reach=B_VISUALIZE_KIN, b_visualize_safe=B_VISUALIZE_KIN,
                    visualizer=visualizer)
            else:
                traversable_regions_dict[fr] = FrameTraversableRegion(
                    fr, ee_halfspace_params[fr],
                    b_visualize_reach=B_VISUALIZE_KIN, b_visualize_safe=B_VISUALIZE_KIN,
                    visualizer=visualizer,
                    root_to_torso_pos=root_to_torso_offset)
                traversable_regions_dict[fr].update_origin_pose(standing_pos)
            traversable_regions_dict[fr].load_iris_regions(safe_regions_mgr_dict[fr])

        if B_USE_KNEES:
            traversable_regions = [traversable_regions_dict[fr]
                                   for fr in ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']]
        else:
            traversable_regions = [traversable_regions_dict[fr]
                                   for fr in ['torso', 'LF', 'RF', 'LH', 'RH']]

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
        T = 3
        ik_cfree_planner.plan(p_init, T, planner_params, visualizer, B_VERBOSE)

        # ---- Build dynamic planner (needed for lf_targets before plan()) ----
        N_horizon_lst   = planner_params.N_HORIZON_LST
        contact_sequence = ContactSequence(contact_seq_planes, N_horizon_lst, T)

        robot_dyn_plan = G1MulticontactPlanner(
            dyn_rob_model, contact_sequence, ik_cfree_planner,
            planner_params, refined_geom_model)
        robot_dyn_plan.set_zero_configuration(q0)

        # ---- Kinematic animation ----
        if B_VISUALIZE_KIN:
            N_knots = len(robot_dyn_plan.lf_targets)
            n_contacts = len(N_horizon_lst)
            save_freq = 10
            kin_display = vis_tools.MeshcatPinocchioAnimation(
                rob_model, col_model, vis_model,
                rob_data, vis_data, col_data,
                ctrl_freq=N_knots / (n_contacts * T), save_freq=save_freq)
            kin_display.add_shapes_from(obstructed_hole.obstacles)
            kin_display.start_animation()
            for t_anim in np.linspace(0, n_contacts * T, N_knots // save_freq):
                frame_targets_dict = ik_cfree_planner.get_frame_targets_from_kin_planner(
                    int(t_anim / T), t_anim)
                kin_display.animate_single_collision(
                    plan_to_model_frames['torso'] + '_0', frame_targets_dict['torso'])
                kin_display.animate_single_collision(
                    plan_to_model_frames['LH'] + '_0', frame_targets_dict['LH'])
                kin_display.animate_single_collision(
                    plan_to_model_frames['RH'] + '_0', frame_targets_dict['RH'])
                if B_USE_KNEES:
                    kin_display.animate_single_collision(
                        plan_to_model_frames['L_knee'] + '_0', frame_targets_dict['L_knee'])
                    kin_display.animate_single_collision(
                        plan_to_model_frames['R_knee'] + '_0', frame_targets_dict['R_knee'])
                kin_display.animate_single_collision(
                    plan_to_model_frames['LF'] + '_0', frame_targets_dict['LF'])
                kin_display.animate_single_collision(
                    plan_to_model_frames['RF'] + '_0', frame_targets_dict['RF'])
                kin_display.animate_target("lfoot_target", [frame_targets_dict['LF']], [1, 1, 0])
                if B_USE_KNEES:
                    kin_display.animate_target("lknee_target", [frame_targets_dict['L_knee']], [0, 0, 1])
                    kin_display.animate_target("rknee_target", [frame_targets_dict['R_knee']], [0, 0, 1])
                kin_display.animate_target("rfoot_target", [frame_targets_dict['RF']], [1, 1, 0])
                kin_display.animate_target("lhand_target", [frame_targets_dict['LH']], [0.5, 0, 0])
                kin_display.animate_target("rhand_target", [frame_targets_dict['RH']], [0.5, 0, 0])
                kin_display.animate_target("base_target", [frame_targets_dict['torso']], [0, 0.5, 0])
                kin_display.animation_step()
            kin_display.hide_visuals(["g1_29dof_lock_waist/visuals"])
            kin_display.hide_visuals(["g1_29dof_lock_waist/collisions"], True)
            kin_display.finish_animation()

        # ---- Dynamic feasibility ----
        robot_dyn_plan.set_plan_to_model_params(plan_to_model_ids)
        robot_dyn_plan.set_initial_configuration(x0)

        t_start = time.time()
        robot_dyn_plan.plan(
            b_solve_hybrid=B_SOLVE_HYBRID,
            integration_type='Euler',
            sca_refinement=B_SCA_REFINEMENT,
            b_solve_by_sections=SOLVE_BY_SECTIONS,
            solver_type=SOLVER_TYPE,
            b_use_knees=B_USE_KNEES)
        result['solve_time'] = time.time() - t_start

        # ---- Collect optimality flag ----
        latest_fddp = robot_dyn_plan.get_latest_fddp()
        result['solver_type'] = robot_dyn_plan.solver_type
        if latest_fddp is not None:
            result['is_optimal'] = bool(robot_dyn_plan.b_sca_converges)

    except Exception as exc:
        result['error'] = traceback.format_exc()
        print(f"[y={y_pos:.4f}] Trial FAILED: {exc}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)
    y_samples = rng.uniform(Y_LB, Y_UB, size=N_TRIALS)

    print("=" * 60)
    print(f"G1 obstructed-hole y-robustness test")
    print(f"  Trials : {N_TRIALS}")
    print(f"  y range: [{Y_LB}, {Y_UB}]  (seed={RNG_SEED})")
    print(f"  Samples: {np.round(y_samples, 4).tolist()}")
    print("=" * 60)

    # One-time setup
    shared = setup_g1_obstructed_hole()

    if B_VISUALIZE_KIN:
        q0_default = get_g1_pose_with_y(shared['rob_model'].nq - 7, 0.0)
        shared['visualizer'] = setup_visualizer(
            shared['rob_model'], shared['col_model'], shared['vis_model'], q0_default)
    else:
        shared['visualizer'] = None

    results = []
    for trial_idx, y_pos in enumerate(y_samples):
        print(f"\n{'─'*60}")
        print(f"Trial {trial_idx + 1}/{N_TRIALS}  |  y = {y_pos:.4f}")
        print(f"{'─'*60}")
        res = run_trial(float(y_pos), shared)
        results.append(res)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY – optimal solution found (HumanoidMulticontactPlanner)")
    print("=" * 60)
    print(f"{'Trial':>6}  {'y_pos':>8}  {'Optimal':>8}  {'Solver':>8}  {'Time(s)':>9}  Notes")
    print("-" * 60)

    optimal_trials   = []
    suboptimal_trials = []
    failed_trials    = []

    for i, res in enumerate(results):
        flag  = "YES" if res['is_optimal'] else "NO"
        stype = res['solver_type'] if res['solver_type'] else "—"
        tstr  = f"{res['solve_time']:.1f}" if res['solve_time'] is not None else "—"
        note  = "EXCEPTION" if res['error'] else ""
        print(f"{i+1:>6}  {res['y_pos']:>8.4f}  {flag:>8}  {stype:>8}  {tstr:>9}  {note}")
        if res['error']:
            failed_trials.append(i + 1)
        elif res['is_optimal']:
            optimal_trials.append(i + 1)
        else:
            suboptimal_trials.append(i + 1)

    print("=" * 60)
    print(f"Optimal    ({len(optimal_trials)}/{N_TRIALS}): trials {optimal_trials}")
    print(f"Suboptimal ({len(suboptimal_trials)}/{N_TRIALS}): trials {suboptimal_trials}")
    print(f"Errors     ({len(failed_trials)}/{N_TRIALS}): trials {failed_trials}")
    print("=" * 60)


if __name__ == "__main__":
    main()
