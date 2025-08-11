import os
import pickle
import sys
from collections import OrderedDict

import config.multicontact.g1_planner_config as g1_params
import config.multicontact.ergoCub_planner_config as ergoCub_params
import config.multicontact.valkyrie_planner_config as valkyrie_params
from pnc.planner.multicontact.kin_feasibility import SCARobotGeometry
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
from util.environment_creator import TiltedStairs
import pnc.planner.multicontact.contact_sequence_plans.tilted_stairs_plans as stairs_plan

cwd = os.getcwd()
sys.path.append(cwd)

import crocoddyl
import numpy as np

# Collision free description
from pydrake.geometry.optimization import HPolyhedron

# Kinematic feasibility
from pnc.planner.multicontact.kin_feasibility.frame_traversable_region import FrameTraversableRegion
from pnc.planner.multicontact.kin_feasibility.planner_surface_contact import PlannerSurfaceContact, \
    MotionFrameSequencer, get_contact_seq_from_fixed_frames_seq, get_contact_planes_from_motion_frames_seq
from pnc.planner.multicontact.kin_feasibility.ik_cfree_planner import *
# Tools for dynamic feasibility
from humanoid_action_models import *
from pnc.planner.multicontact.dyn_feasibility.G1MulticontactPlanner import G1MulticontactPlanner
from pnc.planner.multicontact.dyn_feasibility.ErgoCubMulticontactPlanner import ErgoCubMulticontactPlanner
from pnc.planner.multicontact.dyn_feasibility.ValkyrieMulticontactPlanner import ValkyrieMulticontactPlanner
from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import ContactSequence

# Visualization tools
import matplotlib.pyplot as plt
from plot.helper import plot_vector_traj, Fxyz_labels
import plot.meshcat_utils as vis_tools
from plot.multiontact_plotter import MulticontactPlotter
from vision.iris.iris_regions_manager import IrisRegionsManager, IrisGeomInterface
# Save data
from plot.data_saver import *

B_SHOW_JOINT_PLOTS = False
B_SHOW_JOINT_LIM_PLOTS = True
B_SHOW_COST_PLOTS = False
B_SHOW_GRF_PLOTS = False
B_VISUALIZE = False
B_SAVE_KIN_DATA = False
B_SAVE_DYN_DATA = False
B_VERBOSE = False
B_SAVE_HTML = False
B_USE_SELF_COLLISION_AVOIDANCE = False
B_USE_KNEES = True
B_USE_KNEES_IN_SMOOTH_PLAN = True


def get_g1_default_initial_pose(n_joints:int, env: str = 'door'):
    if env == 'door':
        q0 = np.zeros(n_joints, )
        q0[0] = -0.697  # left_hip_pitch_joint
        # q0[1] = np.radians(hip_yaw_angle)  # left_hip_roll_joint
        # q0[2] = np.radians(hip_yaw_angle)  # left_hip_yaw_joint
        q0[3] = 1.23  # left_knee_joint
        q0[4] = -0.53  # left_ankle_pitch_joint
        # q0[5] = np.radians(-hip_yaw_angle)  # left_ankle_roll_joint
        q0[6] = -0.697  # right_hip_pitch_joint
        # q0[7] = np.pi / 6  # right_hip_roll_joint
        # q0[8] = 0.  # right_hip_yaw_joint
        q0[9] = 1.23  # right_knee_joint
        q0[10] = -0.53  # right_ankle_pitch_joint
        # q0[11] = 0.  # right_ankle_roll_joint

        floating_base = np.array([0., 0., 0.68, 0., 0., 0., 1.])
    elif env == 'stairs':
        q0 = np.zeros(n_joints, )
        q0[0] = -np.pi/6
        q0[3] = np.pi/3
        q0[4] = -np.pi/6
        q0[6] = -np.pi/6
        q0[9] = np.pi/3
        q0[10] = -np.pi/6
        floating_base = np.array([0., 0., 0.73, 0., 0., 0., 1.])
    else:
        raise ValueError(f"Unspecified default initial pose for g1 in environment: {env}")
    return np.concatenate((floating_base, q0))


def get_val_default_initial_pose(n_joints):
    q0 = np.zeros(n_joints, )
    hip_pitch_angle = 35.
    # q0[0] = 0  #     "leftHipYaw",
    # q0[1] = np.radians(hip_yaw_angle)     # "leftHipRoll",
    q0[2] = -np.radians(hip_pitch_angle)    # "leftHipPitch",
    q0[3] = 2*np.radians(hip_pitch_angle)   # "leftKneePitch",
    q0[4] = -np.radians(hip_pitch_angle)    # "leftAnklePitch",
    # q0[5] = np.radians(-hip_yaw_angle)    # "leftAnkleRoll",
    # q0[6] = 0.                            # "rightHipYaw",
    # q0[7] = np.pi / 6                     # "rightHipRoll",
    q0[8] = -np.radians(hip_pitch_angle)    # "rightHipPitch",
    q0[9] = 2*np.radians(hip_pitch_angle)   # "rightKneePitch",
    q0[10] = -np.radians(hip_pitch_angle)   # "rightAnklePitch",
    # q0[11] = 0.                           # "rightAnkleRoll",
    # q0[12] = 0.                           # "torsoYaw",
    # q0[13] = 0.                           # "torsoPitch",
    # q0[14] = np.radians(-hip_yaw_angle)   # "torsoRoll",
    # q0[15] = -np.pi / 4                   # "leftShoulderPitch",
    q0[16] = -np.pi / 2                     # "leftShoulderRoll",
    # q0[17] = np.pi / 4                    # "leftShoulderYaw",
    q0[18] = -np.pi / 2                     # "leftElbowPitch",
    # q0[19] = np.radians(hip_yaw_angle)    # "lowerNeckPitch",
    # q0[20] = 0.                           # "neckYaw",
    # q0[21] = -np.pi / 6                   # "upperNeckPitch",
    # q0[22] = 0.                           # "rightShoulderPitch",
    q0[23] = np.pi / 2                      # "rightShoulderRoll",
    # q0[24] = np.pi/3.                     # "rightShoulderYaw",
    q0[25] = np.pi / 2.                     # "rightElbowPitch"

    floating_base = np.array([0., 0., 1.01, 0., 0., 0., 1.])
    return np.concatenate((floating_base, q0))


def get_ergoCub_default_initial_pose(n_joints):
    q0 = np.zeros(n_joints, )
    hip_pitch_angle = 35.
    hip_roll_angle = 10.
    q0[0] = np.radians(hip_pitch_angle)     # "l_hip_pitch"
    q0[1] = np.radians(hip_roll_angle)     # "l_hip_roll"
    # q0[2] = -np.radians(hip_pitch_angle)    # "l_hip_yaw"
    q0[3] = -2*np.radians(hip_pitch_angle)   # "l_knee"
    q0[4] = -np.radians(hip_pitch_angle)    # "l_ankle_pitch"
    q0[5] = -np.radians(hip_roll_angle)    # "l_ankle_roll"
    q0[6] = np.radians(hip_pitch_angle)    # "r_hip_pitch"
    q0[7] = np.radians(hip_roll_angle)      # "r_hip_roll"
    # q0[8] = -np.radians(hip_pitch_angle)  # "r_hip_yaw"
    q0[9] = -2*np.radians(hip_pitch_angle)   # "r_knee"
    q0[10] = -np.radians(hip_pitch_angle)   # "r_ankle_pitch"
    q0[11] = -np.radians(hip_roll_angle)    # "r_ankle_roll"
    # q0[12] = 0.                           # "torso_roll"
    # q0[13] = 0.                           # "torso_pitch"
    # q0[14] = np.radians(-hip_yaw_angle)   # "torso_yaw"
    # q0[15] = -np.pi / 2                     # "l_shoulder_pitch"
    # q0[16] = -np.pi / 2                     # "l_shoulder_roll"
    # q0[17] = np.pi / 4                    # "l_shoulder_yaw"
    q0[18] = np.pi / 2                     # "l_elbow"
    # q0[19] = np.radians(hip_yaw_angle)    # "l_wrist_yaw"
    # q0[20] = 0.                           # "l_wrist_roll"
    # q0[21] = -np.pi / 6                   # "l_wrist_pitch"
    # q0[22] = 0.                           # "neck_pitch",
    # q0[23] = 0.                           # "neck_roll",
    # q0[24] = 0.                           # "neck_yaw",
    # q0[25] = 0.                           # "camera_tilt",
    # q0[26] = -np.pi / 2                   # "r_shoulder_pitch",
    # q0[27] = 0.                           # "r_shoulder_roll",
    # q0[28] = 0.                           # "r_shoulder_yaw",
    q0[29] = np.pi / 2                      # "r_elbow",
    # q0[30] = 0.                           # "r_wrist_yaw",
    # q0[31] = 0.                           # "r_wrist_roll",
    # q0[32] = 0.                           # "r_wrist_pitch",

    floating_base = np.array([0., 0., 0.65, 0., 0., 0., 1.])
    return np.concatenate((floating_base, q0))


# def load_orig_navy_env(door_pos):
#     # create navy door environment
#     door_quat = np.array([0., 0., 0.7071068, 0.7071068])
#     door_width = np.array([0.03, 0., 0.])
#     dom_ubody_lb = np.array([-1.6, -0.8, 0.5])
#     dom_ubody_ub = np.array([1.6, 0.8, 2.1])
#     dom_lbody_lb = np.array([-1.6, -0.8, -0.])
#     dom_lbody_ub = np.array([1.6, 0.8, 1.2])
#     floor = HPolyhedron.MakeBox(
#         np.array([-2, -0.9, -0.05]) + door_pos + door_width,
#         np.array([2, 0.9, -0.001]) + door_pos + door_width)
#     knee_knocker_base = HPolyhedron.MakeBox(
#         np.array([-0.05, -0.9, 0.0]) + door_pos + door_width,
#         np.array([0.06, 0.9, 0.4]) + door_pos + door_width)
#     knee_knocker_lwall = HPolyhedron.MakeBox(
#         np.array([-0.025, 0.9 - 0.518, 0.0]) + door_pos + door_width,
#         np.array([0.025, 0.9, 2.2]) + door_pos + door_width)
#     knee_knocker_rwall = HPolyhedron.MakeBox(
#         np.array([-0.025, -0.9, 0.0]) + door_pos + door_width,
#         np.array([0.025, -(0.9 - 0.518), 2.2]) + door_pos + door_width)
#     knee_knocker_top = HPolyhedron.MakeBox(
#         np.array([-0.025, -0.9, 1.85]) + door_pos + door_width,
#         np.array([0.025, 0.9, 2.25]) + door_pos + door_width)
#     obstacles = [floor,
#                       knee_knocker_base,
#                       knee_knocker_lwall,
#                       knee_knocker_rwall,
#                       # knee_knocker_llip,
#                       # knee_knocker_rlip,
#                       knee_knocker_top]
#     domain_ubody = HPolyhedron.MakeBox(dom_ubody_lb, dom_ubody_ub)
#     domain_lbody = HPolyhedron.MakeBox(dom_lbody_lb, dom_lbody_ub)
#
#     door_pose = np.concatenate((door_pos, door_quat))
#     return door_pose, obstacles, domain_ubody, domain_lbody


def load_navy_env(robot_name, door_pos):
    # create navy door environment
    door_quat = np.array([0., 0., 0.7071068, 0.7071068])
    door_width = np.array([0.03, 0., 0.])
    dom_ubody_lb = np.array([-1.6, -0.8, 0.5])
    dom_ubody_ub = np.array([1.6, 0.8, 2.1])

    # account for different robot feet dimensions and restrict inwards motion
    if robot_name == 'g1':
        dom_lbody_lb_l = np.array([-1.6, -0.8, -0.])
        dom_lbody_lb_r = np.array([-1.6, -0.8, -0.])
        dom_lbody_ub_l = np.array([1.6, 0.8, 1.2])
        dom_lbody_ub_r = np.array([1.6, 0.8, 1.2])
        knee_knocker_base = HPolyhedron.MakeBox(
            np.array([-0.05, -0.9, 0.0]) + door_pos + door_width,
            np.array([0.06, 0.9, 0.4]) + door_pos + door_width)
    elif robot_name == 'valkyrie':
        dom_lbody_lb_l = np.array([-1.6, -0.05, -0.])
        dom_lbody_lb_r = np.array([-1.6, -0.8, -0.])
        dom_lbody_ub_l = np.array([1.6, 0.8, 1.2])
        dom_lbody_ub_r = np.array([1.6, 0.05, 1.2])
        knee_knocker_base = HPolyhedron.MakeBox(
            np.array([-0.06, -0.9, 0.0]) + door_pos + door_width,
            np.array([0.12, 0.9, 0.45]) + door_pos + door_width)
    elif robot_name == 'ergoCub':
        dom_lbody_lb_l = np.array([-1.6, -0.8, -0.])
        dom_lbody_lb_r = np.array([-1.6, -0.8, -0.])
        dom_lbody_ub_l = np.array([1.6, 0.8, 1.0])
        dom_lbody_ub_r = np.array([1.6, 0.8, 1.0])
        knee_knocker_base = HPolyhedron.MakeBox(
            np.array([-0.065, -0.9, 0.0]) + door_pos + door_width,
            np.array([0.065, 0.9, 0.47]) + door_pos + door_width)
    else:   # default
        dom_lbody_lb_l = np.array([-1.6, -0.05, -0.])
        dom_lbody_ub_r = np.array([1.6, 0.8, 1.2])
        knee_knocker_base = HPolyhedron.MakeBox(
            np.array([-0.05, -0.9, 0.0]) + door_pos + door_width,
            np.array([0.12, 0.9, 0.41]) + door_pos + door_width)
    floor = HPolyhedron.MakeBox(
        np.array([-2, -0.9, -0.05]) + door_pos + door_width,
        np.array([2, 0.9, -0.001]) + door_pos + door_width)
    knee_knocker_lwall = HPolyhedron.MakeBox(
        np.array([-0.025, 0.9 - 0.518, 0.0]) + door_pos + door_width,
        np.array([0.025, 0.9, 2.2]) + door_pos + door_width)
    knee_knocker_rwall = HPolyhedron.MakeBox(
        np.array([-0.025, -0.9, 0.0]) + door_pos + door_width,
        np.array([0.025, -(0.9 - 0.518), 2.2]) + door_pos + door_width)
    knee_knocker_top = HPolyhedron.MakeBox(
        np.array([-0.025, -0.9, 1.85]) + door_pos + door_width,
        np.array([0.025, 0.9, 2.25]) + door_pos + door_width)
    # knee_knocker_llip = HPolyhedron.MakeBox(
    #     np.array([-0.035, 0.9 - 0.518, 0.25]) + door_pos + door_width,
    #     np.array([0.035, 0.9 - 0.518 + 0.15, 2.0]) + door_pos + door_width)
    # knee_knocker_rlip = HPolyhedron.MakeBox(
    #     np.array([-0.035, -(0.9 - 0.518 + 0.15), 0.25]) + door_pos + door_width,
    #     np.array([0.035, -(0.9 - 0.518), 2.0]) + door_pos + door_width)
    obstacles = [floor,
                      knee_knocker_base,
                      knee_knocker_lwall,
                      knee_knocker_rwall,
                      # knee_knocker_llip,
                      # knee_knocker_rlip,
                      knee_knocker_top]
    domain_ubody = HPolyhedron.MakeBox(dom_ubody_lb, dom_ubody_ub)
    domain_lbody_l = HPolyhedron.MakeBox(dom_lbody_lb_l, dom_lbody_ub_l)
    domain_lbody_r = HPolyhedron.MakeBox(dom_lbody_lb_r, dom_lbody_ub_r)

    door_pose = np.concatenate((door_pos, door_quat))
    return door_pose, obstacles, domain_ubody, domain_lbody_l, domain_lbody_r


def load_robot_model(package_dir, urdf_file):
    rob_model, col_model, vis_model = pin.buildModelsFromUrdf(urdf_file,
                                                              package_dir,
                                                              pin.JointModelFreeFlyer())
    rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)

    return rob_model, col_model, vis_model, rob_data, col_data, vis_data


def load_navy_door_models():
    return pin.buildModelsFromUrdf(
        cwd + "/robot_model/ground/navy_door.urdf",
        cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

def compute_iris_regions_mgr(obstacles,
                             domain_ubody,
                             domain_lbody_l,
                             domain_lbody_r,
                             robot_data,
                             plan_to_model_ids,
                             standing_pos,
                             goal_step_length,
                             robot_name='g1',
                             root_to_torso_offset=np.array([0., 0., 0.])):
    # shift (feet) iris seed to get nicer IRIS region
    if robot_name == 'ergoCub':
        iris_lf_shift = np.array([0.25, 0., 0.])
        iris_rf_shift = np.array([0.25, 0., 0.])
        iris_kn_shift = np.array([0.05, 0., -0.15])
        iris_kn_end_shift = np.array([-0.28 , 0., -0.25])
        iris_ft_goal_shift = np.array([-0.03, 0., 0.])
    else:
        iris_lf_shift = np.array([0.1, 0., 0.])
        iris_rf_shift = np.array([0.1, 0., 0.])
        iris_kn_shift = np.array([0.05, 0., -0.05])
        iris_kn_end_shift = np.array([-0.15 , 0., -0.2])
        iris_ft_goal_shift = np.array([0., 0., 0.])
    lhand_door_inner = np.array([0.3, 0.35, 0.9])
    rhand_door_inner = np.array([0.3, -0.35, 0.9])

    # get end effector positions via fwd kin
    starting_torso_pos = standing_pos + root_to_torso_offset
    final_torso_pos = starting_torso_pos + np.array([goal_step_length, 0., 0.])
    starting_lf_pos = robot_data.oMf[plan_to_model_ids['LF']].translation
    final_lf_pos = starting_lf_pos + np.array([goal_step_length, 0., 0.])
    starting_lh_pos = robot_data.oMf[plan_to_model_ids['LH']].translation
    final_lh_pos = starting_lh_pos + np.array([goal_step_length, 0., 0.])
    starting_rf_pos = robot_data.oMf[plan_to_model_ids['RF']].translation
    final_rf_pos = starting_rf_pos + np.array([goal_step_length, 0., 0.])
    starting_rh_pos = robot_data.oMf[plan_to_model_ids['RH']].translation
    final_rh_pos = starting_rh_pos + np.array([goal_step_length, 0., 0.])
    starting_lkn_pos = robot_data.oMf[plan_to_model_ids['L_knee']].translation
    final_lkn_pos = starting_lkn_pos + np.array([goal_step_length, 0., 0.])
    starting_rkn_pos = robot_data.oMf[plan_to_model_ids['R_knee']].translation
    final_rkn_pos = starting_rkn_pos + np.array([goal_step_length, 0., 0.])

    safe_torso_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_torso_pos)
    safe_torso_end_region = IrisGeomInterface(obstacles, domain_ubody, final_torso_pos)
    safe_lf_start_region = IrisGeomInterface(obstacles, domain_lbody_l, starting_lf_pos + iris_lf_shift)
    safe_lf_end_region = IrisGeomInterface(obstacles, domain_lbody_l, final_lf_pos + iris_ft_goal_shift)
    safe_lk_start_region = IrisGeomInterface(obstacles, domain_lbody_l, starting_lkn_pos + iris_kn_shift)
    safe_lk_end_region = IrisGeomInterface(obstacles, domain_lbody_l, final_lkn_pos + iris_kn_end_shift)
    safe_lh_start_region = IrisGeomInterface(obstacles, domain_ubody, lhand_door_inner)
    # safe_lh_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_lh_pos + np.array([0.1, 0., 0.]))
    safe_lh_end_region = IrisGeomInterface(obstacles, domain_ubody, final_lh_pos)
    safe_rf_start_region = IrisGeomInterface(obstacles, domain_lbody_r, starting_rf_pos + iris_rf_shift)
    safe_rf_end_region = IrisGeomInterface(obstacles, domain_lbody_r, final_rf_pos + iris_ft_goal_shift)
    safe_rk_start_region = IrisGeomInterface(obstacles, domain_lbody_r, starting_rkn_pos + iris_kn_shift)
    safe_rk_end_region = IrisGeomInterface(obstacles, domain_lbody_r, final_rkn_pos + iris_kn_end_shift)
    safe_rh_start_region = IrisGeomInterface(obstacles, domain_ubody, rhand_door_inner)
    # safe_rh_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_rh_pos + np.array([0.1, 0., 0.]))
    safe_rh_end_region = IrisGeomInterface(obstacles, domain_ubody, final_rh_pos)
    safe_regions_mgr_dict = {'torso': IrisRegionsManager(safe_torso_start_region, safe_torso_end_region),
                             'LF': IrisRegionsManager(safe_lf_start_region, safe_lf_end_region),
                             'L_knee': IrisRegionsManager(safe_lk_start_region, safe_lk_end_region),
                             'LH': IrisRegionsManager(safe_lh_start_region, safe_lh_end_region),
                             'RF': IrisRegionsManager(safe_rf_start_region, safe_rf_end_region),
                             'R_knee': IrisRegionsManager(safe_rk_start_region, safe_rk_end_region),
                             'RH': IrisRegionsManager(safe_rh_start_region, safe_rh_end_region)}

    # compute and connect IRIS from start to goal
    start_iris_compute_time = time.time()
    for _, irm in safe_regions_mgr_dict.items():
        irm.computeIris()
        irm.connectIrisSeeds()
    print("IRIS computation time: ", time.time() - start_iris_compute_time)

    # save initial/final EE positions
    p_init = {}
    p_init['torso'] = starting_torso_pos
    p_init['LF'] = starting_lf_pos
    p_init['RF'] = starting_rf_pos
    p_init['L_knee'] = starting_lkn_pos
    p_init['R_knee'] = starting_rkn_pos
    p_init['LH'] = starting_lh_pos
    p_init['RH'] = starting_rh_pos

    return safe_regions_mgr_dict, p_init


def get_two_stage_contact_sequence(safe_regions_mgr_dict):
    starting_lh_pos = safe_regions_mgr_dict['LH'].iris_list[0].seed_pos
    starting_rh_pos = safe_regions_mgr_dict['RH'].iris_list[0].seed_pos
    final_lf_pos = safe_regions_mgr_dict['LF'].iris_list[1].seed_pos
    final_rf_pos = safe_regions_mgr_dict['RF'].iris_list[1].seed_pos
    intermediate_lh_pos_door = np.array([0.32, 0.37, 0.9])
    final_torso_pos = safe_regions_mgr_dict['torso'].iris_list[1].seed_pos
    final_lkn_pos = safe_regions_mgr_dict['L_knee'].iris_list[1].seed_pos
    final_rkn_pos = safe_regions_mgr_dict['R_knee'].iris_list[1].seed_pos

    # initialize fixed and motion frame sets
    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Step 1: L hand to frame
    # if B_USE_KNEES:
    #     fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    # else:
    #     fixed_frames.append(['LF', 'RF'])   # frames that must not move
    # motion_frames_seq.add_motion_frame({'LH': intermediate_lh_pos_door})
    # lh_contact_front = PlannerSurfaceContact('LH', np.array([-1, 0, 0]))
    # lh_contact_front.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
    # motion_frames_seq.add_contact_surface(lh_contact_front)

    # ---- Step 2: step through door with left foot
    fixed_frames.append(['RF', 'R_knee'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
        'LF': final_lf_pos,
        'L_knee': final_lkn_pos,
        # 'torso': final_torso_pos + np.array([0.2, 0., 0.]),  # testing
        'LH': starting_lh_pos + np.array([0.2, -0.1, 0.2]),  # testing
        'RH': starting_rh_pos + np.array([0.2, 0.1, 0.2])})  # testing
    lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces(lf_contact_over)

    # ---- Step 3: re-position L/R hands for more stability
    # fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    # motion_frames_seq.add_motion_frame({
    #                     'LH': starting_lh_pos + np.array([0.09, 0.06, 0.18]),
    #                     'RH': starting_rh_pos + np.array([0.09, -0.06, 0.18])})
    # lh_contact_inside = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    # lh_contact_inside.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
    # rh_contact_inside = PlannerSurfaceContact('RH', np.array([0, 1, 0]))
    # motion_frames_seq.add_contact_surface([lh_contact_inside, rh_contact_inside])

    # ---- Step 4: step through door with right foot
    fixed_frames.append(['LF', 'L_knee'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        'RF': final_rf_pos,
                        'torso': final_torso_pos,
                        'R_knee': final_rkn_pos,
                        'LH': starting_lh_pos + np.array([0.4, 0., 0.]),
                        'RH': starting_rh_pos + np.array([0.4, 0., 0.])
    })
    rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces(rf_contact_over)

    # ---- Step 5: square up
    fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
    motion_frames_seq.add_motion_frame({})

    return fixed_frames, motion_frames_seq


def get_five_stage_one_hand_contact_sequence(robot_name, safe_regions_mgr_dict):
    ###### Previously used key locations
    # door_l_outer_location = np.array([0.45, 0.35, 1.2])
    # door_r_outer_location = np.array([0.45, -0.35, 1.2])
    if robot_name == 'g1':
        # G1 settings
        door_l_inner_location = np.array([0.34, 0.37, 0.9])
        door_r_inner_location = np.array([0.34, -0.37, 0.9])
    else:
        # ergoCub settings
        door_l_inner_location = np.array([0.3, 0.35, 1.0])
        door_r_inner_location = np.array([0.34, -0.35, 1.0])

    starting_lh_pos = safe_regions_mgr_dict['LH'].iris_list[0].seed_pos
    starting_rh_pos = safe_regions_mgr_dict['RH'].iris_list[0].seed_pos
    starting_torso_pos = safe_regions_mgr_dict['torso'].iris_list[0].seed_pos
    final_lf_pos = safe_regions_mgr_dict['LF'].iris_list[1].seed_pos
    final_lkn_pos = safe_regions_mgr_dict['L_knee'].iris_list[1].seed_pos
    final_rf_pos = safe_regions_mgr_dict['RF'].iris_list[1].seed_pos
    final_torso_pos = safe_regions_mgr_dict['torso'].iris_list[1].seed_pos
    final_rkn_pos = safe_regions_mgr_dict['R_knee'].iris_list[1].seed_pos
    final_rh_pos = safe_regions_mgr_dict['RH'].iris_list[1].seed_pos
    final_lh_pos = safe_regions_mgr_dict['LH'].iris_list[1].seed_pos

    # initialize fixed and motion frame sets
    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Step 1: L hand to frame
    fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    if robot_name == 'g1':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            # 'torso': starting_torso_pos + np.array([0.07, -0.07, 0.02])
                                            })
    elif robot_name == 'ergoCub':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            # 'torso': starting_torso_pos + np.array([0.05, -0.07, 0])
                                            })
    lh_contact_front = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    lh_contact_front.set_contact_breaking_velocity(np.array([0, -1, 0.]))
    motion_frames_seq.add_contact_surfaces([lh_contact_front])

    # ---- Step 2: step through door with left foot
    fixed_frames.append(['RF', 'R_knee', 'LH'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        'LF': final_lf_pos,
                        'L_knee': final_lf_pos + np.array([0.15, 0., 0.28])})
                        # 'L_knee': final_lkn_pos + np.array([-0.05, 0., 0.07])})
    lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([lf_contact_over])

    # ---- Step 3: re-position L/R hands for more stability
    fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        # 'LH': starting_lh_pos + np.array([0.3, 0., 0.0]),   # <-- G1
                        # 'LH': starting_lh_pos + np.array([0.35, 0.1, 0.0]),   # <-- other
                        # 'torso': final_torso_pos + np.array([-0.15, 0.05, 0.05]),     # good testing
                        'RH': door_r_inner_location})
    rh_contact_inside = PlannerSurfaceContact('RH', np.array([1, 0, 0]))
    motion_frames_seq.add_contact_surfaces([rh_contact_inside])

    # ---- Step 4: step through door with right foot
    # G1 settings
    # fixed_frames.append(['LF', 'L_knee', 'RH', 'LH'])   # frames that must not move
    # other settings
    fixed_frames.append(['LF', 'L_knee', 'RH'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        'RF': final_rf_pos,
                        'torso': final_torso_pos + np.array([0.0, 0., 0.04]),     # good testing
                        'R_knee': final_rf_pos + np.array([0.15, 0., 0.28]),
                        # 'R_knee': final_rkn_pos + np.array([-0.05, 0., 0.07]),
                        # 'LH': starting_lh_pos + np.array([0.35, 0.0, 0.0])
    })
    rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_contact_over])

    # ---- Step 5: square up
    # fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
    fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee'])
    motion_frames_seq.add_motion_frame({
        # 'torso': final_torso_pos,
        'RH': final_rh_pos, # + np.array([-0.20, 0., 0.]),
        'LH': final_lh_pos
    })

    return fixed_frames, motion_frames_seq


def get_five_stage_on_knocker_contact_sequence(robot_name: str,
                                               safe_regions_mgr_dict: dict[str: IrisRegionsManager]):
    ###### Previously used key locations
    # door_l_outer_location = np.array([0.45, 0.35, 1.2])
    # door_r_outer_location = np.array([0.45, -0.35, 1.2])
    if robot_name == 'g1':
        # G1 settings
        door_l_inner_location = np.array([0.3, 0.35, 1.0])
        door_r_inner_location = np.array([0.34, -0.35, 1.0])
        ft_kn_offset = np.array([0.15, 0., 0.28])
    else:
        # ergoCub settings
        door_l_inner_location = np.array([0.3, 0.35, 1.0])
        door_r_inner_location = np.array([0.34, -0.35, 1.0])
        ft_kn_offset = np.array([0.2, 0., 0.3])

    starting_lh_pos = safe_regions_mgr_dict['LH'].iris_list[0].seed_pos
    starting_rh_pos = safe_regions_mgr_dict['RH'].iris_list[0].seed_pos
    starting_torso_pos = safe_regions_mgr_dict['torso'].iris_list[0].seed_pos
    final_lf_pos = safe_regions_mgr_dict['LF'].iris_list[1].seed_pos
    final_lkn_pos = safe_regions_mgr_dict['L_knee'].iris_list[1].seed_pos
    final_rf_pos = safe_regions_mgr_dict['RF'].iris_list[1].seed_pos
    final_torso_pos = safe_regions_mgr_dict['torso'].iris_list[1].seed_pos
    final_rkn_pos = safe_regions_mgr_dict['R_knee'].iris_list[1].seed_pos
    final_rh_pos = safe_regions_mgr_dict['RH'].iris_list[1].seed_pos
    final_lh_pos = safe_regions_mgr_dict['LH'].iris_list[1].seed_pos
    if robot_name == 'g1':
        intermediate_rf_pos = np.array([0.35, final_rf_pos[1], 0.44])
        # intermediate_rh_pos = np.array([0.45, -0.2, 1.0])
    elif robot_name == 'ergoCub':
        intermediate_rf_pos = np.array([0.30, final_rf_pos[1]-0.02, 0.49])
        intermediate_rh_pos = np.array([0.40, -0.15, 1.0])

    # initialize fixed and motion frame sets
    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Step 1: L hand to frame
    fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    if robot_name == 'g1':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            })
    elif robot_name == 'ergoCub':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            })
    lh_contact_front = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    # lh_contact_front.set_contact_breaking_velocity(np.array([0, -1, 0.]))
    motion_frames_seq.add_contact_surfaces([lh_contact_front])

    # ---- Step 2: step on knee-knocker with right foot
    if robot_name == 'g1':
        fixed_frames.append(['LF', 'L_knee', 'LH'])   # frames that must not move
    elif robot_name == 'ergoCub':
        fixed_frames.append(['LF', 'L_knee', 'LH'])
    motion_frames_seq.add_motion_frame({
                        # 'RH': intermediate_rh_pos,      # added for ergoCub
                        'RF': intermediate_rf_pos,
                        'R_knee': intermediate_rf_pos + ft_kn_offset})
    rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_contact_over])

    # ---- Step 3: step through door with left foot
    if robot_name == 'g1':
        fixed_frames.append(['RF', 'R_knee', 'LH'])   # frames that must not move
    elif robot_name == 'ergoCub':
        fixed_frames.append(['RF', 'R_knee', 'LH'])  # frames that must not move
    motion_frames_seq.add_motion_frame({
                        'L_knee': final_lf_pos + ft_kn_offset,
                        'LF': final_lf_pos})
    lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([lf_contact_over])

    # ---- Step 4: balance / square up
    # fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
    fixed_frames.append(['LF', 'L_knee'])
    motion_frames_seq.add_motion_frame({
        'torso': final_torso_pos,
        'RF': final_rf_pos,
        'R_knee': final_rf_pos + ft_kn_offset,
        'RH': final_rh_pos,
        'LH': final_lh_pos
    })
    rf_square_up = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_square_up])

    # ---- Step 5: balance
    fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
    motion_frames_seq.add_motion_frame({})

    return fixed_frames, motion_frames_seq


def get_on_knocker_balanced_contact_sequence(robot_name: str,
                                            safe_regions_mgr_dict: dict[str: IrisRegionsManager]):
    if robot_name == 'g1':
        # G1 settings
        door_l_inner_location = np.array([0.3, 0.35, 1.0])
        door_r_inner_location = np.array([0.34, -0.35, 1.0])
        ft_kn_offset = np.array([0.15, 0., 0.28])
    else:
        # ergoCub settings
        door_l_inner_location = np.array([0.3, 0.35, 1.0])
        door_r_inner_location = np.array([0.34, -0.35, 1.0])
        ft_kn_offset = np.array([0.2, 0., 0.3])

    starting_lh_pos = safe_regions_mgr_dict['LH'].iris_list[0].seed_pos
    starting_rh_pos = safe_regions_mgr_dict['RH'].iris_list[0].seed_pos
    starting_torso_pos = safe_regions_mgr_dict['torso'].iris_list[0].seed_pos
    final_lf_pos = safe_regions_mgr_dict['LF'].iris_list[1].seed_pos
    final_lkn_pos = safe_regions_mgr_dict['L_knee'].iris_list[1].seed_pos
    final_rf_pos = safe_regions_mgr_dict['RF'].iris_list[1].seed_pos
    final_torso_pos = safe_regions_mgr_dict['torso'].iris_list[1].seed_pos
    final_rkn_pos = safe_regions_mgr_dict['R_knee'].iris_list[1].seed_pos
    final_rh_pos = safe_regions_mgr_dict['RH'].iris_list[1].seed_pos
    final_lh_pos = safe_regions_mgr_dict['LH'].iris_list[1].seed_pos
    if robot_name == 'g1':
        intermediate_rf_pos = np.array([0.35, final_rf_pos[1], 0.44])
    elif robot_name == 'ergoCub':
        intermediate_rf_pos = np.array([0.30, final_rf_pos[1]-0.02, 0.49])
        intermediate_rh_pos = np.array([0.40, -0.15, 1.0])

    # initialize fixed and motion frame sets
    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Step 1: L hand to frame
    fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    if robot_name == 'g1':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            'RH': door_r_inner_location,
                                            })
    elif robot_name == 'ergoCub':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            'RH': door_r_inner_location,
                                            })
    lh_contact_front = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    # lh_contact_front.set_contact_breaking_velocity(np.array([0, -1, 0.]))
    rh_contact_front = PlannerSurfaceContact('RH', np.array([0, 1, 0]))
    # rh_contact_front.set_contact_breaking_velocity(np.array([0, 1, 0.]))
    motion_frames_seq.add_contact_surfaces([lh_contact_front, rh_contact_front])

    # ---- Step 2: step on knee-knocker with right foot
    if robot_name == 'g1':
        fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])   # frames that must not move
    elif robot_name == 'ergoCub':
        fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])   # added RH back
    motion_frames_seq.add_motion_frame({
                        # 'RH': intermediate_rh_pos,      # added for ergoCub
                        'RF': intermediate_rf_pos,
                        'R_knee': intermediate_rf_pos + ft_kn_offset})
    rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_contact_over])

    # ---- Step 3: step through door with left foot
    if robot_name == 'g1':
        fixed_frames.append(['RF', 'R_knee', 'LH', 'RH'])   # frames that must not move
    elif robot_name == 'ergoCub':
        fixed_frames.append(['RF', 'R_knee', 'LH', 'RH'])  # # added RH back
    motion_frames_seq.add_motion_frame({
                        'L_knee': final_lf_pos + ft_kn_offset,
                        'LF': final_lf_pos})
    lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([lf_contact_over])

    # ---- Step 4: balance + add RH contact
    # fixed_frames.append(['LF', 'L_knee', 'RF', 'R_knee'])
    # motion_frames_seq.add_motion_frame({
    #     'RH': door_r_inner_location,
    # })
    # rh_balance = PlannerSurfaceContact('RH', np.array([1, 0, 0]))
    # motion_frames_seq.add_contact_surfaces([rh_balance])

    # ---- Step 5: RF square up
    # fixed_frames.append(['LF', 'L_knee', 'RH'])
    # motion_frames_seq.add_motion_frame({
    #     'RF': final_rf_pos,
    #     'R_knee': final_rf_pos + ft_kn_offset,
    # })
    # rf_square_up = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    # motion_frames_seq.add_contact_surfaces([rf_square_up])

    # ---- Step 6: balance + return to zero configuration
    fixed_frames.append(['LF', 'L_knee', 'RH'])
    motion_frames_seq.add_motion_frame({
        'torso': final_torso_pos,
        'RF': final_rf_pos,
        'R_knee': final_rf_pos + ft_kn_offset,
        # 'RH': final_rh_pos,
        'LH': final_lh_pos
    })
    rf_square_up = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([rf_square_up])

    # ---- Step 7: balance + return to zero configuration
    fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH'])
    motion_frames_seq.add_motion_frame({'RH': final_rh_pos})

    return fixed_frames, motion_frames_seq


def visualize_env(rob_model, rob_collision_model, rob_visual_model, q0, door_pose=None):
    # visualize robot and door
    visualizer = MeshcatVisualizer(rob_model, rob_collision_model, rob_visual_model)

    try:
        visualizer.initViewer(open=True)
        visualizer.viewer.wait()
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install Python meshcat"
        )
        print(err)
        sys.exit(0)
    visualizer.loadViewerModel(rootNodeName=rob_model.name)
    visualizer.display(q0)

    # load (real) environment to visualizer
    if door_pose is not None:
        door_model, door_collision_model, door_visual_model = load_navy_door_models()
        door_vis = MeshcatVisualizer(door_model, door_collision_model, door_visual_model)
        door_vis.initViewer(visualizer.viewer)
        door_vis.loadViewerModel(rootNodeName="door")
        door_vis_q = door_pose
        door_vis.display(door_vis_q)
    else:
        door_model, door_collision_model, door_visual_model = None, None, None

    return visualizer, door_model, door_collision_model, door_visual_model

def main(args):
    env = args.env
    contact_seq = args.sequence
    robot_name = args.robot_name
    kin_plan_path = args.kin_plan_path

    #
    # Initialize frames to consider for contact planning
    #
    plan_to_model_frames = OrderedDict()
    force_joint_frames = OrderedDict()
    if robot_name == 'g1':
        plan_to_model_frames['torso'] = 'torso_primitive_shape'
        plan_to_model_frames['LF'] = 'left_ankle_roll_link'
        plan_to_model_frames['RF'] = 'right_ankle_roll_link'
        plan_to_model_frames['L_knee'] = 'left_knee_link'
        plan_to_model_frames['R_knee'] = 'right_knee_link'
        plan_to_model_frames['LH'] = 'left_rubber_hand'
        plan_to_model_frames['RH'] = 'right_rubber_hand'
        force_joint_frames['LF'] = "left_ankle_roll_joint"
        force_joint_frames['RF'] = "right_ankle_roll_joint"
        force_joint_frames['LH'] = "left_wrist_yaw_joint"
        force_joint_frames['RH'] = "right_wrist_yaw_joint"
        package_dir = cwd + "/robot_model/g1_description"
        robot_urdf_file = package_dir + "/g1_29dof_lock_waist_modified.urdf"
    elif robot_name == 'valkyrie':
        plan_to_model_frames['torso'] = 'torso'
        plan_to_model_frames['LF'] = 'leftFoot'
        plan_to_model_frames['RF'] = 'rightFoot'
        plan_to_model_frames['L_knee'] = 'leftKneePitchLink'
        plan_to_model_frames['R_knee'] = 'rightKneePitchLink'
        plan_to_model_frames['LH'] = 'leftWristLink'
        plan_to_model_frames['RH'] = 'rightWristLink'
        force_joint_frames['LF'] = "leftAnkleRoll"
        force_joint_frames['RF'] = "rightAnkleRoll"
        force_joint_frames['LH'] = "leftWrist"
        force_joint_frames['RH'] = "rightWrist"
        package_dir = cwd + "/robot_model/" + robot_name
        robot_urdf_file = package_dir + "/valkyrie_hands.urdf"
    elif robot_name == 'ergoCub':
        plan_to_model_frames['torso'] = 'torso_primitive_shape'
        plan_to_model_frames['LF'] = 'l_ankle_2'
        plan_to_model_frames['RF'] = 'r_ankle_2'
        plan_to_model_frames['L_knee'] = 'l_lower_leg'
        plan_to_model_frames['R_knee'] = 'r_lower_leg'
        plan_to_model_frames['LH'] = 'l_hand_palm'
        plan_to_model_frames['RH'] = 'r_hand_palm'
        force_joint_frames['LF'] = "l_ankle_roll"   # "l_foot_front_ft_sensor"
        force_joint_frames['RF'] = "r_ankle_roll"     # "r_foot_front_ft_sensor"
        force_joint_frames['LH'] = "l_wrist_pitch"
        force_joint_frames['RH'] = "r_wrist_pitch"
        package_dir = cwd + "/robot_model/" + robot_name
        robot_urdf_file = package_dir + "/ergoCub.urdf"
    else:
        raise NotImplementedError('Mapping between planner and robot frames not defined')

    #
    # Load robot model, reachable regions, and environment
    #
    aux_frames_path = (cwd + '/pnc/reachability_map/output/' + robot_name + '/' +
                       robot_name + '_aux_frames.yaml')
    ee_halfspace_params = OrderedDict()
    reach_path = cwd + '/pnc/reachability_map/output/' + robot_name + '/' + robot_name
    for fr in plan_to_model_frames.keys():
        ee_halfspace_params[fr] = reach_path + '_' + fr + '.yaml'

    # load robot model and corresponding robot data
    rob_model, col_model, vis_model, rob_data, col_data, vis_data = load_robot_model(package_dir, robot_urdf_file)

    # get root to torso offset
    root_to_torso_offset = get_root_to_torso_offset(package_dir, rob_model, robot_urdf_file)

    # Getting the frame ids
    plan_to_model_ids = {}
    plan_to_model_ids['RF'] = rob_model.getFrameId(plan_to_model_frames['RF'])
    plan_to_model_ids['LF'] = rob_model.getFrameId(plan_to_model_frames['LF'])
    plan_to_model_ids['R_knee'] = rob_model.getFrameId(plan_to_model_frames['R_knee'])
    plan_to_model_ids['L_knee'] = rob_model.getFrameId(plan_to_model_frames['L_knee'])
    plan_to_model_ids['LH'] = rob_model.getFrameId(plan_to_model_frames['LH'])
    plan_to_model_ids['RH'] = rob_model.getFrameId(plan_to_model_frames['RH'])
    plan_to_model_ids['torso'] = rob_model.getFrameId(plan_to_model_frames['torso'])

    if env == 'door':
        if contact_seq == 0:
            seq_str = 'over'
        elif contact_seq == 1:
            seq_str = 'on'
        elif contact_seq == 2:
            seq_str = 'on_balanced'
        else:
            raise NotImplementedError('Contact sequence not defined')

        # load navy environment (with respective door offset) and initial robot pose
        door_pos = np.array([0.32, 0., 0.])
        step_length = 0.35
        if robot_name == 'g1':
            q0 = get_g1_default_initial_pose(rob_model.nq - 7)
            door_pos = np.array([0.32, 0., 0.])
            step_length = 0.46
            planner_params = g1_params.MultiContactDoorConfig()
        elif robot_name == 'valkyrie':
            q0 = get_val_default_initial_pose(rob_model.nq - 7)
            door_pos = np.array([0.34, 0., 0.])
            step_length = 0.55
            planner_params = valkyrie_params.MultiContactDoorConfig()
        elif robot_name == 'ergoCub':
            q0 = get_ergoCub_default_initial_pose(rob_model.nq - 7)
            door_pos = np.array([0.30, 0., 0.])
            step_length = 0.47
            planner_params = ergoCub_params.MultiContactDoorConfig()
        else:
            raise NotImplementedError('Robot default configuration not specified')
        v0 = np.zeros(rob_model.nv)
        x0 = np.concatenate([q0, v0])
        door_pose, obstacles, domain_ubody, domain_lbody_l, domain_lbody_r = load_navy_env(robot_name, door_pos)
    elif env == 'stairs':
        # create tilted stairs environment
        stairs = TiltedStairs()

        if robot_name == 'g1':
            q0 = get_g1_default_initial_pose(rob_model.nq - 7, env)
            planner_params = g1_params.MultiContactTiltedStairsConfig()
        elif robot_name == 'ergoCub':
            q0 = get_ergoCub_default_initial_pose(rob_model.nq - 7) # TODO add stairs env config
            planner_params = ergoCub_params.MultiContactTiltedStairsConfig()
        else:
            raise NotImplementedError('Robot default configuration not specified for stairs')
        v0 = np.zeros(rob_model.nv)
        x0 = np.concatenate([q0, v0])
    else:
        raise NotImplementedError('Specified environment cannot be loaded')

    if kin_plan_path is None:

        # Update Pinocchio model
        pin.forwardKinematics(rob_model, rob_data, q0)
        pin.updateFramePlacements(rob_model, rob_data)

        # Generate IRIS regions
        if env == 'door':
            standing_pos = q0[:3]
            safe_regions_mgr_dict, p_init = compute_iris_regions_mgr(obstacles, domain_ubody,
                                                                     domain_lbody_l, domain_lbody_r,
                                                                     rob_data, plan_to_model_ids,
                                                                     standing_pos, step_length,
                                                                     robot_name=robot_name,
                                                                     root_to_torso_offset=root_to_torso_offset)
        elif env == 'stairs':
            # set-up easy access to fwd kinematics for IRIS seeds
            robot_fwdk = PinocchioRobotSystem(robot_urdf_file, package_dir, False, False)
            cmd = robot_fwdk.create_cmd_ordered_dict(q0[7:], np.zeros(len(q0[7:])),
                                                          np.zeros(len(q0[7:])))
            robot_fwdk.update_system(None, None, None, None,
                                          q0[:3], q0[3:7], np.zeros(3), np.zeros(3),
                                          cmd["joint_pos"], cmd["joint_vel"])

            # hand-chosen five-stage sequence of contacts
            starting_pose = {}
            for fr in plan_to_model_frames.keys():
                starting_pose[fr] = robot_fwdk.get_link_iso(plan_to_model_frames[fr])[:3, 3]
            fixed_frames_seq, motion_frames_seq = stairs_plan.get_opposing_limbs_contact_sequence(stairs, starting_pose, robot_name, b_use_knees=B_USE_KNEES)

            # process vision and create IRIS regions
            standing_pos = q0[:3]
            safe_regions_mgr_dict = stairs_plan.compute_stairs_iris_regions_mgr(stairs, starting_pose, motion_frames_seq)
            p_init = {}
            p_init['torso'] = starting_pose['torso']
            p_init['LF'] = starting_pose['LF']
            p_init['RF'] = starting_pose['RF']
            p_init['L_knee'] = starting_pose['L_knee']
            p_init['R_knee'] = starting_pose['R_knee']
            p_init['LH'] = starting_pose['LH']
            p_init['RH'] = starting_pose['RH']
        else:
            raise NotImplementedError(f'Assign a method to compute IRIS regions for env {env}')

        if B_VISUALIZE:
            if env == 'door':
                visualizer, door_model, door_collision_model, door_visual_model \
                    = visualize_env(rob_model, col_model, vis_model, q0, door_pose)
            else:
                visualizer, _, __, ___ = visualize_env(rob_model, col_model, vis_model, q0)
        else:
            visualizer = None

        #
        # Initialize IK Frame Planner
        #
        ik_cfree_planner = IKCFreePlanner(rob_model, rob_data, plan_to_model_frames, q0, planner_params)

        # generate all frame traversable regions
        traversable_regions_dict = OrderedDict()
        for fr in plan_to_model_frames.keys():
            if fr == 'torso':
                traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                      b_visualize_reach=B_VISUALIZE,
                                                                      b_visualize_safe=B_VISUALIZE,
                                                                      visualizer=visualizer)
            else:
                traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                      ee_halfspace_params[fr],
                                                                      b_visualize_reach=B_VISUALIZE,
                                                                      b_visualize_safe=B_VISUALIZE,
                                                                      visualizer=visualizer,
                                                                      root_to_torso_pos=root_to_torso_offset)
                traversable_regions_dict[fr].update_origin_pose(standing_pos)
            traversable_regions_dict[fr].load_iris_regions(safe_regions_mgr_dict[fr])

        # hand-chosen five-stage sequence of contacts
        if robot_name == 'valkyrie':
            fixed_frames_seq, motion_frames_seq = get_two_stage_contact_sequence(safe_regions_mgr_dict)
        else:   # smaller robots have been set up with different contact sequences
            # Note: the contact sequence was defined earlier for the stairs environment
            if env == 'door':
                if contact_seq == 0:    # step through door
                    fixed_frames_seq, motion_frames_seq = get_five_stage_one_hand_contact_sequence(robot_name, safe_regions_mgr_dict)
                elif contact_seq == 1:  # step on knee-knocker RF
                    fixed_frames_seq, motion_frames_seq = get_five_stage_on_knocker_contact_sequence(robot_name, safe_regions_mgr_dict)
                elif contact_seq == 2:  # step on knee-knocker RF both hands
                    fixed_frames_seq, motion_frames_seq = get_on_knocker_balanced_contact_sequence(robot_name, safe_regions_mgr_dict)
                else:
                    NotImplementedError(f"Contact sequence {contact_seq} not implemented")

        contact_seqs = get_contact_seq_from_fixed_frames_seq(fixed_frames_seq)
        contact_seq_planes = get_contact_planes_from_motion_frames_seq(contact_seqs, motion_frames_seq)

        # planner parameters
        T = 3

        # use self-collision avoidance
        sca_geometry = None
        sca_str = '_'
        if B_USE_SELF_COLLISION_AVOIDANCE:
            sca_str = '_sca_'
            sca_geometry = SCARobotGeometry(package_dir, robot_urdf_file, plan_to_model_frames)
        traversable_regions = [traversable_regions_dict['torso'],
                               traversable_regions_dict['LF'],
                               traversable_regions_dict['RF'],
                               traversable_regions_dict['L_knee'],
                               traversable_regions_dict['R_knee'],
                               traversable_regions_dict['LH'],
                               traversable_regions_dict['RH']]
        frame_planner = LocomanipulationFramePlanner(traversable_regions,
                                                     aux_frames_path=aux_frames_path,
                                                     fixed_frames=fixed_frames_seq,
                                                     motion_frames_seq=motion_frames_seq,
                                                     sca_robot_geom=sca_geometry,
                                                     b_use_knees_in_smooth_plan=B_USE_KNEES_IN_SMOOTH_PLAN)

        # compute paths and create targets
        ik_cfree_planner.set_planner(frame_planner)
        ik_cfree_planner.set_plan_to_model_frames(plan_to_model_frames)
        ik_cfree_planner.plan(p_init, T, planner_params, visualizer, B_VERBOSE)

        if B_SAVE_KIN_DATA:
            # save the solution parameters needed to reconstruct the Bezier curves
            save_filename = robot_name + sca_str + 'step_' + seq_str + '_knee_knocker_kin.pkl'
            transition_times = []
            n_frames = len(ik_cfree_planner.planner.path)
            kin_data_saver = DataSaver(save_filename)
            kin_data_saver.add('bez_points', ik_cfree_planner.planner.points)
            for i in range(n_frames):
                transition_times.append(ik_cfree_planner.planner.path[i].transition_times)
            kin_data_saver.add('bez_points_transition_times', transition_times)
            kin_data_saver.add('n_frames', n_frames)
            kin_data_saver.add('n_iris_traversed_per_frame', len(ik_cfree_planner.planner.path[0].beziers))
            kin_data_saver.add('bez_path', ik_cfree_planner.planner.path)
            kin_data_saver.add('fixed_frames', fixed_frames_seq)
            kin_data_saver.add('contact_seq_planes', contact_seq_planes)
            kin_data_saver.advance()
            kin_data_saver.close()

    else:
        print(f' {"*" * 8} Loading solution from {kin_plan_path} {"*" * 8}')
        with open(str(kin_plan_path), 'rb') as file:
            while True:
                try:
                    d = pickle.load(file)
                    ik_cfree_planner = d['bez_path']
                    fixed_frames = d['fixed_frames']
                    contact_seq_planes = d['contact_seq_planes']
                except EOFError:
                    break
        # get parameters needed for reconstruction in crocoddyl
        contact_seqs = get_contact_seq_from_fixed_frames_seq(fixed_frames)
        # contact_seqs[-1].remove('LH')
        # contact_seqs[-1].remove('RH')
        T = ik_cfree_planner[0].beziers[0].b

        # load knee knocker visualization and collision models
        door_model, door_collision_model, door_visual_model = load_navy_door_models()

    #
    # Start Dynamic Feasibility Check
    #
    N_horizon_lst = planner_params.N_HORIZON_LST
    contact_sequence = ContactSequence(contact_seq_planes, N_horizon_lst, T)
    if robot_name == 'g1':
        robot_dyn_plan = G1MulticontactPlanner(rob_model, contact_sequence, ik_cfree_planner, planner_params)
        if env == 'door':
            if contact_seq == 1:    # step on knee knocker
                robot_dyn_plan.reset_default_gains('torso', np.array([2.5, 3.5, 1.5] + [0.5, 0.5, 0.001]))
                robot_dyn_plan.set_zero_configuration(q0)
    elif robot_name == 'ergoCub':
        robot_dyn_plan = ErgoCubMulticontactPlanner(rob_model, contact_sequence, ik_cfree_planner, planner_params)
        robot_dyn_plan.set_zero_configuration(q0)
    elif robot_name == 'valkyrie':
        robot_dyn_plan = ValkyrieMulticontactPlanner(rob_model, contact_sequence, ik_cfree_planner, planner_params)
    else:
        raise NotImplementedError(f"Matching multicontact planner for {robot_name} not found")

    # visualize kinematic plan
    if B_VISUALIZE:
        N_knots = len(robot_dyn_plan.lf_targets)
        n_contacts = len(N_horizon_lst)
        save_freq = 10
        kin_display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                                                      rob_data, vis_data, col_data,
                                                      ctrl_freq=N_knots / (n_contacts * T), save_freq=save_freq)
        if env == 'door':
            kin_display.add_robot("door", door_model, door_collision_model, door_visual_model, door_pos, door_pose[3:])
        elif env == 'stairs':
            kin_display.add_shapes_from(stairs.obstacles)
        else:
            raise NotImplementedError(f"Visualization for environment {env} not implemented")

        # start animation
        kin_display.start_animation()
        for t in np.linspace(0, n_contacts * T, N_knots // save_freq):
            if kin_plan_path is None:
                frame_targets_dict = ik_cfree_planner.pack_current_targets(t)
                kin_display.animate_single_collision(plan_to_model_frames['torso'] + '_0', frame_targets_dict['torso'])
                kin_display.animate_single_collision(plan_to_model_frames['LH'] + '_0', frame_targets_dict['LH'])
                kin_display.animate_single_collision(plan_to_model_frames['RH'] + '_0', frame_targets_dict['RH'])
                kin_display.animate_single_collision(plan_to_model_frames['L_knee'] + '_0', frame_targets_dict['L_knee'])
                kin_display.animate_single_collision(plan_to_model_frames['R_knee'] + '_0', frame_targets_dict['R_knee'])
                kin_display.animate_single_collision(plan_to_model_frames['LF'] + '_0', frame_targets_dict['LF'])
                kin_display.animate_single_collision(plan_to_model_frames['RF'] + '_0', frame_targets_dict['RF'])
                kin_display.animate_target("lfoot_target", [frame_targets_dict['LF']], [1, 1, 0])
                kin_display.animate_target("lknee_target", [frame_targets_dict['L_knee']], [0, 0, 1])
                kin_display.animate_target("rfoot_target", [frame_targets_dict['RF']], [1, 1, 0])
                kin_display.animate_target("rknee_target", [frame_targets_dict['R_knee']], [0, 0, 1])
                kin_display.animate_target("lhand_target", [frame_targets_dict['LH']], [0.5, 0, 0])
                kin_display.animate_target("rhand_target", [frame_targets_dict['RH']], [0.5, 0, 0])
                kin_display.animate_target("base_target", [frame_targets_dict['torso']], [0, 0.5, 0])
            else:
                kin_display.animate_single_collision(plan_to_model_frames['torso'] + '_0', get_frame_des_pos(ik_cfree_planner[0], t))
                kin_display.animate_single_collision(plan_to_model_frames['LH'] + '_0', get_frame_des_pos(ik_cfree_planner[5], t))
                kin_display.animate_single_collision(plan_to_model_frames['RH'] + '_0', get_frame_des_pos(ik_cfree_planner[6], t))
                kin_display.animate_target("lfoot_target", [get_frame_des_pos(ik_cfree_planner[1], t)], [1, 1, 0])
                kin_display.animate_target("lknee_target", [get_frame_des_pos(ik_cfree_planner[3], t)], [0, 0, 1])
                kin_display.animate_target("rfoot_target", [get_frame_des_pos(ik_cfree_planner[2], t)], [1, 1, 0])
                kin_display.animate_target("rknee_target", [get_frame_des_pos(ik_cfree_planner[4], t)], [0, 0, 1])
                kin_display.animate_target("lhand_target", [get_frame_des_pos(ik_cfree_planner[5], t)], [0.5, 0, 0])
                kin_display.animate_target("rhand_target", [get_frame_des_pos(ik_cfree_planner[6], t)], [0.5, 0, 0])
                kin_display.animate_target("base_target", [get_frame_des_pos(ik_cfree_planner[0], t)], [0, 0.5, 0])
            kin_display.animation_step()
        if robot_name == 'g1':
            urdf_robot_name = "g1_29dof_lock_waist"
        else:
            urdf_robot_name = robot_name
        kin_display.hide_visuals([urdf_robot_name + "/visuals"])
        kin_display.hide_visuals([urdf_robot_name + "/collisions"], True)
        kin_display.finish_animation()

    robot_dyn_plan.set_plan_to_model_params(plan_to_model_ids)
    robot_dyn_plan.set_initial_configuration(x0)
    robot_dyn_plan.plan()

    # Creating display
    if B_VISUALIZE:
        save_freq = 10
        display_idx = np.arange(0, len(robot_dyn_plan.lf_targets), save_freq)
        display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                          rob_data, vis_data, col_data, ctrl_freq=np.average(N_horizon_lst)/T, save_freq=save_freq)
        if env == 'door':
            display.add_robot("door", door_model, door_collision_model, door_visual_model, door_pos, door_pose[3:])
        elif env == 'stairs':
            display.add_shapes_from(stairs.obstacles)
        display.display_targets("lfoot_target", robot_dyn_plan.lf_targets[display_idx], [1, 1, 0])
        display.display_targets("lknee_target", robot_dyn_plan.lkn_targets[display_idx], [0, 0, 1])
        display.display_targets("rfoot_target", robot_dyn_plan.rf_targets[display_idx], [1, 1, 0])
        display.display_targets("rknee_target", robot_dyn_plan.rkn_targets[display_idx], [0, 0, 1])
        display.display_targets("lhand_target", robot_dyn_plan.lh_targets[display_idx], [0.5, 0, 0])
        display.display_targets("rhand_target", robot_dyn_plan.rh_targets[display_idx], [0.5, 0, 0])
        display.display_targets("base_target", robot_dyn_plan.base_targets[display_idx], [0, 0.5, 0])
        display.add_arrow("forces/" + force_joint_frames['LF'], color=[1, 0, 0])
        display.add_arrow("forces/" + force_joint_frames['RF'], color=[0, 0, 1])
        display.add_arrow("forces/" + force_joint_frames['LH'], color=[0, 1, 0])
        display.add_arrow("forces/" + force_joint_frames['RH'], color=[0, 1, 0])
        display.displayFromCrocoddylSolver(robot_dyn_plan.fddp)
        # viz_to_hide = list(("base_target", "lhand_target", "rhand_target",
        #                     "lfoot_target", "lknee_target",
        #                     "rfoot_target", "rknee_target"))
        display.hide_visuals(["env/1", "env/2"])
        display.hide_visuals(["g1_29dof_lock_waist/collisions"], True)
        if B_SAVE_HTML:
            display.save_html(cwd + "/data/ONR/", robot_name + sca_str + "DYN_" + seq_str + "_anim.html")

    if B_SHOW_JOINT_PLOTS or B_SHOW_COST_PLOTS or B_SHOW_JOINT_LIM_PLOTS:
        plan_plotter = MulticontactPlotter(robot_dyn_plan)
        if B_SHOW_JOINT_PLOTS:
            plan_plotter.plot_reduced_xs_us()
        if B_SHOW_COST_PLOTS:
            plan_plotter.plot_costs()
        if B_SHOW_JOINT_LIM_PLOTS:
            plan_plotter.plot_joint_limit_margins()
        plt.show()

    if B_SHOW_GRF_PLOTS or B_SAVE_DYN_DATA:
        # Note: contact_links are l_ankle_ie, r_ankle_ie, l_wrist_pitch, r_wrist_pitch
        sim_steps_list = [len(robot_dyn_plan.fddp[i].us) for i in range(len(robot_dyn_plan.fddp))]
        sim_steps = np.sum(sim_steps_list)
        sim_time = np.zeros((sim_steps,))
        rf_lfoot, rf_rfoot, rf_lwrist, rf_rwrist = np.zeros((3, sim_steps)), \
            np.zeros((3, sim_steps)), np.zeros((3, sim_steps)), np.zeros((3, sim_steps))
        time_idx = 0
        for it in robot_dyn_plan.fddp:
            rf_list = vis_tools.get_force_trajectory_from_solver(it)
            for rf_t in rf_list:
                for contact in rf_t:
                    # determine contact link
                    cur_link = int(contact['key'])
                    if rob_model.names[cur_link] == force_joint_frames['LF']:
                        rf_lfoot[:, time_idx] = contact['f'].linear
                    elif rob_model.names[cur_link] == force_joint_frames['RF']:
                        rf_rfoot[:, time_idx] = contact['f'].linear
                    elif rob_model.names[cur_link] == force_joint_frames['LH']:
                        rf_lwrist[:, time_idx] = contact['f'].linear
                    elif rob_model.names[cur_link] == force_joint_frames['RH']:
                        rf_rwrist[:, time_idx] = contact['f'].linear
                    else:
                        print(f"ERROR: Non-specified contact {rob_model.names[cur_link]}")
                dt = it.problem.runningModels[0].dt     # assumes constant dt over fddp sequence
                if time_idx < len(sim_time) - 1:
                    sim_time[time_idx+1] = sim_time[time_idx] + dt
                    time_idx += 1
                else:
                    continue

        if B_SHOW_GRF_PLOTS:
            plot_vector_traj(sim_time, rf_lfoot.T, 'RF LFoot (World)', Fxyz_labels)
            plot_vector_traj(sim_time, rf_rfoot.T, 'RF RFoot (World)', Fxyz_labels)
            plot_vector_traj(sim_time, rf_lwrist.T, 'RF LWrist (World)', Fxyz_labels)
            plot_vector_traj(sim_time, rf_rwrist.T, 'RF RWrist (World)', Fxyz_labels)
            plt.show()

    if B_SAVE_DYN_DATA:
        # Saving data tools
        dyn_data_saver = DataSaver(robot_name + sca_str + 'step_' + seq_str + '_knee_knocker.pkl')
        # save kinematic TO solution
        dyn_data_saver.add('bez_path', ik_cfree_planner.planner.path)
        dyn_data_saver.add('bez_points', ik_cfree_planner.planner.points)
        dyn_data_saver.add('n_iris_traversed_per_frame', len(ik_cfree_planner.planner.path[0].beziers))
        dyn_data_saver.add('bez_path', ik_cfree_planner.planner.path)
        dyn_data_saver.add('fixed_frames', fixed_frames_seq)
        dyn_data_saver.add('contact_seq_planes', contact_seq_planes)
        for (i, fp) in enumerate(robot_dyn_plan.fddp):
            com_lst = []
            torso_pos, lf_pos, rf_pos, lkn_pos, rkn_pos, lh_pos, rh_pos = [], [], [], [], [], [], []
            if i == len(robot_dyn_plan.fddp)-1:      # variables that need to be logged only once
                dyn_data_saver.add('grf_lfoot', rf_lfoot.tolist())
                dyn_data_saver.add('grf_rfoot', rf_rfoot.tolist())
                dyn_data_saver.add('grf_lhand', rf_lwrist.tolist())
                dyn_data_saver.add('grf_rhand', rf_rwrist.tolist())
                dyn_data_saver.add('time', sim_time.tolist())
            log = fp.getCallbacks()[0]
            q = np.array(log.xs)[:, :rob_model.nq]
            qd = np.array(log.xs)[:, rob_model.nq:]
            dyn_data_saver.add('joint_pos', q.tolist())
            dyn_data_saver.add('joint_vel', qd.tolist())
            dyn_data_saver.add('joint_torque', (np.array(log.us)[:, :]).tolist())
            for (qi, qdi) in zip(q, qd):
                com_lst.append(pin.centerOfMass(rob_model, rob_data, qi, qdi))
                pin.forwardKinematics(rob_model, rob_data, qi, qdi)
                torso_pos.append(pin.updateFramePlacement(rob_model, rob_data, plan_to_model_ids['torso']).translation.tolist())
                lf_pos.append(pin.updateFramePlacement(rob_model, rob_data, plan_to_model_ids['LF']).translation.tolist())
                rf_pos.append(pin.updateFramePlacement(rob_model, rob_data, plan_to_model_ids['RF']).translation.tolist())
                lkn_pos.append(pin.updateFramePlacement(rob_model, rob_data, plan_to_model_ids['L_knee']).translation.tolist())
                rkn_pos.append(pin.updateFramePlacement(rob_model, rob_data, plan_to_model_ids['R_knee']).translation.tolist())
                lh_pos.append(pin.updateFramePlacement(rob_model, rob_data, plan_to_model_ids['LH']).translation.tolist())
                rh_pos.append(pin.updateFramePlacement(rob_model, rob_data, plan_to_model_ids['RH']).translation.tolist())
            dyn_data_saver.add('center_of_mass', com_lst)
            dyn_data_saver.add('torso_act', torso_pos)
            dyn_data_saver.add('lf_act', lf_pos)
            dyn_data_saver.add('rf_act', rf_pos)
            dyn_data_saver.add('lkn_act', lkn_pos)
            dyn_data_saver.add('rkn_act', rkn_pos)
            dyn_data_saver.add('lh_act', lh_pos)
            dyn_data_saver.add('rh_act', rh_pos)
            dyn_data_saver.advance()
        dyn_data_saver.close()


def get_root_to_torso_offset(package_dir, rob_model, robot_urdf_file):
    geom_model = pin.buildGeomFromUrdf(rob_model,
                                       robot_urdf_file,
                                       pin.GeometryType.COLLISION)
    root_to_torso_offset = None
    for i, gm in enumerate(geom_model.geometryObjects):
        if 'torso_primitive_shape' in gm.name:
        # if 'torso_link' in gm.name:
            root_to_torso_offset = gm.placement.translation
            break
        # if we reach the end and didn't find torso offset, exit with message
        if (i == len(geom_model.geometryObjects) - 1) and root_to_torso_offset is None:
            raise ValueError("Could not find torso primitive shape in geometry model.")

    return root_to_torso_offset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='door',
                        choices=['door', 'stairs'],
                        help="Environment to load for planning")
    parser.add_argument("--sequence", type=int, default=2,
                        help="Contact sequence to solve for")
    parser.add_argument("--robot_name", type=str, default='g1',
                        choices=['g1', 'valkyrie', 'ergoCub'],
                        help="Robot name to use for planning")
    parser.add_argument("--kin_plan_path", type=str, default=None,
                        help="Path to pkl file containing mfpp paths")
    args = parser.parse_args()
    main(args)
