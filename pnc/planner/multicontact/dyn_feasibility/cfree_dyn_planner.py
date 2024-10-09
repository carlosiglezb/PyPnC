import os
import sys
import time
from collections import OrderedDict

cwd = os.getcwd()
sys.path.append(cwd)

import crocoddyl
import numpy as np

from pnc.planner.multicontact.crocoddyl_extensions.ConstraintModelRCJ import ConstraintModelRCJ
# Collision free description
from pydrake.geometry.optimization import HPolyhedron

# Kinematic feasibility
from pnc.planner.multicontact.kin_feasibility.frame_traversable_region import FrameTraversableRegion
from pnc.planner.multicontact.planner_surface_contact import PlannerSurfaceContact, MotionFrameSequencer
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

B_SHOW_JOINT_PLOTS = True
B_SHOW_GRF_PLOTS = True
B_VISUALIZE = True
B_SAVE_DATA = False
B_VERBOSE = True
B_SAVE_HTML = False


def get_draco3_shaft_wrist_default_initial_pose():
    q0 = np.zeros(27, )
    hip_yaw_angle = 5
    q0[0] = 0.  # l_hip_ie
    q0[1] = np.radians(hip_yaw_angle)  # l_hip_aa
    q0[2] = -np.pi / 4  # l_hip_fe
    q0[3] = np.pi / 4  # l_knee_fe_jp
    q0[4] = np.pi / 4  # l_knee_fe_jd
    q0[5] = -np.pi / 4  # l_ankle_fe
    q0[6] = np.radians(-hip_yaw_angle)  # l_ankle_ie
    q0[7] = 0.  # l_shoulder_fe
    q0[8] = np.pi / 6  # l_shoulder_aa
    q0[9] = 0.  # l_shoulder_ie
    q0[10] = -np.pi / 2  # l_elbow_fe
    q0[11] = -np.pi/3.  # l_wrist_ps
    q0[12] = 0.  # l_wrist_pitch
    q0[13] = 0.  # neck pitch
    q0[14] = 0.  # r_hip_ie
    q0[15] = np.radians(-hip_yaw_angle)  # r_hip_aa
    q0[16] = -np.pi / 4  # r_hip_fe
    q0[17] = np.pi / 4  # r_knee_fe_jp
    q0[18] = np.pi / 4  # r_knee_fe_jd
    q0[19] = -np.pi / 4  # r_ankle_fe
    q0[20] = np.radians(hip_yaw_angle)  # r_ankle_ie
    q0[21] = 0.  # r_shoulder_fe
    q0[22] = -np.pi / 6  # r_shoulder_aa
    q0[23] = 0.  # r_shoulder_ie
    q0[24] = -np.pi / 2  # r_elbow_fe
    q0[25] = np.pi/3.   # r_wrist_ps
    q0[26] = 0.  # r_wrist_pitch

    floating_base = np.array([0., 0., 0.741, 0., 0., 0., 1.])
    return np.concatenate((floating_base, q0))


def get_g1_default_initial_pose(n_joints):
    q0 = np.zeros(n_joints, )
    q0[0] = -np.pi / 6  # left_hip_pitch_joint
    # q0[1] = np.radians(hip_yaw_angle)  # left_hip_roll_joint
    # q0[2] = np.radians(hip_yaw_angle)  # left_hip_yaw_joint
    q0[3] = np.pi / 3  # left_knee_joint
    q0[4] = -np.pi / 6  # left_ankle_pitch_joint
    # q0[5] = np.radians(-hip_yaw_angle)  # left_ankle_roll_joint
    q0[6] = -np.pi / 6  # right_hip_pitch_joint
    # q0[7] = np.pi / 6  # right_hip_roll_joint
    # q0[8] = 0.  # right_hip_yaw_joint
    q0[9] = np.pi / 3  # right_knee_joint
    q0[10] = -np.pi / 6  # right_ankle_pitch_joint
    # q0[11] = 0.  # right_ankle_roll_joint

    floating_base = np.array([0., 0., 0.62, 0., 0., 0., 1.])
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
    # q0[22] = 0.                           # "l_index_add"
    # q0[23] = np.pi / 2                      # "l_index_prox"
    # q0[24] = np.pi/3.                     # "l_index_dist"
    # q0[25] = 0.                           # "l_middle_prox",
    # q0[26] = 0.                           # "l_middle_dist",
    # q0[27] = 0.                           # "l_pinkie_prox",
    # q0[28] = 0.                           # "l_pinkie_dist",
    # q0[29] = 0.                           # "l_ring_prox",
    # q0[30] = 0.                           # "l_ring_dist",
    # q0[31] = 0.                           # "l_thumb_add",
    # q0[32] = 0.                           # "l_thumb_prox",
    # q0[33] = 0.                           # "l_thumb_dist",
    # q0[34] = 0.                           # "neck_pitch",
    # q0[35] = 0.                           # "neck_roll",
    # q0[36] = 0.                           # "neck_yaw",
    # q0[37] = 0.                           # "camera_tilt",
    # q0[38] = -np.pi / 2                           # "r_shoulder_pitch",
    # q0[39] = 0.                           # "r_shoulder_roll",
    # q0[40] = 0.                           # "r_shoulder_yaw",
    q0[41] = np.pi / 2                           # "r_elbow",
    # q0[32] = 0.                           # "r_wrist_yaw",
    # q0[32] = 0.                           # "r_wrist_roll",
    # q0[32] = 0.                           # "r_wrist_pitch",
    # q0[32] = 0.                           # "r_index_add",
    # q0[32] = 0.                           # "r_index_prox",
    # q0[32] = 0.                           # "r_index_dist",
    # q0[32] = 0.                           # "r_middle_prox",
    # q0[32] = 0.                           # "r_middle_dist",
    # q0[32] = 0.                           # "r_pinkie_prox",
    # q0[32] = 0.                           # "r_pinkie_dist",
    # q0[32] = 0.                           # "r_thumb_add",
    # q0[32] = 0.                           # "r_thumb_prox",
    # q0[32] = 0.                           # "r_thumb_dist"

    floating_base = np.array([0., 0., 0.65, 0., 0., 0., 1.])
    return np.concatenate((floating_base, q0))


def load_orig_navy_env(door_pos):
    # create navy door environment
    door_quat = np.array([0., 0., 0.7071068, 0.7071068])
    door_width = np.array([0.03, 0., 0.])
    dom_ubody_lb = np.array([-1.6, -0.8, 0.5])
    dom_ubody_ub = np.array([1.6, 0.8, 2.1])
    dom_lbody_lb = np.array([-1.6, -0.8, -0.])
    dom_lbody_ub = np.array([1.6, 0.8, 1.2])
    floor = HPolyhedron.MakeBox(
        np.array([-2, -0.9, -0.05]) + door_pos + door_width,
        np.array([2, 0.9, -0.001]) + door_pos + door_width)
    knee_knocker_base = HPolyhedron.MakeBox(
        np.array([-0.05, -0.9, 0.0]) + door_pos + door_width,
        np.array([0.06, 0.9, 0.4]) + door_pos + door_width)
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
    domain_lbody = HPolyhedron.MakeBox(dom_lbody_lb, dom_lbody_ub)

    door_pose = np.concatenate((door_pos, door_quat))
    return door_pose, obstacles, domain_ubody, domain_lbody


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
            np.array([-0.05, -0.9, 0.0]) + door_pos + door_width,
            np.array([0.085, 0.9, 0.52]) + door_pos + door_width)
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


def load_robot_model(robot_name):
    if robot_name == 'draco3':
        package_dir = cwd + "/robot_model/draco3"
        robot_urdf_file = package_dir + "/draco3_ft_wrist_mesh_updated.urdf"
    elif robot_name == 'g1':
        package_dir = cwd + "/robot_model/g1_description"
        robot_urdf_file = package_dir + "/g1.urdf"
    elif robot_name == 'valkyrie':
        package_dir = cwd + "/robot_model/" + robot_name
        robot_urdf_file = package_dir + "/valkyrie_hands.urdf"
    elif robot_name == 'ergoCub':
        package_dir = cwd + "/robot_model/" + robot_name
        robot_urdf_file = package_dir + "/ergoCub.urdf"
    else:
        raise NotImplementedError('Robot model URDF path not specified')
    rob_model, col_model, vis_model = pin.buildModelsFromUrdf(robot_urdf_file,
                                                              package_dir,
                                                              pin.JointModelFreeFlyer())
    rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)

    return rob_model, col_model, vis_model, rob_data, col_data, vis_data


def compute_iris_regions_mgr(obstacles,
                             domain_ubody,
                             domain_lbody_l,
                             domain_lbody_r,
                             robot_data,
                             plan_to_model_ids,
                             standing_pos,
                             goal_step_length):
    # shift (feet) iris seed to get nicer IRIS region
    iris_lf_shift = np.array([0.1, 0., 0.])
    iris_rf_shift = np.array([0.1, 0., 0.])
    iris_kn_shift = np.array([0.05, 0., -0.05])
    iris_kn_end_shift = np.array([-0.15 , 0., -0.2])

    # get end effector positions via fwd kin
    starting_torso_pos = standing_pos
    final_torso_pos = starting_torso_pos + np.array([goal_step_length, 0., 0.])
    starting_lf_pos = robot_data.oMf[plan_to_model_ids['LF']].translation
    final_lf_pos = starting_lf_pos + np.array([goal_step_length, 0., 0.])
    # starting_lh_pos = robot_data.oMf[plan_to_model_ids['LH']].translation - np.array([0.01, 0., 0.])
    starting_lh_pos = robot_data.oMf[plan_to_model_ids['LH']].translation
    final_lh_pos = starting_lh_pos + np.array([goal_step_length, 0., 0.])
    starting_rf_pos = robot_data.oMf[plan_to_model_ids['RF']].translation
    final_rf_pos = starting_rf_pos + np.array([goal_step_length, 0., 0.])
    starting_rh_pos = robot_data.oMf[plan_to_model_ids['RH']].translation
    final_rh_pos = starting_rh_pos + np.array([goal_step_length, 0., 0.])
    starting_lkn_pos = robot_data.oMf[plan_to_model_ids['L_knee']].translation #+ np.array([0.02, 0., -0.05])
    final_lkn_pos = starting_lkn_pos + np.array([goal_step_length, 0., 0.])
    starting_rkn_pos = robot_data.oMf[plan_to_model_ids['R_knee']].translation
    final_rkn_pos = starting_rkn_pos + np.array([goal_step_length, 0., 0.])

    safe_torso_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_torso_pos)
    safe_torso_end_region = IrisGeomInterface(obstacles, domain_ubody, final_torso_pos)
    safe_lf_start_region = IrisGeomInterface(obstacles, domain_lbody_l, starting_lf_pos + iris_lf_shift)
    safe_lf_end_region = IrisGeomInterface(obstacles, domain_lbody_l, final_lf_pos)
    safe_lk_start_region = IrisGeomInterface(obstacles, domain_lbody_l, starting_lkn_pos + iris_kn_shift)
    safe_lk_end_region = IrisGeomInterface(obstacles, domain_lbody_l, final_lkn_pos + iris_kn_end_shift)
    safe_lh_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_lh_pos + np.array([0.1, 0., 0.]))
    safe_lh_end_region = IrisGeomInterface(obstacles, domain_ubody, final_lh_pos)
    safe_rf_start_region = IrisGeomInterface(obstacles, domain_lbody_r, starting_rf_pos + iris_rf_shift)
    safe_rf_end_region = IrisGeomInterface(obstacles, domain_lbody_r, final_rf_pos)
    safe_rk_start_region = IrisGeomInterface(obstacles, domain_lbody_r, starting_rkn_pos + iris_kn_shift)
    safe_rk_end_region = IrisGeomInterface(obstacles, domain_lbody_r, final_rkn_pos + iris_kn_end_shift)
    safe_rh_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_rh_pos + np.array([0.1, 0., 0.]))
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
    # if b_use_knees:
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
    motion_frames_seq.add_contact_surface(lf_contact_over)

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
    motion_frames_seq.add_contact_surface(rf_contact_over)

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
        door_l_inner_location = np.array([0.3, 0.35, 0.9])
        door_r_inner_location = np.array([0.34, -0.35, 0.9])
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
                                            'torso': starting_torso_pos + np.array([0.07, -0.07, 0.02])
                                            })
    elif robot_name == 'ergoCub':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            'torso': starting_torso_pos + np.array([0.05, -0.07, 0])
                                            })
    lh_contact_front = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    lh_contact_front.set_contact_breaking_velocity(np.array([0, -1, 0.]))
    motion_frames_seq.add_contact_surface(lh_contact_front)

    # ---- Step 2: step through door with left foot
    fixed_frames.append(['RF', 'R_knee', 'LH'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        'LF': final_lf_pos,
                        'L_knee': final_lf_pos + np.array([0.15, 0., 0.28])})
                        # 'L_knee': final_lkn_pos + np.array([-0.05, 0., 0.07])})
    lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surface(lf_contact_over)

    # ---- Step 3: re-position L/R hands for more stability
    fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        # 'LH': starting_lh_pos + np.array([0.3, 0., 0.0]),   # <-- G1
                        # 'LH': starting_lh_pos + np.array([0.35, 0.1, 0.0]),   # <-- other
                        'torso': final_torso_pos + np.array([-0.15, 0.05, 0.05]),     # good testing
                        'RH': door_r_inner_location})
    rh_contact_inside = PlannerSurfaceContact('RH', np.array([1, 0, 0]))
    motion_frames_seq.add_contact_surface(rh_contact_inside)

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
    motion_frames_seq.add_contact_surface(rf_contact_over)

    # ---- Step 5: balance / square up
    # fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
    fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee'])
    motion_frames_seq.add_motion_frame({
        # 'torso': final_torso_pos,
        'RH': final_rh_pos, # + np.array([-0.20, 0., 0.]),
        'LH': final_lh_pos
    })

    return fixed_frames, motion_frames_seq


def get_five_stage_on_knocker_contact_sequence(robot_name, safe_regions_mgr_dict):
    ###### Previously used key locations
    # door_l_outer_location = np.array([0.45, 0.35, 1.2])
    # door_r_outer_location = np.array([0.45, -0.35, 1.2])
    if robot_name == 'g1':
        # G1 settings
        door_l_inner_location = np.array([0.3, 0.35, 1.0])
        door_r_inner_location = np.array([0.34, -0.35, 1.0])
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
    intermediate_rf_pos = np.array([0.3, final_rf_pos[1], 0.44])

    # initialize fixed and motion frame sets
    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Step 1: L hand to frame
    fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    if robot_name == 'g1':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            'RH': door_r_inner_location,
                                            # 'torso': starting_torso_pos + np.array([0.07, -0.07, 0])
                                            })
    elif robot_name == 'ergoCub':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            'torso': starting_torso_pos + np.array([0.05, -0.07, 0])
                                            })
    lh_contact_front = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    lh_contact_front.set_contact_breaking_velocity(np.array([0, -1, 0.]))
    motion_frames_seq.add_contact_surface(lh_contact_front)

    # ---- Step 2: step on knee-knocker with right foot
    fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        'RF': intermediate_rf_pos,
                        'R_knee': intermediate_rf_pos + np.array([0.15, 0., 0.28])})    # + np.array([-0.05, 0., 0.035])
    rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surface(rf_contact_over)

    # ---- Step 3: step through door with right foot
    fixed_frames.append(['RF', 'R_knee', 'LH', 'RH'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        # 'LH': starting_lh_pos + np.array([0.3, 0., 0.0]),   # <-- G1
                        # 'LH': starting_lh_pos + np.array([0.35, 0.1, 0.0]),   # <-- other
                        # 'torso': final_torso_pos + np.array([-0.15, 0.05, -0.05]),     # good testing
                        # 'L_knee': final_lkn_pos + np.array([-0.05, 0., 0.035]),
                        'L_knee': final_lf_pos + np.array([0.15, 0., 0.28]),
                        'LF': final_lf_pos})
    lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surface(lf_contact_over)

    # ---- Step 4: balance / square up
    # fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
    fixed_frames.append(['LF', 'L_knee'])
    motion_frames_seq.add_motion_frame({
        'torso': final_torso_pos,
        'RF': final_rf_pos,
        # 'R_knee': final_rkn_pos + np.array([-0.05, 0., 0.035]),
        'R_knee': final_rf_pos + np.array([0.15, 0., 0.28]),
        'RH': final_rh_pos, # + np.array([-0.20, 0., 0.]),
        'LH': final_lh_pos
    })
    rf_square_up = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surface(rf_square_up)

    # ---- Step 5: balance
    fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
    motion_frames_seq.add_motion_frame({})

    return fixed_frames, motion_frames_seq


def visualize_env(rob_model, rob_collision_model, rob_visual_model, q0, door_pose):
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

    # load (real) door to visualizer
    door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
        cwd + "/robot_model/ground/navy_door.urdf",
        cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

    door_vis = MeshcatVisualizer(door_model, door_collision_model, door_visual_model)
    door_vis.initViewer(visualizer.viewer)
    door_vis.loadViewerModel(rootNodeName="door")
    door_vis_q = door_pose
    door_vis.display(door_vis_q)

    return visualizer, door_model, door_collision_model, door_visual_model


def get_contact_seq_from_fixed_frames_seq(fixed_frames_seq):
    contact_frames = ['LF', 'RF', 'LH', 'RH']
    contact_frames_seq = []
    for seq in fixed_frames_seq:
        cf = []
        for ff in seq:
            # check for frames that can be in contact:
            if ff in contact_frames:
               cf.append(ff)

        # throw error if no contact frames were detected
        if len(cf) == 0:
            print(f"No contact frames found in fixed frames")
            return

        # append contact frames found in current sequence
        contact_frames_seq.append(cf)

    # move contact-making to the end
    for cs_i, cs in enumerate(contact_frames_seq):
        if cs_i == 0:   # ignore first contact sequence
            continue

        b_phase_done = False
        # find new contact to place them at the end for impulse model
        for ccon in cs:
            if (not b_phase_done) and (ccon not in contact_frames_seq[cs_i - 1]):
                contact_frames_seq[cs_i].remove(ccon)
                contact_frames_seq[cs_i].append(ccon)
                b_phase_done = True

    # remove hands from last sequence when added for smoothing
    if contact_frames_seq[-1] == contact_frames:
        contact_frames_seq.pop(-1)
        contact_frames_seq.pop(-1)

    return contact_frames_seq


def main(args):
    contact_seq = args.sequence
    robot_name = args.robot_name

    if B_SAVE_DATA:
        # Saving data tools
        data_saver = DataSaver(robot_name + '_knee_knocker.pkl')

    #
    # Initialize frames to consider for contact planning
    #
    plan_to_model_frames = OrderedDict()
    force_joint_frames = OrderedDict()
    if robot_name == 'draco3':
        plan_to_model_frames['torso'] = 'torso_link'
        plan_to_model_frames['LF'] = 'l_foot_contact'
        plan_to_model_frames['RF'] = 'r_foot_contact'
        plan_to_model_frames['L_knee'] = 'l_knee_fe_ld'
        plan_to_model_frames['R_knee'] = 'r_knee_fe_ld'
        plan_to_model_frames['LH'] = 'l_hand_contact'
        plan_to_model_frames['RH'] = 'r_hand_contact'
    elif robot_name == 'g1':
        plan_to_model_frames['torso'] = 'torso_link'
        plan_to_model_frames['LF'] = 'left_ankle_roll_link'
        plan_to_model_frames['RF'] = 'right_ankle_roll_link'
        plan_to_model_frames['L_knee'] = 'left_knee_link'
        plan_to_model_frames['R_knee'] = 'right_knee_link'
        plan_to_model_frames['LH'] = 'left_palm_link'
        plan_to_model_frames['RH'] = 'right_palm_link'
        force_joint_frames['LF'] = "left_ankle_roll_joint"
        force_joint_frames['RF'] = "right_ankle_roll_joint"
        force_joint_frames['LH'] = "left_elbow_roll_joint"
        force_joint_frames['RH'] = "right_elbow_roll_joint"
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
    elif robot_name == 'ergoCub':
        plan_to_model_frames['torso'] = 'root_link'
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
    rob_model, col_model, vis_model, rob_data, col_data, vis_data = load_robot_model(robot_name)

    # load navy environment (with respective door offset) and initial robot pose
    door_pos = np.array([0.32, 0., 0.])
    step_length = 0.35
    if robot_name == 'draco3':
        q0 = get_draco3_shaft_wrist_default_initial_pose()
    elif robot_name == 'g1':
        q0 = get_g1_default_initial_pose(rob_model.nq - 7)
        door_pos = np.array([0.28, 0., 0.])
        step_length = 0.42
        # weights_rigid_link = np.array([10., 0., 3.])    # step over door in single step
        weights_rigid_link = np.array([10., 0., 0.])    # step on knee knocker
    elif robot_name == 'valkyrie':
        q0 = get_val_default_initial_pose(rob_model.nq - 7)
        door_pos = np.array([0.34, 0., 0.])
        step_length = 0.55
        weights_rigid_link = np.array([500., 0., 50.])
    elif robot_name == 'ergoCub':
        q0 = get_ergoCub_default_initial_pose(rob_model.nq - 7)
        door_pos = np.array([0.30, 0., 0.])
        step_length = 0.47
        weights_rigid_link = np.array([6500., 0., 1500.])
    else:
        raise NotImplementedError('Robot default configuration not specified')
    door_pose, obstacles, domain_ubody, domain_lbody_l, domain_lbody_r = load_navy_env(robot_name, door_pos)
    v0 = np.zeros(rob_model.nv)
    x0 = np.concatenate([q0, v0])

    # Update Pinocchio model
    pin.forwardKinematics(rob_model, rob_data, q0)
    pin.updateFramePlacements(rob_model, rob_data)

    # Getting the frame ids
    plan_to_model_ids = {}
    plan_to_model_ids['RF'] = rob_model.getFrameId(plan_to_model_frames['RF'])
    plan_to_model_ids['LF'] = rob_model.getFrameId(plan_to_model_frames['LF'])
    plan_to_model_ids['R_knee'] = rob_model.getFrameId(plan_to_model_frames['R_knee'])
    plan_to_model_ids['L_knee'] = rob_model.getFrameId(plan_to_model_frames['L_knee'])
    plan_to_model_ids['LH'] = rob_model.getFrameId(plan_to_model_frames['LH'])
    plan_to_model_ids['RH'] = rob_model.getFrameId(plan_to_model_frames['RH'])
    plan_to_model_ids['torso'] = rob_model.getFrameId(plan_to_model_frames['torso'])

    # Generate IRIS regions
    standing_pos = q0[:3]
    safe_regions_mgr_dict, p_init = compute_iris_regions_mgr(obstacles, domain_ubody,
                                                             domain_lbody_l, domain_lbody_r,
                                                             rob_data, plan_to_model_ids,
                                                             standing_pos, step_length)

    if B_VISUALIZE:
        visualizer, door_model, door_collision_model, door_visual_model \
            = visualize_env(rob_model, col_model, vis_model, q0, door_pose)
    else:
        visualizer = None

    #
    # Initialize IK Frame Planner
    #
    if robot_name == 'valkyrie':
        w_rigid_poly = np.array([0.1621, 0.0, 0.])
    else:
        w_rigid_poly = None
    ik_cfree_planner = IKCFreePlanner(rob_model, rob_data, plan_to_model_frames, q0, w_rigid_poly=w_rigid_poly)

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
                                                                  visualizer=visualizer)
            traversable_regions_dict[fr].update_origin_pose(standing_pos)
        traversable_regions_dict[fr].load_iris_regions(safe_regions_mgr_dict[fr])

    # hand-chosen five-stage sequence of contacts
    if robot_name == 'valkyrie':
        fixed_frames_seq, motion_frames_seq = get_two_stage_contact_sequence(safe_regions_mgr_dict)
    else:   # smaller robots have been set up with different contact sequences
        if contact_seq == 0:    # step through door
            fixed_frames_seq, motion_frames_seq = get_five_stage_one_hand_contact_sequence(robot_name, safe_regions_mgr_dict)
        elif contact_seq == 1:  # step on knee-knocker
            fixed_frames_seq, motion_frames_seq = get_five_stage_on_knocker_contact_sequence(robot_name, safe_regions_mgr_dict)
        else:
                NotImplementedError(f"Contact sequence {contact_seq} not implemented")
    contact_seqs = get_contact_seq_from_fixed_frames_seq(fixed_frames_seq)


    # planner parameters
    T = 3
    alpha = [0, 0, 1]
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
                                                 motion_frames_seq=motion_frames_seq)

    # compute paths and create targets
    ik_cfree_planner.set_planner(frame_planner)
    ik_cfree_planner.plan(p_init, T, alpha, weights_rigid_link, visualizer, B_VERBOSE)

    #
    # Start Dynamic Feasibility Check
    #
    state = crocoddyl.StateMultibody(rob_model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)

    ee_rpy = {'LH': [0., 0., 0.], 'RH': [0., 0., 0.]}
    if robot_name == 'draco3':
        n_q = len(q0)
        l_constr_ids, r_constr_ids = [9 + n_q, 10 + n_q], [23 + n_q, 24 + n_q]  # qdot
        l_constr_ids_u, r_constr_ids_u = [3, 4], [17, 18]  # u

        constr_mgr = crocoddyl.ConstraintModelManager(state, actuation.nu)
        # -------- Existent constraint --------
        # res_model = crocoddyl.ResidualModelState(state, x0, actuation.nu)
        # constr_model_res = crocoddyl.ConstraintModelResidual(state, res_model)
        # constr_mgr.addConstraint("residual_model", constr_model_res)
        # -------- New constraint --------
        l_res_model = ResidualModelStateError(state, 1, nu=actuation.nu, q_dependent=False)
        l_res_model.constr_ids = l_constr_ids
        # l_res_model.constr_ids_u = l_constr_ids_u
        l_rcj_constr = ConstraintModelRCJ(state, residual=l_res_model, ng=0, nh=1)
        constr_mgr.addConstraint("l_rcj_constr", l_rcj_constr)
        r_res_model = ResidualModelStateError(state, 1, nu=actuation.nu, q_dependent=False)
        r_res_model.constr_ids = r_constr_ids
        # r_res_model.constr_ids_u = r_constr_ids_u
        r_rcj_constr = ConstraintModelRCJ(state, residual=r_res_model, ng=0, nh=1)
        constr_mgr.addConstraint("r_rcj_constr", r_rcj_constr)
        ee_rpy = {'LH': [0., -np.pi/2, 0.], 'RH': [0., -np.pi/2, 0.]}

    #
    # Dynamic solve
    #
    if robot_name == 'g1':
        N_horizon_lst = [180, 200, 200, 150, 200]
        contact_seqs = ContactSequence(contact_seqs, N_horizon_lst, T)
        robot_dyn_plan = G1MulticontactPlanner(rob_model, contact_seqs, T, ik_cfree_planner)
    elif robot_name == 'ergoCub':
        N_horizon_lst = [100, 220, 100, 180, 80]
        contact_seqs = ContactSequence(contact_seqs, N_horizon_lst, T)
        robot_dyn_plan = ErgoCubMulticontactPlanner(rob_model, contact_seqs, T, ik_cfree_planner)
    elif robot_name == 'valkyrie':
        N_horizon_lst = [150, 150, 100]
        contact_seqs = ContactSequence(contact_seqs, N_horizon_lst, T)
        robot_dyn_plan = ValkyrieMulticontactPlanner(rob_model, contact_seqs, T, ik_cfree_planner)
    else:
        raise NotImplementedError(f"Matching multicontact planner for {robot_name} not found")

    robot_dyn_plan.set_plan_to_model_params(plan_to_model_frames, plan_to_model_ids)
    robot_dyn_plan.set_initial_configuration(x0)
    robot_dyn_plan.plan()

    # Creating display
    if B_VISUALIZE:
        save_freq = 10
        display_idx = np.arange(0, len(robot_dyn_plan.lf_targets), save_freq)
        display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                          rob_data, vis_data, ctrl_freq=np.average(N_horizon_lst)/T, save_freq=save_freq)
        display.add_robot("door", door_model, door_collision_model, door_visual_model, door_pos, door_pose[3:])
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
        viz_to_hide = list(("base_target", "lhand_target", "rhand_target",
                            "lfoot_target", "lknee_target",
                            "rfoot_target", "rknee_target"))
        display.hide_visuals(viz_to_hide)
        if B_SAVE_HTML:
            display.save_html(cwd + "/data/", robot_name + "_door_crossing.html")

    if B_SHOW_JOINT_PLOTS:
        plan_plotter = MulticontactPlotter(robot_dyn_plan)
        plan_plotter.plot_reduced_xs_us()

    if B_SHOW_GRF_PLOTS:
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

        plot_vector_traj(sim_time, rf_lfoot.T, 'RF LFoot (World)', Fxyz_labels)
        plot_vector_traj(sim_time, rf_rfoot.T, 'RF RFoot (World)', Fxyz_labels)
        plot_vector_traj(sim_time, rf_lwrist.T, 'RF LWrist (World)', Fxyz_labels)
        plot_vector_traj(sim_time, rf_rwrist.T, 'RF RWrist (World)', Fxyz_labels)
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=0,
                        help="Contact sequence to solve for")
    parser.add_argument("--robot_name", type=str, default='g1',
                        help="Robot name to use for planning")
    args = parser.parse_args()
    main(args)
