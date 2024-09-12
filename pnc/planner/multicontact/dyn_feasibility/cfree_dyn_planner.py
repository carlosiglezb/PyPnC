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

# Visualization tools
import matplotlib.pyplot as plt
from plot.helper import plot_vector_traj, Fxyz_labels
import plot.meshcat_utils as vis_tools
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
            np.array([0.12, 0.9, 0.45]) + door_pos + door_width)
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
    iris_kn_shift = np.array([0.01, 0., -0.05])

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
    safe_lk_end_region = IrisGeomInterface(obstacles, domain_lbody_l, final_lkn_pos)
    safe_lh_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_lh_pos + np.array([0.1, 0., 0.]))
    safe_lh_end_region = IrisGeomInterface(obstacles, domain_ubody, final_lh_pos)
    safe_rf_start_region = IrisGeomInterface(obstacles, domain_lbody_r, starting_rf_pos + iris_rf_shift)
    safe_rf_end_region = IrisGeomInterface(obstacles, domain_lbody_r, final_rf_pos)
    safe_rk_start_region = IrisGeomInterface(obstacles, domain_lbody_r, starting_rkn_pos)
    safe_rk_end_region = IrisGeomInterface(obstacles, domain_lbody_r, final_rkn_pos)
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
                                            'torso': starting_torso_pos + np.array([0.07, -0.07, 0])
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
                        'L_knee': final_lkn_pos + np.array([-0.05, 0., 0.07])})
    lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surface(lf_contact_over)

    # ---- Step 3: re-position L/R hands for more stability
    fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    motion_frames_seq.add_motion_frame({
                        # 'LH': starting_lh_pos + np.array([0.3, 0., 0.0]),   # <-- G1
                        # 'LH': starting_lh_pos + np.array([0.35, 0.1, 0.0]),   # <-- other
                        'torso': final_torso_pos + np.array([-0.15, 0.05, -0.05]),     # good testing
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
                        'R_knee': final_rkn_pos + np.array([-0.05, 0., 0.07]),
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


def get_g1_lleg_joint_ids(robot_model):
    lleg_j_ids = []
    for side in ['left_', 'right_']:
        lleg_j_ids.append(list(robot_model.names).index(side + 'hip_roll_joint') - 2 + 7)
        lleg_j_ids.append(list(robot_model.names).index(side + 'hip_pitch_joint') - 2 + 7)
        lleg_j_ids.append(list(robot_model.names).index(side + 'hip_yaw_joint') - 2 + 7)
        lleg_j_ids.append(list(robot_model.names).index(side + 'knee_joint') - 2 + 7)
        lleg_j_ids.append(list(robot_model.names).index(side + 'ankle_roll_joint') - 2 + 7)
        lleg_j_ids.append(list(robot_model.names).index(side + 'ankle_pitch_joint') - 2 + 7)
    return lleg_j_ids

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
    elif robot_name == 'valkyrie':
        plan_to_model_frames['torso'] = 'torso'
        plan_to_model_frames['LF'] = 'leftFoot'
        plan_to_model_frames['RF'] = 'rightFoot'
        plan_to_model_frames['L_knee'] = 'leftKneePitchLink'
        plan_to_model_frames['R_knee'] = 'rightKneePitchLink'
        plan_to_model_frames['LH'] = 'leftWristLink'
        plan_to_model_frames['RH'] = 'rightWristLink'
    elif robot_name == 'ergoCub':
        plan_to_model_frames['torso'] = 'root_link'
        plan_to_model_frames['LF'] = 'l_ankle_2'
        plan_to_model_frames['RF'] = 'r_ankle_2'
        plan_to_model_frames['L_knee'] = 'l_lower_leg'
        plan_to_model_frames['R_knee'] = 'r_lower_leg'
        plan_to_model_frames['LH'] = 'l_hand_palm'
        plan_to_model_frames['RH'] = 'r_hand_palm'
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
        weights_rigid_link = np.array([3500., 0., 0.])
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
    else:
        fixed_frames_seq, motion_frames_seq = get_five_stage_one_hand_contact_sequence(robot_name, safe_regions_mgr_dict)

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
    dyn_solve_time = 0

    # Connecting the sequences of models
    NUM_OF_CONTACT_CONFIGURATIONS = 2 #len(motion_frames_seq.motion_frame_lst)

    if robot_name == 'g1':
        N_horizon_lst = [100, 150] #, 100, 150, 100]
        robot_dyn_plan = G1MulticontactPlanner(rob_model, N_horizon_lst, T, ik_cfree_planner)
        robot_dyn_plan.set_plan_to_model_params(plan_to_model_frames, plan_to_model_ids)
        robot_dyn_plan.set_initial_configuration(x0)
        robot_dyn_plan.plan()

    # lh_targets, rh_targets, lf_targets, rf_targets, lkn_targets, rkn_targets = [], [], [], [], [], []
    # base_targets = []
    #
    # # Defining the problem and the solver
    # fddp = [crocoddyl.SolverFDDP] * NUM_OF_CONTACT_CONFIGURATIONS
    # for i in range(NUM_OF_CONTACT_CONFIGURATIONS):
    #     model_seqs = []
    #     if robot_name == 'valkyrie':
    #         N_horizon_lst = [150, 150, 100]
    #         b_terminal_step = False
    #         gains = {
    #             'torso': np.array([3.0] * 3 + [0.5, 0.5, 0.01]),    # (lin, ang)
    #             'feet': np.array([6.] * 3 + [0.00001] * 3),         # (lin, ang)
    #             'L_knee': np.array([2.] * 3 + [0.00001] * 3),
    #             'R_knee': np.array([2.] * 3 + [0.00001] * 3),
    #             'hands': np.array([2.] * 3 + [0.00001] * 3)
    #         }
    #         if i == 0:
    #             # Cross door with left foot
    #             N_rf_support = N_horizon_lst[i]
    #             for t in np.linspace(i * T, (i + 1) * T, N_rf_support):
    #                 if t == (i+1)*T:
    #                     b_terminal_step = True
    #                     gains['feet'] = np.array([10.] * 3 + [1.5] * 3)
    #                 frame_targets_dict = pack_current_targets(ik_cfree_planner, plan_to_model_frames, t)
    #                 dmodel = createMultiFrameActionModel(state,
    #                                                      actuation,
    #                                                      x0,
    #                                                      plan_to_model_ids,
    #                                                      ['RF'],
    #                                                      ee_rpy,
    #                                                      frame_targets_dict,
    #                                                      None,
    #                                                      gains=gains,
    #                                                      terminal_step=b_terminal_step)
    #                 lh_targets.append(frame_targets_dict['LH'])
    #                 rh_targets.append(frame_targets_dict['RH'])
    #                 base_targets.append(frame_targets_dict['torso'])
    #                 lf_targets.append(frame_targets_dict['LF'])
    #                 lkn_targets.append(frame_targets_dict['L_knee'])
    #                 model_seqs += createSequence([dmodel], T / (N_rf_support - 1), 1)
    #
    #         elif i == 1:
    #             # Cross door with left foot
    #             N_lf_support = N_horizon_lst[i]  # knots for left foot crossing
    #             for t in np.linspace(i * T, (i + 1) * T, N_lf_support):
    #                 if t == (i+1)*T:
    #                     b_terminal_step = True
    #                     gains['feet'] = np.array([10.] * 3 + [1.5] * 3)
    #                 frame_targets_dict = pack_current_targets(ik_cfree_planner, plan_to_model_frames, t)
    #                 dmodel = createMultiFrameActionModel(state,
    #                                                      actuation,
    #                                                      x0,
    #                                                      plan_to_model_ids,
    #                                                      ['LF'],
    #                                                      ee_rpy,
    #                                                      frame_targets_dict,
    #                                                      None,
    #                                                      gains=gains,
    #                                                      terminal_step=b_terminal_step)
    #                 lh_targets.append(frame_targets_dict['LH'])
    #                 rh_targets.append(frame_targets_dict['RH'])
    #                 base_targets.append(frame_targets_dict['torso'])
    #                 rf_targets.append(frame_targets_dict['RF'])
    #                 rkn_targets.append(frame_targets_dict['R_knee'])
    #                 model_seqs += createSequence([dmodel], T / (N_lf_support - 1), 1)
    #
    #         elif i == 2:
    #             # Croos door with right feet
    #             N_square_up = N_horizon_lst[i]  # knots for right foot crossing
    #             for t in np.linspace(i*T, (i+1)*T, N_square_up):
    #                 if t == (i+1)*T:
    #                     b_terminal_step = True
    #                     gains['feet'] = np.array([10.] * 3 + [1.5] * 3)
    #                 frame_targets_dict = pack_current_targets(ik_cfree_planner, plan_to_model_frames, t)
    #                 dmodel = createMultiFrameActionModel(state,
    #                                                      actuation,
    #                                                      x0,
    #                                                      plan_to_model_ids,
    #                                                      ['LF', 'RF'],
    #                                                      ee_rpy,
    #                                                      frame_targets_dict,
    #                                                      None,
    #                                                      gains=gains,
    #                                                      terminal_step=b_terminal_step)
    #                 lh_targets.append(frame_targets_dict['LH'])
    #                 rh_targets.append(frame_targets_dict['RH'])
    #                 base_targets.append(frame_targets_dict['torso'])
    #                 model_seqs += createSequence([dmodel], T/(N_square_up-1), 1)
    #     else:
    #         if robot_name == 'g1':
    #             gains = {
    #                 'torso': np.array([1.0] * 3 + [0.5] * 3),  # (lin, ang)
    #                 'feet': np.array([8.] * 3 + [0.00001] * 3),  # (lin, ang)
    #                 'L_knee': np.array([3.] * 3 + [0.00001] * 3),
    #                 'R_knee': np.array([3.] * 3 + [0.00001] * 3),
    #                 'hands': np.array([2.] * 3 + [0.00001] * 3)
    #             }
    #         elif robot_name == 'ergoCub':
    #             N_horizon_lst = [80, 220, 100, 180, 80]
    #             gains = {
    #                 'torso': np.array([1.0, 5., 0.5] + [0.8] * 3),  # (lin, ang)
    #                 'feet': np.array([8.] * 3 + [0.00001] * 3),  # (lin, ang)
    #                 'L_knee': np.array([4.] * 3 + [0.00001] * 3),
    #                 'R_knee': np.array([4.] * 3 + [0.00001] * 3),
    #                 'hands': np.array([2.] * 3 + [0.00001] * 3)
    #             }
    #         else:
    #             raise NotImplementedError('Horizon list and gains not defined for robot')
    #         b_terminal_step = False
    #         if i == 0:
    #             # Reach door with left hand
    #             N_lhand_to_door = N_horizon_lst[i]  # knots for left hand reaching
    #             DT = T / (N_lhand_to_door - 1)
    #             for t in np.linspace(i*T, (i+1)*T, N_lhand_to_door):
    #                 if t == (i+1)*T:
    #                     b_terminal_step = True
    #                     if robot_name == 'g1':
    #                         gains['feet'] = np.array([10.] * 3 + [1.5] * 3)
    #                     elif robot_name == 'ergoCub':
    #                         gains['feet'] = np.array([10.] * 3 + [1.0] * 3)
    #                 frame_targets_dict = pack_current_targets(ik_cfree_planner, plan_to_model_frames, t)
    #                 if t < (i+1)*T:
    #                     dmodel = createMultiFrameActionModel(state,
    #                                                          actuation,
    #                                                          x0,
    #                                                          plan_to_model_ids,
    #                                                          ['LF', 'RF'],
    #                                                          ee_rpy,
    #                                                          frame_targets_dict,
    #                                                          None,
    #                                                          gains=gains,
    #                                                          terminal_step=b_terminal_step)
    #                     model_seqs += createSequence([dmodel], DT, 1)
    #                 else:
    #                     dmodel = createMultiFrameFinalActionModel(state,
    #                                                          actuation,
    #                                                          x0,
    #                                                          plan_to_model_ids,
    #                                                          ['LF', 'RF'],
    #                                                          ee_rpy,
    #                                                          frame_targets_dict,
    #                                                          None,
    #                                                          gains=gains,
    #                                                          terminal_step=b_terminal_step)
    #                     model_seqs += createFinalSequence([dmodel])
    #                     print(f"Applying Impulse model at {i}")
    #
    #                 lh_targets.append(frame_targets_dict['LH'])
    #                 rh_targets.append(frame_targets_dict['RH'])
    #                 base_targets.append(frame_targets_dict['torso'])
    #
    #         elif i == 1:
    #             ee_rpy['LH'] = [np.pi/2, 0., 0.]
    #             # Using left-hand support, pass left-leg through door
    #             N_base_through_door = N_horizon_lst[i]  # knots per waypoint to pass through door
    #             DT = T / (N_base_through_door - 1)
    #             for t in np.linspace(i*T, (i+1)*T, N_base_through_door):
    #                 if t == (i+1)*T:
    #                     b_terminal_step = True
    #                     if robot_name == 'g1':
    #                         gains['feet'] = np.array([10.] * 3 + [1.5] * 3)
    #                     elif robot_name == 'ergoCub':
    #                         gains['feet'] = np.array([10.] * 3 + [1.0] * 3)
    #                 frame_targets_dict = pack_current_targets(ik_cfree_planner, plan_to_model_frames, t)
    #                 if t < (i + 1) * T:
    #                     dmodel = createMultiFrameActionModel(state,
    #                                                          actuation,
    #                                                          x0,
    #                                                          plan_to_model_ids,
    #                                                          ['LH', 'RF'],
    #                                                          ee_rpy,
    #                                                          frame_targets_dict,
    #                                                          None,
    #                                                          gains=gains,
    #                                                          terminal_step=b_terminal_step)
    #                     model_seqs += createSequence([dmodel], DT, 1)
    #                 else:
    #                     dmodel = createMultiFrameFinalActionModel(state,
    #                                                               actuation,
    #                                                               x0,
    #                                                               plan_to_model_ids,
    #                                                               ['LF', 'RF'],
    #                                                               ee_rpy,
    #                                                               frame_targets_dict,
    #                                                               None,
    #                                                               gains=gains,
    #                                                               terminal_step=b_terminal_step)
    #                     model_seqs += createFinalSequence([dmodel])
    #                     print(f"Applying Impulse model at {i}")
    #
    #                 lh_targets.append(frame_targets_dict['LH'])
    #                 rh_targets.append(frame_targets_dict['RH'])
    #                 base_targets.append(frame_targets_dict['torso'])
    #                 lf_targets.append(frame_targets_dict['LF'])
    #                 lkn_targets.append(frame_targets_dict['L_knee'])
    #
    #         elif i == 2:
    #             # Reach door with left and right hand from inside
    #             N_rhand_to_door = N_horizon_lst[i]  # knots for left hand reaching
    #             for t in np.linspace(i*T, (i+1)*T, N_rhand_to_door):
    #                 if t == (i+1)*T:
    #                     b_terminal_step = True
    #                     if robot_name == 'g1':
    #                         gains['feet'] = np.array([10.] * 3 + [1.5] * 3)
    #                     elif robot_name == 'ergoCub':
    #                         gains['feet'] = np.array([10.] * 3 + [1.0] * 3)
    #                 frame_targets_dict = pack_current_targets(ik_cfree_planner, plan_to_model_frames, t)
    #                 if t < (i + 1) * T:
    #                     dmodel = createMultiFrameActionModel(state,
    #                                                          actuation,
    #                                                          x0,
    #                                                          plan_to_model_ids,
    #                                                          ['LF', 'RF'],
    #                                                          ee_rpy,
    #                                                          frame_targets_dict,
    #                                                          None,
    #                                                          gains=gains,
    #                                                          terminal_step=b_terminal_step)
    #                     model_seqs += createSequence([dmodel], T/(N_rhand_to_door), 1)
    #                 else:
    #                     dmodel = createMultiFrameFinalActionModel(state,
    #                                                               actuation,
    #                                                               x0,
    #                                                               plan_to_model_ids,
    #                                                               ['LF', 'RF'],
    #                                                               ee_rpy,
    #                                                               frame_targets_dict,
    #                                                               None,
    #                                                               gains=gains,
    #                                                               terminal_step=b_terminal_step)
    #                     model_seqs += createFinalSequence([dmodel])
    #                 lh_targets.append(frame_targets_dict['LH'])
    #                 rh_targets.append(frame_targets_dict['RH'])
    #                 base_targets.append(frame_targets_dict['torso'])
    #
    #         elif i == 3:
    #             # Using left-hand and right-foot supports, pass right-leg through door
    #             N_base_square_up = N_horizon_lst[i]  # knots per waypoint to pass through door
    #             for t in np.linspace(i*T, (i+1)*T, N_base_square_up):
    #                 if t == (i+1)*T:
    #                     b_terminal_step = True
    #                     if robot_name == 'g1':
    #                         gains['feet'] = np.array([10.] * 3 + [1.5] * 3)
    #                     elif robot_name == 'ergoCub':
    #                         gains['feet'] = np.array([10.] * 3 + [1.0] * 3)
    #                 frame_targets_dict = pack_current_targets(ik_cfree_planner, plan_to_model_frames, t)
    #                 if t < (i + 1) * T:
    #                     dmodel = createMultiFrameActionModel(state,
    #                                                          actuation,
    #                                                          x0,
    #                                                          plan_to_model_ids,
    #                                                          ['RH', 'LF'],
    #                                                          ee_rpy,
    #                                                          frame_targets_dict,
    #                                                          None,
    #                                                          gains=gains,
    #                                                          terminal_step=b_terminal_step)
    #                     model_seqs += createSequence([dmodel], T/(N_base_square_up), 1)
    #                 else:
    #                     dmodel = createMultiFrameFinalActionModel(state,
    #                                                          actuation,
    #                                                          x0,
    #                                                          plan_to_model_ids,
    #                                                          ['RH', 'LF'],
    #                                                          ee_rpy,
    #                                                          frame_targets_dict,
    #                                                          None,
    #                                                          gains=gains,
    #                                                          terminal_step=b_terminal_step)
    #                     model_seqs += createFinalSequence([dmodel])
    #                 lh_targets.append(frame_targets_dict['LH'])
    #                 rh_targets.append(frame_targets_dict['RH'])
    #                 base_targets.append(frame_targets_dict['torso'])
    #                 rf_targets.append(frame_targets_dict['RF'])
    #                 rkn_targets.append(frame_targets_dict['R_knee'])
    #
    #         elif i == 4:
    #             # Reach door with left and right hand from inside
    #             N_square_up = N_horizon_lst[i]  # knots for squaring up
    #             for t in np.linspace(i*T, (i+1)*T, N_square_up):
    #                 if t == (i+1)*T:
    #                     b_terminal_step = True
    #                     if robot_name == 'g1':
    #                         gains['feet'] = np.array([10.] * 3 + [1.5] * 3)
    #                     elif robot_name == 'ergoCub':
    #                         gains['feet'] = np.array([10.] * 3 + [1.0] * 3)
    #                 frame_targets_dict = pack_current_targets(ik_cfree_planner, plan_to_model_frames, t)
    #                 if t < (i + 1) * T:
    #                     dmodel = createMultiFrameActionModel(state,
    #                                                          actuation,
    #                                                          x0,
    #                                                          plan_to_model_ids,
    #                                                          ['LF', 'RF'],
    #                                                          ee_rpy,
    #                                                          frame_targets_dict,
    #                                                          None,
    #                                                          gains=gains,
    #                                                          terminal_step=b_terminal_step)
    #                     model_seqs += createSequence([dmodel], T/(N_square_up), 1)
    #                 else:
    #                     dmodel = createMultiFrameFinalActionModel(state,
    #                                                               actuation,
    #                                                               x0,
    #                                                               plan_to_model_ids,
    #                                                               ['LF', 'RF'],
    #                                                               ee_rpy,
    #                                                               frame_targets_dict,
    #                                                               None,
    #                                                               gains=gains,
    #                                                               terminal_step=b_terminal_step)
    #                     model_seqs += createFinalSequence([dmodel])
    #                 lh_targets.append(frame_targets_dict['LH'])
    #                 rh_targets.append(frame_targets_dict['RH'])
    #                 base_targets.append(frame_targets_dict['torso'])
    #
    #     problem = crocoddyl.ShootingProblem(x0, sum(model_seqs, [])[:-1], model_seqs[-1][-1])
    #     fddp[i] = crocoddyl.SolverFDDP(problem)
    #
    #     # Adding callbacks to inspect the evolution of the solver (logs are printed in the terminal)
    #     fddp[i].setCallbacks([crocoddyl.CallbackLogger()])
    #
    #     # Solver settings
    #     max_iter = 200
    #     fddp[i].th_stop = 1e-3
    #
    #     # Set initial guess
    #     xs = [x0] * (fddp[i].problem.T + 1)
    #     us = fddp[i].problem.quasiStatic([x0] * fddp[i].problem.T)
    #     start_ddp_solve_time = time.time()
    #     print("Problem solved:", fddp[i].solve(xs, us, max_iter))
    #     dyn_seg_solve_time = time.time() - start_ddp_solve_time
    #     print("Number of iterations:", fddp[i].iter)
    #     print("Total cost:", fddp[i].cost)
    #     print("Gradient norm:", fddp[i].stoppingCriteria())
    #     print("Time to solve:", dyn_seg_solve_time)
    #     dyn_solve_time += dyn_seg_solve_time
    #
    #     # save data
    #     if B_SAVE_DATA:
    #         for ti in range(len(fddp[i].us)):
    #             data_saver.add('time', float(i*T + ti*T/(len(fddp[i].xs)-1)))
    #             data_saver.add('q_base', list(fddp[i].xs[ti][:7]))
    #             data_saver.add('q_joints', list(fddp[i].xs[ti][7:state.nq]))
    #             data_saver.add('qd_base', list(fddp[i].xs[ti][state.nq:state.nq+6]))
    #             data_saver.add('qd_joints', list(fddp[i].xs[ti][state.nq+6:]))
    #             data_saver.add('tau_joints', list(fddp[i].us[ti]))
    #             data_saver.advance()
    #
    #     # Set final state as initial state of next phase
    #     x0 = fddp[i].xs[-1]

    # print("[Compute Time] Dynamic feasibility check: ", dyn_solve_time)

    # lf_targets = robot_dyn_plan.lf_targets
    # lkn_targets = robot_dyn_plan.lkn_targets
    # rf_targets = robot_dyn_plan.rf_targets
    # rkn_targets = robot_dyn_plan.rkn_targets
    # lh_targets = robot_dyn_plan.lh_targets
    # rh_targets = robot_dyn_plan.rh_targets
    # base_targets = robot_dyn_plan.base_targets
    # fddp = robot_dyn_plan.fddp

    # Creating display
    if B_VISUALIZE:
        save_freq = 1
        display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                          rob_data, vis_data, ctrl_freq=np.average(N_horizon_lst)/T, save_freq=save_freq)
        display.add_robot("door", door_model, door_collision_model, door_visual_model, door_pos, door_pose[3:])
        display.display_targets("lfoot_target", robot_dyn_plan.lf_targets, [1, 1, 0])
        display.display_targets("lknee_target", robot_dyn_plan.lkn_targets, [0, 0, 1])
        display.display_targets("rfoot_target", robot_dyn_plan.rf_targets, [1, 1, 0])
        display.display_targets("rknee_target", robot_dyn_plan.rkn_targets, [0, 0, 1])
        display.display_targets("lhand_target", robot_dyn_plan.lh_targets, [0.5, 0, 0])
        display.display_targets("rhand_target", robot_dyn_plan.rh_targets, [0.5, 0, 0])
        display.display_targets("base_target", robot_dyn_plan.base_targets, [0, 0.5, 0])
        # TODO grab joint name below from pinocchio
        display.add_arrow("forces/left_ankle_roll_joint", color=[1, 0, 0])
        display.add_arrow("forces/right_ankle_roll_joint", color=[0, 0, 1])
        display.add_arrow("forces/left_elbow_roll_joint", color=[0, 1, 0])
        display.add_arrow("forces/right_elbow_roll_joint", color=[0, 1, 0])
        display.displayFromCrocoddylSolver(robot_dyn_plan.fddp)
        viz_to_hide = list(("base_target", "lhand_target", "rhand_target",
                            "lfoot_target", "lknee_target",
                            "rfoot_target", "rknee_target"))
        display.hide_visuals(viz_to_hide)
        if B_SAVE_HTML:
            display.save_html(cwd + "/data/", robot_name + "_door_crossing.html")

    fig_idx = 1
    if B_SHOW_JOINT_PLOTS:
        for it in robot_dyn_plan.fddp:
            log = it.getCallbacks()[0]
            # Reduced to check Draco3's RCJ constraint
            if robot_name == 'draco3':
                xs_reduced = np.array(log.xs)[:, [l_constr_ids[0], l_constr_ids[1],
                                                  r_constr_ids[0], r_constr_ids[1]]]
                us_reduced = np.array(log.us)[:, [l_constr_ids_u[0], l_constr_ids_u[1],
                                                  r_constr_ids_u[0], r_constr_ids_u[1]]]
            elif robot_name == 'g1':
                g1_leg_joint_ids = get_g1_lleg_joint_ids(rob_model)
                xs_reduced = np.array(log.xs)[:, g1_leg_joint_ids]
                us_reduced = np.array(log.us)[:, g1_leg_joint_ids]
            else:
                xs_reduced = np.array(log.xs)
                us_reduced = np.array(log.us)
            crocoddyl.plotOCSolution(xs_reduced, us_reduced, figIndex=fig_idx, show=False)
            fig_idx += 1
            # crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads,
            #                           log.stops, log.steps, figIndex=fig_idx, show=False)
            # fig_idx +=1
        plt.show()

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
                    if rob_model.names[cur_link] == "left_ankle_roll_joint":    #  "l_ankle_ie":
                        rf_lfoot[:, time_idx] = contact['f'].linear
                    elif rob_model.names[cur_link] == "right_ankle_roll_joint": # "r_ankle_ie":
                        rf_rfoot[:, time_idx] = contact['f'].linear
                    elif rob_model.names[cur_link] == "left_elbow_roll_joint":  # l_wrist_pitch
                        rf_lwrist[:, time_idx] = contact['f'].linear
                    elif rob_model.names[cur_link] == "right_elbow_roll_joint": # r_wrist_pitch
                        rf_rwrist[:, time_idx] = contact['f'].linear
                    else:
                        print(f"ERROR: Non-specified contact {rob_model.names[cur_link]}")
                dt = it.problem.runningModels[0].dt     # assumes constant dt over fddp sequence
                if time_idx < len(sim_time) - 1:
                    sim_time[time_idx+1] = sim_time[time_idx] + dt
                    time_idx += 1
                else:
                    continue

        plot_vector_traj(sim_time, rf_lfoot.T, 'RF LFoot (Local)', Fxyz_labels)
        plot_vector_traj(sim_time, rf_rfoot.T, 'RF RFoot (Local)', Fxyz_labels)
        plot_vector_traj(sim_time, rf_lwrist.T, 'RF LWrist (Local)', Fxyz_labels)
        plot_vector_traj(sim_time, rf_rwrist.T, 'RF RWrist (Local)', Fxyz_labels)
        plt.show()


# def pack_current_targets(ik_cfree_planner, plan_to_model_frames, t):
#     lfoot_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('LF'), t)
#     lknee_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('L_knee'), t)
#     rfoot_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('RF'), t)
#     rknee_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('R_knee'), t)
#     lhand_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('LH'), t)
#     rhand_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('RH'), t)
#     base_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('torso'), t)
#     frame_targets_dict = {
#         'torso': base_t,
#         'LF': lfoot_t,
#         'RF': rfoot_t,
#         'L_knee': lknee_t,
#         'R_knee': rknee_t,
#         'LH': lhand_t,
#         'RH': rhand_t
#     }
#     return frame_targets_dict
#

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=0,
                        help="Contact sequence to solve for")
    parser.add_argument("--robot_name", type=str, default='g1',
                        help="Robot name to use for planning")
    args = parser.parse_args()
    main(args)
