import os
import pickle
import sys
from collections import OrderedDict

import mim_solvers
import pinocchio
from pinocchio.visualize import MeshcatVisualizer

import config.multicontact.g1_baseline_planner_config as g1_baseline_params
from pnc.planner.multicontact.kin_feasibility import SCARobotGeometry
from util.environment_creator import TiltedStairs
import pnc.planner.multicontact.contact_sequence_plans.tilted_stairs_plans as stairs_plan

cwd = os.getcwd()
sys.path.append(cwd)

# Collision free description
from pydrake.geometry.optimization import HPolyhedron

# Kinematic feasibility
from pnc.planner.multicontact.kin_feasibility.planner_surface_contact import PlannerSurfaceContact, \
    MotionFrameSequencer, get_contact_seq_from_fixed_frames_seq, get_contact_planes_from_motion_frames_seq
# Tools for dynamic feasibility
from humanoid_action_models import *
from pnc.planner.multicontact.dyn_feasibility.G1MulticontactPlanner import G1MulticontactPlanner
from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import ContactSequence

# Visualization tools
import matplotlib.pyplot as plt
from plot.helper import plot_vector_traj, Fxyz_labels
import plot.meshcat_utils as vis_tools
from plot.multiontact_plotter import MulticontactPlotter
# Save data
from plot.data_saver import *

B_SHOW_JOINT_PLOTS = True
B_SHOW_JOINT_LIM_PLOTS = True
B_SHOW_COST_PLOTS = True
B_SHOW_GRF_PLOTS = True
B_VISUALIZE = True
B_SAVE_DYN_DATA = False
B_SAVE_HTML = False
B_USE_SELF_COLLISION_AVOIDANCE = True
B_USE_KNEES = True
B_USE_IMPULSE_MODEL = False
SOLVER_TYPE = 'BoxFDDP' # {SQP, BoxFDDP}

frame_names_lst = ['torso', 'LF', 'RF', 'L_knee', 'R_knee' ,'LH', 'RH']

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
        floating_base = np.array([0., 0., 0.7, 0., 0., 0., 1.])
    else:
        raise ValueError(f"Unspecified default initial pose for g1 in environment: {env}")
    return np.concatenate((floating_base, q0))


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
            np.array([0.14, 0.9, 0.4]) + door_pos + door_width)
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


def get_five_stage_two_hand_contact_sequence(robot_name: str,
                                             robot_data: pinocchio.Data,
                                             plan_to_model_ids: dict[str: int],
                                             goal_step_length: float,
                                             standing_pos: np.array, ):
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

    final_lf_pos = robot_data.oMf[plan_to_model_ids['LF']].translation + np.array([goal_step_length, 0., 0.])
    final_rf_pos = robot_data.oMf[plan_to_model_ids['RF']].translation + np.array([goal_step_length, 0., 0.])
    final_torso_pos = standing_pos + np.array([goal_step_length, 0., 0.,])
    final_rh_pos = robot_data.oMf[plan_to_model_ids['RH']].translation + np.array([goal_step_length, 0., 0.,])
    final_lh_pos = robot_data.oMf[plan_to_model_ids['LH']].translation + np.array([goal_step_length, 0., 0.,])

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
                                               robot_data: pinocchio.Data,
                                               plan_to_model_ids: dict[str: int],
                                               goal_step_length: float,
                                               standing_pos: np.array, ):
    ###### Previously used key locations
    # door_l_outer_location = np.array([0.45, 0.35, 1.2])
    # door_r_outer_location = np.array([0.45, -0.35, 1.2])
    if robot_name == 'g1':
        # G1 settings
        door_l_inner_location = np.array([0.3, 0.35, 1.1])
        door_r_inner_location = np.array([0.34, -0.35, 1.1])
        ft_kn_offset = np.array([0.15, 0., 0.28])
    else:
        # ergoCub settings
        door_l_inner_location = np.array([0.3, 0.35, 1.0])
        door_r_inner_location = np.array([0.34, -0.35, 1.0])
        ft_kn_offset = np.array([0.2, 0., 0.3])

    final_lf_pos = robot_data.oMf[plan_to_model_ids['LF']].translation + np.array([goal_step_length, 0., 0.])
    final_rf_pos = robot_data.oMf[plan_to_model_ids['RF']].translation + np.array([goal_step_length, 0., 0.])
    final_torso_pos = standing_pos + np.array([goal_step_length, 0., 0.,])
    final_rh_pos = robot_data.oMf[plan_to_model_ids['RH']].translation + np.array([goal_step_length, 0., 0.,])
    final_lh_pos = robot_data.oMf[plan_to_model_ids['LH']].translation + np.array([goal_step_length, 0., 0.,])

    if robot_name == 'g1':
        intermediate_lf_pos = np.array([0.35, final_lf_pos[1], 0.44])
        # intermediate_rh_pos = np.array([0.45, -0.2, 1.0])
    elif robot_name == 'ergoCub':
        intermediate_lf_pos = np.array([0.30, final_lf_pos[1]-0.02, 0.49])
        intermediate_rh_pos = np.array([0.40, -0.15, 1.0])

    # initialize fixed and motion frame sets
    fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

    # ---- Step 1: L hand to frame
    fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
    if robot_name == 'g1':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            # 'torso': starting_torso_pos + np.array([0.1, starting_lf_pos[1], 0.0])
                                            })
    elif robot_name == 'ergoCub':
        motion_frames_seq.add_motion_frame({
                                            'LH': door_l_inner_location,
                                            })
    lh_contact_front = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    # lh_contact_front.set_contact_breaking_velocity(np.array([0, -1, 0.]))
    motion_frames_seq.add_contact_surfaces([lh_contact_front])

    # ---- Step 2: step on knee-knocker with left foot
    if robot_name == 'g1':
        fixed_frames.append(['RF', 'R_knee', 'LH'])   # frames that must not move
    elif robot_name == 'ergoCub':
        fixed_frames.append(['RF', 'R_knee', 'LH'])
    motion_frames_seq.add_motion_frame({
                        # 'RH': intermediate_rh_pos,      # added for ergoCub
                        'LF': intermediate_lf_pos,
                        'L_knee': intermediate_lf_pos + ft_kn_offset,
                        'RH': door_r_inner_location,
    })
    lf_contact_on = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    rh_contact_wall = PlannerSurfaceContact('RH', np.array([0, 1, 0]))
    motion_frames_seq.add_contact_surfaces([lf_contact_on, rh_contact_wall])

    # ---- Step 3: switch hand contacts
    # fixed_frames.append(['LF', 'L_knee', 'RF', 'R_knee'])   # frames that must not move
    # motion_frames_seq.add_motion_frame({
    # })
    # motion_frames_seq.add_contact_surfaces([rh_contact_wall])

    # ---- Step 3: step through door with right foot
    if robot_name == 'g1':
        fixed_frames.append(['LF', 'L_knee', 'RH'])   # frames that must not move
    elif robot_name == 'ergoCub':
        fixed_frames.append(['LF', 'L_knee', 'RH'])  # frames that must not move
    motion_frames_seq.add_motion_frame({
                        'R_knee': final_rf_pos + ft_kn_offset,
                        'RF': final_rf_pos,
                        'LH': door_l_inner_location})
    rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
    lh_contact_inner = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
    motion_frames_seq.add_contact_surfaces([rf_contact_over, lh_contact_inner])

    # ---- Step 5: balance / square up
    # fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
    fixed_frames.append(['RF', 'R_knee', 'LH'])
    motion_frames_seq.add_motion_frame({
        'torso': final_torso_pos,
        'LF': final_lf_pos,
        'L_knee': final_lf_pos + ft_kn_offset,
        'RH': final_rh_pos,
        # 'LH': final_lh_pos
    })
    lf_square_up = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
    motion_frames_seq.add_contact_surfaces([lf_square_up])

    # ---- Step 6: balance
    fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'RH'])
    motion_frames_seq.add_motion_frame({'LH': final_lh_pos})

    return fixed_frames, motion_frames_seq


def get_on_knocker_balanced_contact_sequence(robot_name: str,
                                               robot_data: pinocchio.Data,
                                               plan_to_model_ids: dict[str: int],
                                               goal_step_length: float,
                                               standing_pos: np.array, ):
    if robot_name == 'g1':
        # G1 settings
        door_l_inner_location = np.array([0.3, 0.35, 1.0])
        door_r_inner_location = np.array([0.34, -0.35, 1.0])
        ft_kn_offset = np.array([0.15, 0., 0.28])
    else:
        # ergoCub settings
        door_l_inner_location = np.array([0.3, 0.35, 1.1])
        door_r_inner_location = np.array([0.34, -0.35, 1.1])
        ft_kn_offset = np.array([0.2, 0., 0.3])

    final_lf_pos = robot_data.oMf[plan_to_model_ids['LF']].translation + np.array([goal_step_length, 0., 0.])
    final_rf_pos = robot_data.oMf[plan_to_model_ids['RF']].translation + np.array([goal_step_length, 0., 0.])
    final_torso_pos = standing_pos + np.array([goal_step_length, 0., 0.,])
    final_rh_pos = robot_data.oMf[plan_to_model_ids['RH']].translation + np.array([goal_step_length, 0., 0.,])
    final_lh_pos = robot_data.oMf[plan_to_model_ids['LH']].translation + np.array([goal_step_length, 0., 0.,])

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


def create_frame_targets_dict(motion_frames_seq: MotionFrameSequencer,
                              fixed_frames: list[list[str]],
                              robot_data: pinocchio.Data,
                              plan_to_model_ids: dict[str: int]):
    all_frame_targets = []
    current_frame_targets = {}
    phase = 0

    # get initial frame positions
    for fr_name in frame_names_lst:
        current_frame_targets[fr_name] = robot_data.oMf[plan_to_model_ids[fr_name]].translation
    phase += 1
    all_frame_targets.append(current_frame_targets)

    # get the next frame positions from either fixed frames or motion frames
    for _ in range(len(motion_frames_seq.motion_frame_lst)):
        current_frame_targets = {}
        for fr_name in frame_names_lst:
            if fr_name in fixed_frames[phase - 1]:
                # fixed frame: keep previous position
                current_frame_targets[fr_name] = all_frame_targets[-1][fr_name]
            elif fr_name in motion_frames_seq.motion_frame_lst[phase - 1].keys():
                # motion frame: get new target position
                current_frame_targets[fr_name] = motion_frames_seq.motion_frame_lst[phase - 1][fr_name]
            else:
                continue
        all_frame_targets.append(current_frame_targets)
        phase += 1

    return all_frame_targets


def main(args):
    env = args.env
    contact_seq = args.sequence
    robot_name = args.robot_name

    #
    # Initialize frames to consider for contact planning
    #
    plan_to_model_frames = OrderedDict()
    force_joint_frames = OrderedDict()
    if robot_name == 'g1':
        plan_to_model_frames['torso'] = 'torso_primitive_shape'
        # plan_to_model_frames['torso'] = 'pelvis'
        plan_to_model_frames['LF'] = 'left_ankle_roll_link'
        plan_to_model_frames['RF'] = 'right_ankle_roll_link'
        plan_to_model_frames['L_knee'] = 'left_knee_link'
        plan_to_model_frames['R_knee'] = 'right_knee_link'
        plan_to_model_frames['LH'] = 'left_rubber_hand'
        plan_to_model_frames['RH'] = 'right_rubber_hand'
        # plan_to_model_frames['LH'] = 'left_hand_palm_joint'
        # plan_to_model_frames['RH'] = 'right_hand_palm_joint'
        force_joint_frames['LF'] = "left_ankle_roll_joint"
        force_joint_frames['RF'] = "right_ankle_roll_joint"
        force_joint_frames['LH'] = "left_wrist_yaw_joint"
        force_joint_frames['RH'] = "right_wrist_yaw_joint"
        package_dir = cwd + "/robot_model/g1_description"
        # robot_urdf_file = package_dir + "/g1_29dof_simple_collisions.urdf"
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
    geom_model = pin.buildGeomFromUrdf(rob_model,
                                       robot_urdf_file,
                                       pin.GeometryType.COLLISION)
    geom_model.addAllCollisionPairs()
    root_to_torso_offset = get_root_to_torso_offset(geom_model)
    # if B_USE_SELF_COLLISION_AVOIDANCE:
    #     root_to_torso_offset = get_root_to_torso_offset(geom_model)
    # else:
    #     root_to_torso_offset = np.array([0., 0., 0.])

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
            seq_str = 'over_'
        elif contact_seq == 1:
            seq_str = 'on_'
        elif contact_seq == 2:
            seq_str = 'on_balanced_'
        else:
            raise NotImplementedError('Contact sequence not defined')

        # load navy environment (with respective door offset) and initial robot pose
        door_pos = np.array([0.32, 0., 0.])
        step_length = 0.35
        if robot_name == 'g1':
            q0 = get_g1_default_initial_pose(rob_model.nq - 7)
            door_pos = np.array([0.32, 0., 0.])
            step_length = 0.46
            planner_params = g1_baseline_params.MultiContactDoorConfig()
        elif robot_name == 'valkyrie':
            raise NotImplementedError('Valkyrie door environment not yet implemented')
        elif robot_name == 'ergoCub':
            raise NotImplementedError('ergoCub door environment not yet implemented')
        else:
            raise NotImplementedError('Robot default configuration not specified')
        v0 = np.zeros(rob_model.nv)
        x0 = np.concatenate([q0, v0])
        door_pose, obstacles, domain_ubody, domain_lbody_l, domain_lbody_r = load_navy_env(robot_name, door_pos)
    elif env == 'stairs':
        seq_str = '_'
        # create tilted stairs environment
        stairs = TiltedStairs()

        if robot_name == 'g1':
            q0 = get_g1_default_initial_pose(rob_model.nq - 7, env)
            planner_params = g1_baseline_params.MultiContactTiltedStairsConfig()
        elif robot_name == 'ergoCub':
            raise NotImplementedError('ergoCub stairs environment not yet implemented')
        else:
            raise NotImplementedError('Robot default configuration not specified for stairs')
        v0 = np.zeros(rob_model.nv)
        x0 = np.concatenate([q0, v0])
    else:
        raise NotImplementedError('Specified environment cannot be loaded')


    # Update Pinocchio model
    pin.forwardKinematics(rob_model, rob_data, q0)
    pin.updateFramePlacements(rob_model, rob_data)

    if B_VISUALIZE:
        if env == 'door':
            visualizer, door_model, door_collision_model, door_visual_model \
                = visualize_env(rob_model, col_model, vis_model, q0, door_pose)
        else:
            visualizer, _, __, ___ = visualize_env(rob_model, col_model, vis_model, q0)
    else:
        visualizer = None


    # hand-chosen five-stage sequence of contacts
    standing_pos = q0[:3] + root_to_torso_offset
    if robot_name in {'g1', 'ergoCub'}:   # smaller robots have been set up with different contact sequences
        # Note: the contact sequence was defined earlier for the stairs environment
        if env == 'door':
            if contact_seq == 0:    # step through door
                fixed_frames_seq, motion_frames_seq = get_five_stage_two_hand_contact_sequence(robot_name,
                                                                                               rob_data,
                                                                                               plan_to_model_ids,
                                                                                               step_length,
                                                                                               standing_pos,
                                                                                                )
            elif contact_seq == 1:  # step on knee-knocker RF
                fixed_frames_seq, motion_frames_seq = get_five_stage_on_knocker_contact_sequence(robot_name,
                                                                                                 rob_data,
                                                                                                 plan_to_model_ids,
                                                                                                 step_length,
                                                                                                 standing_pos,
                                                                                                 )
            elif contact_seq == 2:  # step on knee-knocker RF both hands
                fixed_frames_seq, motion_frames_seq = get_on_knocker_balanced_contact_sequence(robot_name,
                                                                                                 rob_data,
                                                                                                 plan_to_model_ids,
                                                                                                 step_length,
                                                                                                 standing_pos,
                                                                                               )
            else:
                NotImplementedError(f"Contact sequence {contact_seq} not implemented")
    else:
        raise NotImplementedError(f"Contact sequence for robot {robot_name} not implemented")

    contact_seqs = get_contact_seq_from_fixed_frames_seq(fixed_frames_seq)
    contact_seq_planes = get_contact_planes_from_motion_frames_seq(contact_seqs, motion_frames_seq)

    # planner parameters
    T = 3

    # use self-collision avoidance
    sca_geometry = None
    if B_USE_SELF_COLLISION_AVOIDANCE:
        sca_str = '_sca_'
        sca_geometry = SCARobotGeometry(package_dir, robot_urdf_file, plan_to_model_frames)

    # create targets dictionary
    N_horizon_lst = planner_params.N_HORIZON_LST
    frame_targets_dict = create_frame_targets_dict(motion_frames_seq, fixed_frames_seq,
                                                   rob_data,
                                                   plan_to_model_ids
    )

    #
    # Set up trajectory optimization problem
    #
    #
    state = crocoddyl.StateMultibody(rob_model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    contact_sequence = ContactSequence(contact_seq_planes, N_horizon_lst, T)
    model_seq = []
    if robot_name == 'g1':
        # construct MulticontactPlanner just to get same name conventions
        g1_planner_dummy = G1MulticontactPlanner(rob_model, contact_sequence, None, None)
        for i in range(len(N_horizon_lst)):
            DT = T / (N_horizon_lst[i] - 1)
            frames_in_contact = contact_sequence.contact_planes_seq[i]
            if i < len(N_horizon_lst) - 1:
                next_frames_in_contact = contact_sequence.contact_planes_seq[i+1]
            else:   # at the last phase, next contact is the same as current
                next_frames_in_contact = contact_sequence.contact_planes_seq[i]
            dmodel = createMultiFrameActionModel(state,
                                                 actuation,
                                                 x0,   # used to dampen joint velocities (via params weights)
                                                 plan_to_model_ids,
                                                 frames_in_contact,
                                                 next_frames_in_contact,
                                                 frame_targets_dict[i+1],
                                                 joint_names_dict=g1_planner_dummy.joint_names_dict,
                                                 planner_weights=planner_params,
                                                 geom_model=geom_model,
                                                 robot_model=rob_model,
                                                 b_sca=False
            )
            model_seq += createSequence([dmodel], DT, N_horizon_lst[i])
            if B_USE_IMPULSE_MODEL:
                imp_model = createMultiFrameFinalImpulseModel(state,
                                                actuation,
                                                None,
                                                plan_to_model_ids,
                                                planner_params,
                                                contact_sequence)
                # TODO double-check how impulse model is added to sequence
                model_seq += createSequence([imp_model], DT, N_horizon_lst[i])


        frames_in_contact = contact_sequence.contact_planes_seq[-1]
        dmodel = createMultiFrameFinalActionModel(state,
                                                  actuation,
                                                  x0,   # used to dampen joint velocities (via params weights)
                                                  plan_to_model_ids,
                                                  frames_in_contact,
                                                  frames_in_contact,
                                                  frame_targets_dict[-1],
                                                  joint_names_dict=g1_planner_dummy.joint_names_dict,
                                                  planner_weights=planner_params,
                                                  robot_model=rob_model,
                                                  )
        model_seq += createFinalSequence([dmodel])

    else:
        raise NotImplementedError(f"Matching multicontact planner for {robot_name} not found")

    #
    # Solve problem
    #
    problem = crocoddyl.ShootingProblem(x0, sum(model_seq, [])[:-1], model_seq[-1][-1])
    if SOLVER_TYPE == 'SQP':
        fddp = mim_solvers.SolverCSQP(problem)
        fddp.setCallbacks([mim_solvers.CallbackLogger(), mim_solvers.CallbackVerbose()])
        # fddp.filter_size = 10
        fddp.eps_abs = 1e-1
        fddp.eps_rel = 1e-1
    elif SOLVER_TYPE == 'BoxFDDP':
        fddp = crocoddyl.SolverBoxFDDP(problem)
        fddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    else:
        raise NotImplementedError('Selected solver type not implemented')
    max_iter = 500
    fddp.th_stop = 1e-3
    fddp.th_gapTol = 1e-2
    fddp.reg_incFactor = 3
    fddp.reg_decFactor = 3

    # Set initial guess
    ini_frames_in_contact = contact_sequence.contact_planes_seq[0]
    xs = [x0] * (fddp.problem.T + 1)
    us_static = quasi_static_ocp(ini_frames_in_contact, plan_to_model_ids, state.pinocchio, x0)
    us = [us_static] * fddp.problem.T
    print("[Crocoddyl] Problem solved to convergence:", fddp.solve(xs, us, max_iter))
    print(f"[Crocoddyl] Is feasible: {fddp.isFeasible}")
    print("[Crocoddyl] Number of iterations:", fddp.iter)

    # Create display
    if B_VISUALIZE:
        save_freq = 10
        display_idx = len(N_horizon_lst) - 1
        display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                          rob_data, vis_data, col_data, ctrl_freq=np.average(N_horizon_lst)/T, save_freq=save_freq)
        if 'door' in env:
            display.add_robot("door", door_model, door_collision_model, door_visual_model, door_pos, door_pose[3:])
        elif 'stairs' in env:
            display.add_shapes_from(stairs.obstacles_vis)
        # display.display_targets("lfoot_target", robot_dyn_plan.lf_targets[display_idx], [1, 1, 0])
        # display.display_targets("lknee_target", robot_dyn_plan.lkn_targets[display_idx], [0, 0, 1])
        # display.display_targets("rfoot_target", robot_dyn_plan.rf_targets[display_idx], [1, 1, 0])
        # display.display_targets("rknee_target", robot_dyn_plan.rkn_targets[display_idx], [0, 0, 1])
        # display.display_targets("lhand_target", robot_dyn_plan.lh_targets[display_idx], [0.5, 0, 0])
        # display.display_targets("rhand_target", robot_dyn_plan.rh_targets[display_idx], [0.5, 0, 0])
        # display.display_targets("base_target", robot_dyn_plan.base_targets[display_idx], [0, 0.5, 0])
        display.add_arrow("forces/" + force_joint_frames['LF'], color=[1, 0, 0])
        display.add_arrow("forces/" + force_joint_frames['RF'], color=[0, 0, 1])
        display.add_arrow("forces/" + force_joint_frames['LH'], color=[0, 1, 0])
        display.add_arrow("forces/" + force_joint_frames['RH'], color=[0, 1, 0])
        display.displayFromCrocoddylSolver([fddp])
        # viz_to_hide = list(("base_target", "lhand_target", "rhand_target",
        #                     "lfoot_target", "lknee_target",
        #                     "rfoot_target", "rknee_target"))
        display.hide_visuals(["env/2"])
        display.hide_visuals(["g1_29dof_lock_waist/collisions"])
        if B_SAVE_HTML:
            display.save_html(cwd + "/experiment_data/RAL/baseline_", robot_name + sca_str + seq_str + env + "_anim.html")

    # if B_SHOW_JOINT_PLOTS or B_SHOW_COST_PLOTS or B_SHOW_JOINT_LIM_PLOTS:
    #     plan_plotter = MulticontactPlotter(robot_dyn_plan)
    #     if B_SHOW_JOINT_PLOTS:
    #         plan_plotter.plot_reduced_xs_us()
    #     if B_SHOW_COST_PLOTS:
    #         # plan_plotter.plot_costs('seq')
    #         # plan_plotter.plot_costs('full', ['right_hip_roll_joint_to_torso_primitive_shape_0_sca'])
    #         plan_plotter.plot_costs()
    #         plan_plotter.plot_constraint_violations()
    #     if B_SHOW_JOINT_LIM_PLOTS:
    #         plan_plotter.plot_joint_limit_margins()
        # plt.show()

    if B_SHOW_GRF_PLOTS or B_SAVE_DYN_DATA:
        # Note: contact_links are l_ankle_ie, r_ankle_ie, l_wrist_pitch, r_wrist_pitch
        # sim_steps_list = [len(fddp[i].us) for i in range(len(fddp))]
        sim_steps = np.sum(sum(N_horizon_lst))
        sim_time = np.zeros((sim_steps,))
        rf_lfoot, rf_rfoot, rf_lwrist, rf_rwrist = np.zeros((3, sim_steps)), \
            np.zeros((3, sim_steps)), np.zeros((3, sim_steps)), np.zeros((3, sim_steps))
        w_rf_lfoot, w_rf_rfoot, w_rf_lwrist, w_rf_rwrist = np.zeros((3, sim_steps)), \
            np.zeros((3, sim_steps)), np.zeros((3, sim_steps)), np.zeros((3, sim_steps))
        time_idx = 0

        rf_list = vis_tools.get_force_trajectory_from_solver(fddp)
        for rf_t in rf_list:
            for contact in rf_t:
                # determine contact link
                cur_link = int(contact['key'])
                if rob_model.names[cur_link] == force_joint_frames['LF']:
                    rf_lfoot[:, time_idx] = contact['f'].linear
                    w_rf_lfoot[:, time_idx] = contact['w_f'].linear
                elif rob_model.names[cur_link] == force_joint_frames['RF']:
                    rf_rfoot[:, time_idx] = contact['f'].linear
                    w_rf_rfoot[:, time_idx] = contact['w_f'].linear
                elif rob_model.names[cur_link] == force_joint_frames['LH']:
                    rf_lwrist[:, time_idx] = contact['f'].linear
                    w_rf_lwrist[:, time_idx] = contact['w_f'].linear
                elif rob_model.names[cur_link] == force_joint_frames['RH']:
                    rf_rwrist[:, time_idx] = contact['f'].linear
                    w_rf_rwrist[:, time_idx] = contact['w_f'].linear
                else:
                    print(f"ERROR: Non-specified contact {rob_model.names[cur_link]}")
            dt = fddp.problem.runningModels[0].dt     # assumes constant dt over fddp sequence
            if time_idx < len(sim_time) - 1:
                sim_time[time_idx+1] = sim_time[time_idx] + dt
                time_idx += 1
            else:
                continue

        if B_SHOW_GRF_PLOTS:
            plot_vector_traj(sim_time, w_rf_lfoot.T, 'RF LFoot (World)', Fxyz_labels)
            plot_vector_traj(sim_time, w_rf_rfoot.T, 'RF RFoot (World)', Fxyz_labels)
            plot_vector_traj(sim_time, w_rf_lwrist.T, 'RF LWrist (World)', Fxyz_labels)
            plot_vector_traj(sim_time, w_rf_rwrist.T, 'RF RWrist (World)', Fxyz_labels)
            plt.show()

    if B_SAVE_DYN_DATA:
        # Saving data tools
        dyn_data_saver = DataSaver(robot_name + sca_str + seq_str + env +'.pkl')
        dyn_data_saver.add('contact_seq_planes', contact_seq_planes)
        for (i, fp) in enumerate([fddp]):
            com_lst = []
            torso_pos, lf_pos, rf_pos, lkn_pos, rkn_pos, lh_pos, rh_pos = [], [], [], [], [], [], []
            if i == len([fddp])-1:      # variables that need to be logged only once
                dyn_data_saver.add('w_grf_lfoot', w_rf_lfoot.tolist())
                dyn_data_saver.add('w_grf_rfoot', w_rf_rfoot.tolist())
                dyn_data_saver.add('w_grf_lhand', w_rf_lwrist.tolist())
                dyn_data_saver.add('w_grf_rhand', w_rf_rwrist.tolist())
                dyn_data_saver.add('time', sim_time.tolist())
            log = fp.getCallbacks()[0]
            q = np.array(log.xs)[:, :rob_model.nq]
            qd = np.array(log.xs)[:, rob_model.nq:]
            dyn_data_saver.add('joint_pos', q.tolist())
            dyn_data_saver.add('joint_vel', qd.tolist())
            dyn_data_saver.add('joint_torque', [vec.tolist() for vec in log.us.tolist()])
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


def get_root_to_torso_offset(geom_model):
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
    parser.add_argument("--sequence", type=int, default=1,
                        help="Contact sequence to solve for")
    parser.add_argument("--robot_name", type=str, default='g1',
                        choices=['g1', 'valkyrie', 'ergoCub'],
                        help="Robot name to use for planning")
    args = parser.parse_args()
    main(args)
