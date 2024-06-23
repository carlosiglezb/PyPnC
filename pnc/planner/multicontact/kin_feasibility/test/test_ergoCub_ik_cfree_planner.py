import unittest

import os
import sys
from collections import OrderedDict

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from pydrake.geometry.optimization import HPolyhedron

from pnc.planner.multicontact.kin_feasibility.frame_traversable_region import FrameTraversableRegion
from pnc.planner.multicontact.kin_feasibility.ik_cfree_planner import IKCFreePlanner
from pnc.planner.multicontact.kin_feasibility.locomanipulation_frame_planner import LocomanipulationFramePlanner
from pnc.planner.multicontact.planner_surface_contact import MotionFrameSequencer, PlannerSurfaceContact
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
from util import util
from vision.iris.iris_geom_interface import IrisGeomInterface
from vision.iris.iris_regions_manager import IrisRegionsManager

cwd = os.getcwd()
sys.path.append(cwd)

b_visualize = True
b_use_knees = True


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


class TestIKCFreePlanner(unittest.TestCase):

    def setUp(self):
        robot_name = 'ergoCub'
        self.robot_name = robot_name
        self.aux_frames_path = cwd + '/pnc/reachability_map/output/' + robot_name + '/' + \
                                robot_name + '_aux_frames.yaml'

        # create navy door environment
        self.door_pos = np.array([0.30, 0., 0.])
        door_width = np.array([0.025, 0., 0.])
        dom_ubody_lb = np.array([-1.6, -0.8, 0.5])
        dom_ubody_ub = np.array([1.6, 0.8, 2.1])
        dom_lbody_lb = np.array([-1.6, -0.8, -0.])
        dom_lbody_ub = np.array([1.6, 0.8, 0.95])
        floor = HPolyhedron.MakeBox(
                                np.array([-2, -0.9, -0.05]) + self.door_pos + door_width,
                                np.array([2, 0.9, -0.001]) + self.door_pos + door_width)
        knee_knocker_base = HPolyhedron.MakeBox(
                                    np.array([-0.025, -0.9, 0.0]) + self.door_pos + door_width,
                                    np.array([0.025, 0.9, 0.4]) + self.door_pos + door_width)
        knee_knocker_lwall = HPolyhedron.MakeBox(
                                    np.array([-0.025, 0.9-0.518, 0.0]) + self.door_pos + door_width,
                                    np.array([0.025, 0.9, 2.2]) + self.door_pos + door_width)
        knee_knocker_rwall = HPolyhedron.MakeBox(
                                    np.array([-0.025, -0.9, 0.0]) + self.door_pos + door_width,
                                    np.array([0.025, -(0.9-0.518), 2.2]) + self.door_pos + door_width)
        knee_knocker_top = HPolyhedron.MakeBox(
                                    np.array([-0.025, -0.9, 1.85]) + self.door_pos + door_width,
                                    np.array([0.025, 0.9, 2.25]) + self.door_pos + door_width)
        knee_knocker_llip = HPolyhedron.MakeBox(
                                    np.array([-0.035, 0.9-0.518, 0.25]) + self.door_pos + door_width,
                                    np.array([0.035, 0.9-0.518+0.15, 2.0]) + self.door_pos + door_width)
        knee_knocker_rlip = HPolyhedron.MakeBox(
                                    np.array([-0.035, -(0.9-0.518+0.15), 0.25]) + self.door_pos + door_width,
                                    np.array([0.035,  -(0.9-0.518), 2.0]) + self.door_pos + door_width)
        self.obstacles = [floor,
                          knee_knocker_base,
                          knee_knocker_lwall,
                          knee_knocker_rwall,
                          knee_knocker_llip,
                          knee_knocker_rlip,
                          knee_knocker_top]
        self.domain_ubody = HPolyhedron.MakeBox(dom_ubody_lb, dom_ubody_ub)
        self.domain_lbody = HPolyhedron.MakeBox(dom_lbody_lb, dom_lbody_ub)

        # load robot
        mesh_dir = cwd + "/robot_model/" + robot_name
        self.robot = pin.RobotWrapper.BuildFromURDF(
            mesh_dir + "/" + robot_name + ".urdf",
            mesh_dir,
            root_joint=pin.JointModelFreeFlyer())

        # set-up easy access to fwd kinematics for IRIS seeds
        self.robot_fwdk = PinocchioRobotSystem(
            mesh_dir + "/" + robot_name + ".urdf",
            mesh_dir, False, False)

        # load default standing pos configuration
        nq = self.robot_fwdk.n_q
        self.q0 = get_ergoCub_default_initial_pose(nq - 7)
        cmd = self.robot_fwdk.create_cmd_ordered_dict(self.q0[7:], np.zeros(len(self.q0[7:])), np.zeros(len(self.q0[7:])))
        self.robot_fwdk.update_system(None, None, None, None,
                             self.q0[:3], self.q0[3:7], np.zeros(3), np.zeros(3),
                                      cmd["joint_pos"], cmd["joint_vel"])

    def get_navy_door_default_initial_pose(self):
        # rotates, then translates
        pos = self.door_pos
        R = util.euler_to_rot([0., 0., np.pi / 2.])
        quat = util.rot_to_quat(R)
        return np.concatenate((pos, quat))

    def _compute_iris_regions_mgr(self, plan_to_model_frames, standing_pos, goal_step_length):
        # load obstacle, domain, and start / end seed for IRIS
        obstacles = self.obstacles
        domain_ubody = self.domain_ubody
        domain_lbody = self.domain_lbody
        # shift (feet) iris seed to get nicer IRIS region
        iris_lf_shift = np.array([0.0, 0., 0.])
        iris_rf_shift = np.array([0.0, 0., 0.])
        iris_kn_shift = np.array([0.0, 0., 0.0])
        # get end effector positions via fwd kin
        starting_torso_pos = standing_pos
        final_torso_pos = starting_torso_pos + np.array([goal_step_length, 0., 0.])
        starting_lf_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['LF'])[:3, 3]
        final_lf_pos = starting_lf_pos + np.array([goal_step_length, 0., 0.])
        starting_lh_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['LH'])[:3, 3] + np.array([0.1, 0., 0.])
        final_lh_pos = starting_lh_pos + np.array([goal_step_length, 0., 0.])
        starting_rf_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['RF'])[:3, 3]
        final_rf_pos = starting_rf_pos + np.array([goal_step_length, 0., 0.])
        starting_rh_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['RH'])[:3, 3] + np.array([0.1, 0., 0.])
        final_rh_pos = starting_rh_pos + np.array([goal_step_length, 0., 0.])

        safe_torso_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_torso_pos)
        safe_torso_end_region = IrisGeomInterface(obstacles, domain_ubody, final_torso_pos)
        safe_lf_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_lf_pos + iris_lf_shift)
        safe_lf_end_region = IrisGeomInterface(obstacles, domain_lbody, final_lf_pos)
        safe_lh_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_lh_pos)
        safe_lh_end_region = IrisGeomInterface(obstacles, domain_ubody, final_lh_pos)
        safe_rf_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_rf_pos + iris_rf_shift)
        safe_rf_end_region = IrisGeomInterface(obstacles, domain_lbody, final_rf_pos)
        safe_rh_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_rh_pos)
        safe_rh_end_region = IrisGeomInterface(obstacles, domain_ubody, final_rh_pos)
        safe_regions_mgr_dict = {'torso': IrisRegionsManager(safe_torso_start_region, safe_torso_end_region),
                                 'LF': IrisRegionsManager(safe_lf_start_region, safe_lf_end_region),
                                 'LH': IrisRegionsManager(safe_lh_start_region, safe_lh_end_region),
                                 'RF': IrisRegionsManager(safe_rf_start_region, safe_rf_end_region),
                                 'RH': IrisRegionsManager(safe_rh_start_region, safe_rh_end_region)}
        if b_use_knees:
            starting_lkn_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['L_knee'])[:3, 3] #+ np.array([0.02, 0., -0.05])
            final_lkn_pos = starting_lkn_pos + np.array([goal_step_length, 0., 0.])
            starting_rkn_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['R_knee'])[:3, 3]
            final_rkn_pos = starting_rkn_pos + np.array([goal_step_length, 0., 0.])

            safe_lk_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_lkn_pos)
            safe_lk_end_region = IrisGeomInterface(obstacles, domain_lbody, final_lkn_pos + iris_kn_shift)
            safe_rk_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_rkn_pos)
            safe_rk_end_region = IrisGeomInterface(obstacles, domain_lbody, final_rkn_pos + iris_kn_shift)

            safe_regions_mgr_dict['L_knee'] = IrisRegionsManager(safe_lk_start_region, safe_lk_end_region)
            safe_regions_mgr_dict['R_knee'] = IrisRegionsManager(safe_rk_start_region, safe_rk_end_region)

            self.starting_lkn_pos = starting_lkn_pos
            self.final_lkn_pos = final_lkn_pos
            self.starting_rkn_pos = starting_rkn_pos
            self.final_rkn_pos = final_rkn_pos

        # compute and connect IRIS from start to goal
        for _, irm in safe_regions_mgr_dict.items():
            irm.computeIris()
            irm.connectIrisSeeds()

        # save initial/final EE positions
        self.starting_torso_pos = starting_torso_pos
        self.final_torso_pos = final_torso_pos
        self.starting_lf_pos = starting_lf_pos
        self.final_lf_pos = final_lf_pos
        self.starting_lh_pos = starting_lh_pos
        self.final_lh_pos = final_lh_pos
        self.starting_rf_pos = starting_rf_pos
        self.final_rf_pos = final_rf_pos
        self.starting_rh_pos = starting_rh_pos
        self.final_rh_pos = final_rh_pos

        return safe_regions_mgr_dict

    def get_two_stage_contact_sequence(self, safe_regions_mgr_dict):
        starting_lh_pos = safe_regions_mgr_dict['LH'].iris_list[0].seed_pos
        starting_rh_pos = safe_regions_mgr_dict['RH'].iris_list[0].seed_pos
        final_lf_pos = safe_regions_mgr_dict['LF'].iris_list[1].seed_pos
        final_rf_pos = safe_regions_mgr_dict['RF'].iris_list[1].seed_pos
        intermediate_lh_pos_door = np.array([self.door_pos[0], 0.37, 0.9])
        final_torso_pos = safe_regions_mgr_dict['torso'].iris_list[1].seed_pos
        if b_use_knees:
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
        if b_use_knees:
            fixed_frames.append(['RF', 'R_knee'])   # frames that must not move
            motion_frames_seq.add_motion_frame({
                'LF': final_lf_pos,
                'L_knee': final_lkn_pos,
                # 'torso': final_torso_pos + np.array([0.2, 0., 0.]),  # testing
                'LH': starting_lh_pos + np.array([0.2, 0., 0.]),  # testing
                'RH': starting_rh_pos + np.array([0.2, 0., 0.])})  # testing
        else:
            fixed_frames.append(['RF', 'LH'])   # frames that must not move
            motion_frames_seq.add_motion_frame({
                'LF': final_lf_pos,
                'torso': final_torso_pos,  # testing
                'LH': starting_lh_pos + np.array([0.2, 0., 0.]),
                'RH': starting_rh_pos + np.array([0.2, 0., 0.])})  # testing

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
        if b_use_knees:
            fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])   # frames that must not move
            motion_frames_seq.add_motion_frame({
                                'RF': final_rf_pos,
                                'torso': final_torso_pos,
                                'R_knee': final_rkn_pos})
        else:
            fixed_frames.append(['LF', 'LH', 'RH'])   # frames that must not move
            motion_frames_seq.add_motion_frame({
                                'RF': final_rf_pos,
                                'torso': final_torso_pos})
        rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(rf_contact_over)

        # ---- Step 5: square up
        if b_use_knees:
            fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
        else:
            fixed_frames.append(['torso', 'LF', 'RF', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({})

        return fixed_frames, motion_frames_seq

    def get_five_stage_one_hand_contact_sequence(self, safe_regions_mgr_dict):
        # door position
        door_l_inner_location = np.array([0.3, 0.35, 0.9])
        door_r_inner_location = np.array([0.34, -0.35, 0.9])
        # starting and final positions
        starting_lh_pos = safe_regions_mgr_dict['LH'].iris_list[0].seed_pos
        starting_rh_pos = safe_regions_mgr_dict['RH'].iris_list[0].seed_pos
        final_lf_pos = safe_regions_mgr_dict['LF'].iris_list[1].seed_pos
        final_lkn_pos = safe_regions_mgr_dict['L_knee'].iris_list[1].seed_pos
        final_rf_pos = safe_regions_mgr_dict['RF'].iris_list[1].seed_pos
        final_torso_pos = safe_regions_mgr_dict['torso'].iris_list[1].seed_pos
        final_rkn_pos = safe_regions_mgr_dict['R_knee'].iris_list[1].seed_pos
        final_rh_pos = safe_regions_mgr_dict['RH'].iris_list[1].seed_pos
        final_lh_pos = safe_regions_mgr_dict['LH'].iris_list[1].seed_pos
        # final_rh_pos = starting_rh_pos + np.array([0.3, 0.0, 0.0])

        # initialize fixed and motion frame sets
        fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

        # ---- Step 1: L hand to frame
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'RH'])   # frames that must not move
        motion_frames_seq.add_motion_frame({'LH': door_l_inner_location})
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
                            'torso': final_torso_pos + np.array([-0.10, 0., 0.05]), # testing
                            # 'LH': starting_lh_pos + np.array([0.3, 0.0, 0.0])})
                            'RH': door_r_inner_location})
        rh_contact_inside = PlannerSurfaceContact('RH', np.array([1, 0, 0]))
        # rh_contact_inside.set_contact_breaking_velocity(np.array([1, 0., 0.]))
        motion_frames_seq.add_contact_surface(rh_contact_inside)

        # ---- Step 4: step through door with right foot
        fixed_frames.append(['LF', 'L_knee', 'RH'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'RF': final_rf_pos,
                            'torso': final_torso_pos,
                            'R_knee': final_rkn_pos,})
                            # 'RH': final_rh_pos,
                            # 'LH': starting_lh_pos + np.array([0.35, 0.0, 0.0])})
        rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(rf_contact_over)

        # ---- Step 5: square up
        # fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee'])
        motion_frames_seq.add_motion_frame({
                        'RH': final_rh_pos + np.array([-0.20, 0., 0.]),
                        'LH': final_lh_pos
        })

        return fixed_frames, motion_frames_seq

    def test_five_stage_plan_feet_knee_hands(self):
        # TODO simplify with append()
        if b_use_knees:
            frame_names = ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']
            plan_to_model_frames = {
                'torso': 'root_link',
                'LF': 'l_ankle_2',
                'RF': 'r_ankle_2',
                'L_knee': 'l_lower_leg',
                'R_knee': 'r_lower_leg',
                'LH': 'l_hand_palm',
                'RH': 'r_hand_palm',
            }
        else:
            frame_names = ['torso', 'LF', 'RF', 'LH', 'RH']
            plan_to_model_frames = {
                'torso': 'root_link',
                'LF': 'l_ankle_2',
                'RF': 'r_ankle_2',
                'LH': 'l_hand_palm',
                'RH': 'r_hand_palm',
            }
        ik_cfree_planner = IKCFreePlanner(self.robot.model, self.robot.data, plan_to_model_frames, self.q0)
        ee_halfspace_params = OrderedDict()
        reach_path = cwd + '/pnc/reachability_map/output/' + self.robot_name + '/' + self.robot_name
        for fr in frame_names:
            ee_halfspace_params[fr] = reach_path + '_' + fr + '.yaml'

        # process vision and create IRIS regions
        standing_pos = self.q0[:3]
        step_length = 0.40
        safe_regions_mgr_dict = self._compute_iris_regions_mgr(plan_to_model_frames, standing_pos, step_length)

        # visualize robot and door
        if b_visualize:
            visualizer = MeshcatVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model)

            try:
                visualizer.initViewer(open=True)
                visualizer.viewer.wait()
            except ImportError as err:
                print(
                    "Error while initializing the viewer. It seems you should install Python meshcat"
                )
                print(err)
                sys.exit(0)
            visualizer.loadViewerModel(rootNodeName=self.robot_name)
            visualizer.display(self.q0)

            # load (real) door to visualizer
            door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
                cwd + "/robot_model/ground/navy_door.urdf",
                cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

            door_vis = MeshcatVisualizer(door_model, door_collision_model, door_visual_model)
            door_vis.initViewer(visualizer.viewer)
            door_vis.loadViewerModel(rootNodeName="door")
            door_vis_q = self.get_navy_door_default_initial_pose()
            door_vis.display(door_vis_q)
        else:
            visualizer = None

        # generate all frame traversable regions
        traversable_regions_dict = OrderedDict()
        for fr in frame_names:
            if fr == 'torso':
                traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                      b_visualize_reach=b_visualize,
                                                                      b_visualize_safe=b_visualize,
                                                                      visualizer=visualizer)
            else:
                traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                      ee_halfspace_params[fr],
                                                                      b_visualize_reach=b_visualize,
                                                                      b_visualize_safe=b_visualize,
                                                                      visualizer=visualizer)
                traversable_regions_dict[fr].update_origin_pose(standing_pos)
            traversable_regions_dict[fr].load_iris_regions(safe_regions_mgr_dict[fr])
        self.assertEqual(True, True)

        # initial and desired final positions for each frame
        p_init = {}
        for fr in frame_names:
            p_init[fr] = safe_regions_mgr_dict[fr].iris_list[0].seed_pos  # starting_pos

        # hand-chosen five-stage sequence of contacts
        fixed_frames_seq, motion_frames_seq = self.get_two_stage_contact_sequence(safe_regions_mgr_dict)

        # planner parameters
        T = 3
        alpha = [0, 0, 1]
        if b_use_knees:
            traversable_regions = [traversable_regions_dict['torso'],
                                   traversable_regions_dict['LF'],
                                   traversable_regions_dict['RF'],
                                   traversable_regions_dict['L_knee'],
                                   traversable_regions_dict['R_knee'],
                                   traversable_regions_dict['LH'],
                                   traversable_regions_dict['RH']]
            frame_planner = LocomanipulationFramePlanner(traversable_regions,
                                                         aux_frames_path=self.aux_frames_path,
                                                         fixed_frames=fixed_frames_seq,
                                                         motion_frames_seq=motion_frames_seq)
        else:
            traversable_regions = [traversable_regions_dict['torso'],
                                   traversable_regions_dict['LF'],
                                   traversable_regions_dict['RF'],
                                   traversable_regions_dict['LH'],
                                   traversable_regions_dict['RH']]

            frame_planner = LocomanipulationFramePlanner(traversable_regions,
                                                     fixed_frames=fixed_frames_seq,
                                                     motion_frames_seq=motion_frames_seq)

        # set planner
        ik_cfree_planner.set_planner(frame_planner)
        ik_cfree_planner.plan(p_init, T, alpha, visualizer)

        self.assertEqual(True, True)  # add assertion here

    def test_five_stage_plan_one_hand_at_a_time(self):
        # TODO simplify with append()
        if b_use_knees:
            frame_names = ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']
            plan_to_model_frames = {
                'torso': 'root_link',
                'LF': 'l_ankle_2',
                'RF': 'r_ankle_2',
                'L_knee': 'l_lower_leg',
                'R_knee': 'r_lower_leg',
                'LH': 'l_hand_palm',
                'RH': 'r_hand_palm',
            }
        else:
            frame_names = ['torso', 'LF', 'RF', 'LH', 'RH']
            plan_to_model_frames = {
                'torso': 'root_link',
                'LF': 'l_ankle_2',
                'RF': 'r_ankle_2',
                'LH': 'l_hand_palm',
                'RH': 'r_hand_palm',
            }
        ik_cfree_planner = IKCFreePlanner(self.robot.model, self.robot.data, plan_to_model_frames, self.q0)
        ee_halfspace_params = OrderedDict()
        reach_path = cwd + '/pnc/reachability_map/output/' + self.robot_name + '/' + self.robot_name
        for fr in frame_names:
            ee_halfspace_params[fr] = reach_path + '_' + fr + '.yaml'

        # process vision and create IRIS regions
        standing_pos = self.q0[:3]
        step_length = 0.40
        safe_regions_mgr_dict = self._compute_iris_regions_mgr(plan_to_model_frames, standing_pos, step_length)

        # visualize robot and door
        if b_visualize:
            visualizer = MeshcatVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model)

            try:
                visualizer.initViewer(open=True)
                visualizer.viewer.wait()
            except ImportError as err:
                print(
                    "Error while initializing the viewer. It seems you should install Python meshcat"
                )
                print(err)
                sys.exit(0)
            visualizer.loadViewerModel(rootNodeName=self.robot_name)
            visualizer.display(self.q0)

            # load (real) door to visualizer
            door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
                cwd + "/robot_model/ground/navy_door.urdf",
                cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

            door_vis = MeshcatVisualizer(door_model, door_collision_model, door_visual_model)
            door_vis.initViewer(visualizer.viewer)
            door_vis.loadViewerModel(rootNodeName="door")
            door_vis_q = self.get_navy_door_default_initial_pose()
            door_vis.display(door_vis_q)
        else:
            visualizer = None

        # generate all frame traversable regions
        traversable_regions_dict = OrderedDict()
        for fr in frame_names:
            if fr == 'torso':
                traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                      b_visualize_reach=b_visualize,
                                                                      b_visualize_safe=b_visualize,
                                                                      visualizer=visualizer)
            else:
                traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                      ee_halfspace_params[fr],
                                                                      b_visualize_reach=b_visualize,
                                                                      b_visualize_safe=b_visualize,
                                                                      visualizer=visualizer)
                traversable_regions_dict[fr].update_origin_pose(standing_pos)
            traversable_regions_dict[fr].load_iris_regions(safe_regions_mgr_dict[fr])
        self.assertEqual(True, True)

        # initial and desired final positions for each frame
        p_init = {}
        for fr in frame_names:
            p_init[fr] = safe_regions_mgr_dict[fr].iris_list[0].seed_pos  # starting_pos

        # hand-chosen five-stage sequence of contacts
        fixed_frames_seq, motion_frames_seq = self.get_five_stage_one_hand_contact_sequence(safe_regions_mgr_dict)

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
                                                     aux_frames_path=self.aux_frames_path,
                                                     fixed_frames=fixed_frames_seq,
                                                     motion_frames_seq=motion_frames_seq)

        # set planner
        ik_cfree_planner.set_planner(frame_planner)
        ik_cfree_planner.plan(p_init, T, alpha, visualizer)

        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
