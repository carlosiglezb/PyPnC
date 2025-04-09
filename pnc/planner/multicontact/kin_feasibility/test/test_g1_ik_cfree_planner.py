import unittest

import os
import sys
from collections import OrderedDict

from pnc.data_saver import DataSaver
from ..self_collision_avoidance.sca_robot_geometry import SCARobotGeometry

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from pydrake.geometry.optimization import HPolyhedron

from ..frame_traversable_region import FrameTraversableRegion
from ..ik_cfree_planner import IKCFreePlanner
from ..locomanipulation_frame_planner import LocomanipulationFramePlanner
from ..planner_surface_contact import MotionFrameSequencer, PlannerSurfaceContact
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
from util import util
from vision.iris.iris_geom_interface import IrisGeomInterface
from vision.iris.iris_regions_manager import IrisRegionsManager
import plot.meshcat_utils as vis_tools

b_visualize = True
b_use_knees = True


def get_g1_default_initial_pose(n_joints):
    q0 = np.zeros(n_joints, )
    hip_yaw_angle = 5
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
    # q0[12] = 0.  # neck pitch
    # q0[13] = 0.  # r_hip_ie
    # q0[14] = np.radians(-hip_yaw_angle)  # r_hip_aa
    # q0[15] = -np.pi / 4  # r_hip_fe
    # q0[16] = np.pi / 4  # r_knee_fe_jp
    # q0[17] = np.pi / 4  # r_knee_fe_jd
    # q0[18] = -np.pi / 4  # r_ankle_fe
    # q0[19] = np.radians(hip_yaw_angle)  # r_ankle_ie
    # q0[20] = 0.  # r_shoulder_fe
    # q0[21] = -np.pi / 6  # r_shoulder_aa
    # q0[22] = 0.  # r_shoulder_ie
    # q0[23] = -np.pi / 2  # r_elbow_fe
    # q0[24] = np.pi/3.   # r_wrist_ps
    # q0[25] = 0.  # r_wrist_pitch

    floating_base = np.array([0., 0., 0.62, 0., 0., 0., 1.])
    return np.concatenate((floating_base, q0))


class TestIKCFreePlanner(unittest.TestCase):

    def setUp(self):
        robot_name = 'g1'
        self.b_use_knees = True
        self.robot_name = robot_name
        self.frame_names, self.plan_to_model_frames = self.get_robot_link_names()
        self.aux_frames_path = cwd + '/pnc/reachability_map/output/' + robot_name + '/' + \
                                robot_name + '_aux_frames.yaml'

        # create navy door environment
        self.door_pos = np.array([0.28, 0., 0.])
        door_width = np.array([0.025, 0., 0.])
        dom_ubody_lb = np.array([-1.6, -0.8, 0.5])
        dom_ubody_ub = np.array([1.6, 0.8, 2.1])
        dom_lbody_lb = np.array([-1.6, -0.8, -0.])
        dom_lbody_ub = np.array([1.6, 0.8, 1.1])
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
        mesh_dir = cwd + "/robot_model/" + robot_name + "_description"
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
        self.q0 = get_g1_default_initial_pose(nq - 7)
        cmd = self.robot_fwdk.create_cmd_ordered_dict(self.q0[7:], np.zeros(len(self.q0[7:])), np.zeros(len(self.q0[7:])))
        self.robot_fwdk.update_system(None, None, None, None,
                             self.q0[:3], self.q0[3:7], np.zeros(3), np.zeros(3),
                                      cmd["joint_pos"], cmd["joint_vel"])

        # load robot model and corresponding robot data for self-collision avoidance
        self.package_dir = cwd + "/robot_model/g1_description"
        # self.robot_urdf_file = self.package_dir + "/g1_cube_collisions.urdf"
        self.robot_urdf_file = self.package_dir + "/g1_cube_sphere_collisions.urdf"

    # needed for self-collision checks
    @staticmethod
    def load_robot_model(package_dir, robot_urdf_file):
        rob_model, col_model, vis_model = pin.buildModelsFromUrdf(robot_urdf_file,
                                                                  package_dir,
                                                                  pin.JointModelFreeFlyer())
        rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)

        return rob_model, col_model, vis_model, rob_data, col_data, vis_data

    def get_robot_link_names(self):
        if self.b_use_knees:
            frame_names = ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']
            plan_to_model_frames = {
                'torso': 'torso_link',
                'LF': 'left_ankle_roll_link',
                'RF': 'right_ankle_roll_link',
                'L_knee': 'left_knee_link',
                'R_knee': 'right_knee_link',
                'LH': 'left_palm_link',
                'RH': 'right_palm_link'
            }
        else:
            frame_names = ['torso', 'LF', 'RF', 'LH', 'RH']
            plan_to_model_frames = {
                'torso': 'torso_link',
                'LF': 'left_ankle_roll_link',
                'RF': 'right_ankle_roll_link',
                'LH': 'left_palm_link',
                'RH': 'right_palm_link'
            }
        return frame_names, plan_to_model_frames

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
        iris_lf_shift = np.array([0.2, 0., 0.])     # useful in motion stepping on knee-knocker
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
        final_rh_pos = starting_rh_pos + np.array([goal_step_length, 0., 0.])   #+ 0.15 on x

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
        if self.b_use_knees:
            starting_lkn_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['L_knee'])[:3, 3]
            final_lkn_pos = starting_lkn_pos + np.array([goal_step_length, 0., 0.])
            starting_rkn_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['R_knee'])[:3, 3]
            final_rkn_pos = starting_rkn_pos + np.array([goal_step_length, 0., 0.])

            safe_lk_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_lkn_pos + + np.array([0.02, 0., -0.05]))
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

    def get_five_stage_contact_sequence(self, safe_regions_mgr_dict):
        starting_lh_pos = safe_regions_mgr_dict['LH'].iris_list[0].seed_pos
        starting_rh_pos = safe_regions_mgr_dict['RH'].iris_list[0].seed_pos
        final_lf_pos = safe_regions_mgr_dict['LF'].iris_list[1].seed_pos
        final_rf_pos = safe_regions_mgr_dict['RF'].iris_list[1].seed_pos
        intermediate_lh_pos_door = np.array([self.door_pos[0], 0.37, 0.9])
        final_torso_pos = safe_regions_mgr_dict['torso'].iris_list[1].seed_pos
        if self.b_use_knees:
            final_lkn_pos = safe_regions_mgr_dict['L_knee'].iris_list[1].seed_pos
            final_rkn_pos = safe_regions_mgr_dict['R_knee'].iris_list[1].seed_pos

        # initialize fixed and motion frame sets
        fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

        # ---- Step 1: L hand to frame
        if self.b_use_knees:
            fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
        else:
            fixed_frames.append(['LF', 'RF'])   # frames that must not move
        motion_frames_seq.add_motion_frame({'LH': intermediate_lh_pos_door})
        lh_contact_front = PlannerSurfaceContact('LH', np.array([-1, 0, 0]))
        lh_contact_front.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
        motion_frames_seq.add_contact_surface(lh_contact_front)

        # ---- Step 2: step through door with left foot
        if self.b_use_knees:
            fixed_frames.append(['RF', 'R_knee', 'LH'])   # frames that must not move
            motion_frames_seq.add_motion_frame({
                'LF': final_lf_pos,
                'L_knee': final_lkn_pos,
                'torso': final_torso_pos,  # testing
                'RH': starting_rh_pos + np.array([0.2, 0., 0.])})  # testing
        else:
            fixed_frames.append(['RF', 'LH'])   # frames that must not move
            motion_frames_seq.add_motion_frame({
                'LF': final_lf_pos,
                'torso': final_torso_pos,  # testing
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
        # fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])   # frames that must not move
        # motion_frames_seq.add_motion_frame({
        #                     'RF': final_rf_pos,
        #                     'torso': final_torso_pos,
        #                     'R_knee': final_rkn_pos})
        # rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
        # motion_frames_seq.add_contact_surface(rf_contact_over)

        # ---- Step 5: square up
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({})

        return fixed_frames, motion_frames_seq

    def get_five_stage_one_hand_contact_sequence(self, safe_regions_mgr_dict):
        starting_lh_pos = safe_regions_mgr_dict['LH'].iris_list[0].seed_pos
        starting_rh_pos = safe_regions_mgr_dict['RH'].iris_list[0].seed_pos
        final_lf_pos = safe_regions_mgr_dict['LF'].iris_list[1].seed_pos
        final_lkn_pos = safe_regions_mgr_dict['L_knee'].iris_list[1].seed_pos
        final_rf_pos = safe_regions_mgr_dict['RF'].iris_list[1].seed_pos
        final_torso_pos = safe_regions_mgr_dict['torso'].iris_list[1].seed_pos
        final_rkn_pos = safe_regions_mgr_dict['R_knee'].iris_list[1].seed_pos
        final_rh_pos = safe_regions_mgr_dict['RH'].iris_list[1].seed_pos
        # final_rh_pos = starting_rh_pos + np.array([0.3, 0.0, 0.0])

        # initialize fixed and motion frame sets
        fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

        # ---- Step 1: L hand to frame
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'RH'])   # frames that must not move
        motion_frames_seq.add_motion_frame({'LH': starting_lh_pos + np.array([0.08, 0.07, 0.15])})
        lh_contact_front = PlannerSurfaceContact('LH', np.array([-1, 0, 0]))
        lh_contact_front.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
        motion_frames_seq.add_contact_surface(lh_contact_front)

        # ---- Step 2: step through door with left foot
        fixed_frames.append(['RF', 'R_knee', 'LH'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'LF': final_lf_pos,
                            'L_knee': final_lkn_pos})
        lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(lf_contact_over)

        # ---- Step 3: re-position L/R hands for more stability
        fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'LH': starting_lh_pos + np.array([0.3, 0.0, 0.0])})
                            # 'RH': starting_rh_pos + np.array([0.08, -0.07, 0.15])})
        lh_contact_inside = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
        lh_contact_inside.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
        # rh_contact_inside = PlannerSurfaceContact('RH', np.array([0, 1, 0]))
        motion_frames_seq.add_contact_surface([lh_contact_inside])

        # ---- Step 4: step through door with right foot
        fixed_frames.append(['LF', 'L_knee'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'RF': final_rf_pos,
                            'torso': final_torso_pos,
                            'R_knee': final_rkn_pos,
                            'RH': final_rh_pos,
                            'LH': starting_lh_pos + np.array([0.55, 0.0, 0.0])})
        rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(rf_contact_over)

        # ---- Step 5: square up
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({})

        return fixed_frames, motion_frames_seq

    def get_five_stage_on_knocker_contact_sequence(self, safe_regions_mgr_dict):
        ###### Previously used key locations
        # door_l_outer_location = np.array([0.45, 0.35, 1.2])
        # door_r_outer_location = np.array([0.45, -0.35, 1.2])
        # G1 settings
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
        fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])  # frames that must not move
        motion_frames_seq.add_motion_frame({
            'LH': door_l_inner_location,
            # 'torso': starting_torso_pos + np.array([0.07, -0.07, 0])
        })
        lh_contact_front = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
        lh_contact_front.set_contact_breaking_velocity(np.array([0, -1, 0.]))
        motion_frames_seq.add_contact_surface(lh_contact_front)

        # ---- Step 2: step on knee-knocker with right foot
        fixed_frames.append(['LF', 'L_knee', 'LH'])  # frames that must not move
        motion_frames_seq.add_motion_frame({
            'RF': intermediate_rf_pos,
            'R_knee': intermediate_rf_pos + (final_rkn_pos - final_rf_pos) + np.array([-0.05, 0., 0.035])})
        rf_contact_knocker = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
        rf_contact_knocker.set_contact_breaking_velocity(np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(rf_contact_knocker)

        # ---- Step 3: step through door with left foot
        fixed_frames.append(['RF', 'R_knee', 'LH'])  # frames that must not move
        motion_frames_seq.add_motion_frame({
            # 'LH': starting_lh_pos + np.array([0.3, 0., 0.0]),   # <-- G1
            # 'LH': starting_lh_pos + np.array([0.35, 0.1, 0.0]),   # <-- other
            # 'torso': final_torso_pos + np.array([-0.15, 0.05, -0.05]),     # good testing
            'L_knee': final_lkn_pos + np.array([-0.05, 0., 0.035]),
            'LF': final_lf_pos})
        lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(lf_contact_over)

        # ---- Step 4: square up
        # fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
        fixed_frames.append(['LF', 'L_knee'])
        motion_frames_seq.add_motion_frame({
            'torso': final_torso_pos,
            'RF': final_rf_pos,
            'R_knee': final_rkn_pos + np.array([-0.05, 0., 0.035]),
            'RH': final_rh_pos,  # + np.array([-0.20, 0., 0.]),
            'LH': final_lh_pos
        })
        lf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(lf_contact_over)

        # ---- Step 5: balance
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({})

        return fixed_frames, motion_frames_seq

    def test_five_stage_plan_feet_knee_hands(self):
        frame_names = self.frame_names
        plan_to_model_frames = self.plan_to_model_frames

        ik_cfree_planner = IKCFreePlanner(self.robot.model, self.robot.data, plan_to_model_frames, self.q0)
        ee_halfspace_params = OrderedDict()
        reach_path = cwd + '/pnc/reachability_map/output/' + self.robot_name + '/' + self.robot_name
        for fr in frame_names:
            ee_halfspace_params[fr] = reach_path + '_' + fr + '.yaml'

        # process vision and create IRIS regions
        standing_pos = self.q0[:3]
        step_length = 0.4
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
        fixed_frames_seq, motion_frames_seq = self.get_five_stage_contact_sequence(safe_regions_mgr_dict)

        # planner parameters
        T = 3
        alpha = [0, 0, 1]
        if self.b_use_knees:
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
        weights_rigid_link = np.array([3500., 0., 0.])
        ik_cfree_planner.set_planner(frame_planner)
        ik_cfree_planner.plan(p_init, T, alpha, weights_rigid_link, visualizer)

        self.assertEqual(True, True)

    def test_five_stage_plan_one_hand_at_a_time(self, sca_geometry=None):
        frame_names = self.frame_names
        plan_to_model_frames = self.plan_to_model_frames

        ik_cfree_planner = IKCFreePlanner(self.robot.model, self.robot.data, plan_to_model_frames, self.q0)
        ee_halfspace_params = OrderedDict()
        for fr in frame_names:
            ee_halfspace_params[fr] = cwd + '/pnc/reachability_map/output/g1/g1_' + fr + '.yaml'

        # process vision and create IRIS regions
        standing_pos = self.q0[:3]
        step_length = 0.37
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
            visualizer.loadViewerModel(rootNodeName="g1")
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
        # for fr in frame_names:
        #     p_init[fr] = safe_regions_mgr_dict[fr].iris_list[0].seed_pos  # starting_pos
        p_init['torso'] = self.starting_torso_pos
        p_init['LF'] = self.starting_lf_pos
        p_init['RF'] = self.starting_rf_pos
        if self.b_use_knees:
            p_init['L_knee'] = self.starting_lkn_pos
            p_init['R_knee'] = self.starting_rkn_pos
        p_init['LH'] = self.starting_lh_pos
        p_init['RH'] = self.starting_rh_pos

        # hand-chosen five-stage sequence of contacts
        # fixed_frames_seq, motion_frames_seq = self.get_five_stage_one_hand_contact_sequence(safe_regions_mgr_dict)
        fixed_frames_seq, motion_frames_seq = self.get_five_stage_on_knocker_contact_sequence(safe_regions_mgr_dict)

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
                                                     motion_frames_seq=motion_frames_seq,
                                                     sca_robot_geom=sca_geometry)

        # set planner
        weights_rigid_link = np.array([10, 0., 0.])
        ik_cfree_planner.set_planner(frame_planner)
        ik_cfree_planner.set_plan_to_model_frames(plan_to_model_frames)
        ik_cfree_planner.plan(p_init, T, alpha, weights_rigid_link, visualizer)

        self.assertEqual(True, True)
        return ik_cfree_planner

    def test_self_collision_avoidance(self):
        b_visualize = True
        ik_cfree_planner = self.test_five_stage_plan_one_hand_at_a_time()

        # populate trajectories
        N_knots = 100
        base_targets = np.zeros((N_knots, 3))
        lf_targets = np.zeros((N_knots, 3))
        rf_targets = np.zeros((N_knots, 3))
        lkn_targets = np.zeros((N_knots, 3))
        rkn_targets = np.zeros((N_knots, 3))
        lh_targets = np.zeros((N_knots, 3))
        rh_targets = np.zeros((N_knots, 3))
        n_contacts = len(ik_cfree_planner.planner.fixed_frames)
        T = 3
        idx = 0
        for t in np.linspace(0, n_contacts * T, N_knots):
            targets_dict = ik_cfree_planner.pack_current_targets(t)
            base_targets[idx] = targets_dict['torso']
            lf_targets[idx] = targets_dict['LF']
            rf_targets[idx] = targets_dict['RF']
            lkn_targets[idx] = targets_dict['L_knee']
            rkn_targets[idx] = targets_dict['R_knee']
            lh_targets[idx] = targets_dict['LH']
            rh_targets[idx] = targets_dict['RH']
            idx += 1

        if b_visualize:
            package_dir = self.package_dir
            robot_urdf_file = self.robot_urdf_file
            rob_model, col_model, vis_model, rob_data, col_data, vis_data = self.load_robot_model(package_dir, robot_urdf_file)

            save_freq = 1
            display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                                                          rob_data, vis_data, col_data,
                                                          ctrl_freq=N_knots / (n_contacts * T), save_freq=save_freq)
            # load (real) door to visualizer
            door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
                cwd + "/robot_model/ground/navy_door.urdf",
                cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

            door_vis_q = self.get_navy_door_default_initial_pose()
            display.add_robot("door", door_model, door_collision_model, door_visual_model, door_vis_q[:3], door_vis_q[3:])

            # start animation
            display.start_animation()
            for i in range(N_knots):
                display.animate_single_collision(ik_cfree_planner.plan_to_model_frames['torso'] + '_0', base_targets[i])
                display.animate_target("lfoot_target", [lf_targets[i]], [1, 1, 0])
                display.animate_target("lknee_target", [lkn_targets[i]], [0, 0, 1])
                display.animate_target("rfoot_target", [rf_targets[i]], [1, 1, 0])
                display.animate_target("rknee_target", [rkn_targets[i]], [0, 0, 1])
                display.animate_target("lhand_target", [lh_targets[i]], [0.5, 0, 0])
                display.animate_target("rhand_target", [rh_targets[i]], [0.5, 0, 0])
                display.animate_target("base_target", [base_targets[i]], [0, 0.5, 0])
                display.animation_step()
            display.finish_animation()

        self.assertEqual(True, True)


    def test_sca_plan_five_stage_plan_one_hand_at_a_time(self):
        b_visualize = True
        b_save_plan = True

        plan_to_model_frames = self.plan_to_model_frames
        sca_geometry = SCARobotGeometry(self.package_dir, self.robot_urdf_file, plan_to_model_frames)
        sca_kin_cfree_planner = self.test_five_stage_plan_one_hand_at_a_time(sca_geometry)

        # populate trajectories
        N_knots = 100
        base_targets = np.zeros((N_knots, 3))
        lf_targets = np.zeros((N_knots, 3))
        rf_targets = np.zeros((N_knots, 3))
        lkn_targets = np.zeros((N_knots, 3))
        rkn_targets = np.zeros((N_knots, 3))
        lh_targets = np.zeros((N_knots, 3))
        rh_targets = np.zeros((N_knots, 3))
        n_contacts = len(sca_kin_cfree_planner.planner.fixed_frames)
        T = 3
        idx = 0
        for t in np.linspace(0, n_contacts * T, N_knots):
            targets_dict = sca_kin_cfree_planner.pack_current_targets(t)
            base_targets[idx] = targets_dict['torso']
            lf_targets[idx] = targets_dict['LF']
            rf_targets[idx] = targets_dict['RF']
            lkn_targets[idx] = targets_dict['L_knee']
            rkn_targets[idx] = targets_dict['R_knee']
            lh_targets[idx] = targets_dict['LH']
            rh_targets[idx] = targets_dict['RH']
            idx += 1

        if b_visualize:
            package_dir = self.package_dir
            robot_urdf_file = self.robot_urdf_file
            rob_model, col_model, vis_model, rob_data, col_data, vis_data = self.load_robot_model(package_dir, robot_urdf_file)

            save_freq = 1
            display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                                                          rob_data, vis_data, col_data,
                                                          ctrl_freq=N_knots / (n_contacts * T), save_freq=save_freq)
            # load (real) door to visualizer
            door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
                cwd + "/robot_model/ground/navy_door.urdf",
                cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

            door_vis_q = self.get_navy_door_default_initial_pose()
            display.add_robot("door", door_model, door_collision_model, door_visual_model, door_vis_q[:3], door_vis_q[3:])

            # start animation
            display.start_animation()
            for i in range(N_knots):
                display.animate_single_collision(sca_kin_cfree_planner.plan_to_model_frames['torso'] + '_0', base_targets[i])
                display.animate_target("lfoot_target", [lf_targets[i]], [1, 1, 0])
                display.animate_target("lknee_target", [lkn_targets[i]], [0, 0, 1])
                display.animate_target("rfoot_target", [rf_targets[i]], [1, 1, 0])
                display.animate_target("rknee_target", [rkn_targets[i]], [0, 0, 1])
                display.animate_target("lhand_target", [lh_targets[i]], [0.5, 0, 0])
                display.animate_target("rhand_target", [rh_targets[i]], [0.5, 0, 0])
                display.animate_target("base_target", [base_targets[i]], [0, 0.5, 0])
                display.animation_step()
            display.finish_animation()

        if b_save_plan:
            # save the solution parameters needed to reconstruct the Bezier curves
            save_filename = self.robot_name + '_sca_five_stage_plan_box_sphere_clean.pkl'
            transition_times = []
            n_frames = len(sca_kin_cfree_planner.planner.path)
            data_saver = DataSaver(save_filename)
            data_saver.add('bez_points', sca_kin_cfree_planner.planner.points)
            for i in range(n_frames):
                transition_times.append(sca_kin_cfree_planner.planner.path[i].transition_times)
            data_saver.add('bez_points_transition_times', transition_times)
            data_saver.add('n_frames', n_frames)
            data_saver.add('n_iris_traversed_per_frame', len(sca_kin_cfree_planner.planner.path[0].beziers))
            data_saver.add('bez_path', sca_kin_cfree_planner.planner.path)
            data_saver.add('fixed_frames', sca_kin_cfree_planner.planner.fixed_frames)
            data_saver.advance()
            data_saver.close()

        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
