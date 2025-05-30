import unittest

import os
import sys
from collections import OrderedDict

from pnc.data_saver import DataSaver
from util.environment_creator import TiltedStairs
from ..self_collision_avoidance.sca_robot_geometry import SCARobotGeometry
import pnc.planner.multicontact.contact_sequence_plans.tilted_stairs_plans as stairs_plan

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from pydrake.geometry.optimization import HPolyhedron

from ..frame_traversable_region import FrameTraversableRegion
from ..ik_cfree_planner import IKCFreePlanner
from ..locomanipulation_frame_planner import LocomanipulationFramePlanner
from ..planner_surface_contact import MotionFrameSequencer, PlannerSurfaceContact, \
    get_contact_seq_from_fixed_frames_seq, get_contact_planes_from_motion_frames_seq
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
from util import util
from vision.iris.iris_geom_interface import IrisGeomInterface
from vision.iris.iris_regions_manager import IrisRegionsManager
import plot.meshcat_utils as vis_tools

b_visualize = False


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

    floating_base = np.array([0., 0., 0.73, 0., 0., 0., 1.])
    return np.concatenate((floating_base, q0))


class TestIKCFreePlanner(unittest.TestCase):

    def setUp(self):
        robot_name = 'g1'
        self.b_use_knees = True
        self.robot_name = robot_name
        self.frame_names, self.plan_to_model_frames = self.get_robot_link_names()
        self.aux_frames_path = None
        if self.b_use_knees:
            self.aux_frames_path = cwd + '/pnc/reachability_map/output/' + robot_name + '/' + \
                                robot_name + '_aux_frames.yaml'

        # create tilted stairs environment
        self.stairs = TiltedStairs()

        # load robot
        mesh_dir = cwd + "/robot_model/" + robot_name + "_description"
        self.robot = pin.RobotWrapper.BuildFromURDF(
            mesh_dir + "/g1_29dof_lock_waist_modified.urdf",
            mesh_dir,
            root_joint=pin.JointModelFreeFlyer())

        # set-up easy access to fwd kinematics for IRIS seeds
        self.robot_fwdk = PinocchioRobotSystem(
            mesh_dir + "/g1_29dof_lock_waist_modified.urdf",
            mesh_dir, False, False)

        # load default standing pos configuration
        nq = self.robot_fwdk.n_q
        self.q0 = get_g1_default_initial_pose(nq - 7)
        cmd = self.robot_fwdk.create_cmd_ordered_dict(self.q0[7:], np.zeros(len(self.q0[7:])), np.zeros(len(self.q0[7:])))
        self.robot_fwdk.update_system(None, None, None, None,
                             self.q0[:3], self.q0[3:7], np.zeros(3), np.zeros(3),
                                      cmd["joint_pos"], cmd["joint_vel"])

        self.starting_pose = {}
        for fr in self.plan_to_model_frames.keys():
            self.starting_pose[fr] = self.robot_fwdk.get_link_iso(self.plan_to_model_frames[fr])[:3, 3]

        # load robot model and corresponding robot data for self-collision avoidance
        self.package_dir = cwd + "/robot_model/g1_description"
        # self.robot_urdf_file = self.package_dir + "/g1_cube_collisions.urdf"
        self.robot_urdf_file = self.package_dir + "/g1_29dof_lock_waist_modified.urdf"

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
                'LH': 'left_rubber_hand',
                'RH': 'right_rubber_hand'
            }
        else:
            frame_names = ['torso', 'LF', 'RF', 'LH', 'RH']
            plan_to_model_frames = {
                'torso': 'torso_link',
                'LF': 'left_ankle_roll_link',
                'RF': 'right_ankle_roll_link',
                'LH': 'left_rubber_hand',
                'RH': 'right_rubber_hand'
            }
        return frame_names, plan_to_model_frames

    def _compute_iris_regions_mgr(self, plan_to_model_frames, standing_pos, goal_step_length):
        # load obstacle, domain, and start / end seed for IRIS
        obstacles = self.stairs.obstacles
        domain = self.stairs.domain
        # domain = self.domain
        step_height = 0.45 / 2
        # shift (feet) iris seed to get nicer IRIS region
        iris_lf_shift = np.array([0.0, 0., 0.])
        iris_rf_shift = np.array([0.0, 0., 0.])
        iris_kn_shift = np.array([0.0, 0., 0.0])
        # get end effector positions via fwd kin
        starting_torso_pos = standing_pos
        final_torso_pos = starting_torso_pos + np.array([goal_step_length, 0., step_height])
        starting_lf_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['LF'])[:3, 3]
        final_lf_pos = starting_lf_pos + np.array([goal_step_length, 0., step_height])
        starting_lh_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['LH'])[:3, 3] #+ np.array([0.1, 0., 0.])
        final_lh_pos = starting_lh_pos + np.array([goal_step_length, 0., step_height])
        starting_rf_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['RF'])[:3, 3]
        final_rf_pos = starting_rf_pos + np.array([goal_step_length, 0., step_height])
        starting_rh_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['RH'])[:3, 3] #+ np.array([0.1, 0., 0.])
        final_rh_pos = starting_rh_pos + np.array([goal_step_length, 0., step_height])

        safe_torso_start_region = IrisGeomInterface(obstacles, domain, starting_torso_pos)
        safe_torso_end_region = IrisGeomInterface(obstacles, domain, final_torso_pos)
        safe_lf_start_region = IrisGeomInterface(obstacles, domain, starting_lf_pos + iris_lf_shift)
        safe_lf_end_region = IrisGeomInterface(obstacles, domain, final_lf_pos)
        safe_lh_start_region = IrisGeomInterface(obstacles, domain, starting_lh_pos)
        safe_lh_end_region = IrisGeomInterface(obstacles, domain, final_lh_pos)
        safe_rf_start_region = IrisGeomInterface(obstacles, domain, starting_rf_pos + iris_rf_shift)
        safe_rf_end_region = IrisGeomInterface(obstacles, domain, final_rf_pos)
        safe_rh_start_region = IrisGeomInterface(obstacles, domain, starting_rh_pos)
        safe_rh_end_region = IrisGeomInterface(obstacles, domain, final_rh_pos)
        safe_regions_mgr_dict = {'torso': IrisRegionsManager(safe_torso_start_region, safe_torso_end_region),
                                 'LF': IrisRegionsManager(safe_lf_start_region, safe_lf_end_region),
                                 'LH': IrisRegionsManager(safe_lh_start_region, safe_lh_end_region),
                                 'RF': IrisRegionsManager(safe_rf_start_region, safe_rf_end_region),
                                 'RH': IrisRegionsManager(safe_rh_start_region, safe_rh_end_region)}
        if self.b_use_knees:
            starting_lkn_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['L_knee'])[:3, 3]
            final_lkn_pos = starting_lkn_pos + np.array([goal_step_length, 0., step_height])
            starting_rkn_pos = self.robot_fwdk.get_link_iso(plan_to_model_frames['R_knee'])[:3, 3]
            final_rkn_pos = starting_rkn_pos + np.array([goal_step_length, 0., step_height])

            safe_lk_start_region = IrisGeomInterface(obstacles, domain, starting_lkn_pos + np.array([0.02, 0., -0.05]))
            safe_lk_end_region = IrisGeomInterface(obstacles, domain, final_lkn_pos + iris_kn_shift)
            safe_rk_start_region = IrisGeomInterface(obstacles, domain, starting_rkn_pos)
            safe_rk_end_region = IrisGeomInterface(obstacles, domain, final_rkn_pos + iris_kn_shift)

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

    def test_stairs_plan_one_hand_at_a_time(self, sca_geometry=None):
        frame_names = self.frame_names
        plan_to_model_frames = self.plan_to_model_frames

        ik_cfree_planner = IKCFreePlanner(self.robot.model, self.robot.data, plan_to_model_frames, self.q0)
        ee_halfspace_params = OrderedDict()
        for fr in frame_names:
            ee_halfspace_params[fr] = cwd + '/pnc/reachability_map/output/g1/g1_' + fr + '.yaml'

        # hand-chosen five-stage sequence of contacts
        fixed_frames_seq, motion_frames_seq = stairs_plan.get_opposing_limbs_contact_sequence(self.stairs, self.starting_pose, self.b_use_knees)

        # process vision and create IRIS regions
        standing_pos = self.q0[:3]
        safe_regions_mgr_dict = stairs_plan.compute_stairs_iris_regions_mgr(self.stairs, self.starting_pose, motion_frames_seq)

        # visualize robot and stairs
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
        p_init['torso'] = self.starting_pose['torso']
        p_init['LF'] = self.starting_pose['LF']
        p_init['RF'] = self.starting_pose['RF']
        if self.b_use_knees:
            p_init['L_knee'] = self.starting_pose['L_knee']
            p_init['R_knee'] = self.starting_pose['R_knee']
        p_init['LH'] = self.starting_pose['LH']
        p_init['RH'] = self.starting_pose['RH']

        contact_seqs = get_contact_seq_from_fixed_frames_seq(fixed_frames_seq)
        contact_planes = get_contact_planes_from_motion_frames_seq(contact_seqs, motion_frames_seq)

        # planner parameters
        T = 3
        alpha = [1, 0.1, 0.01]
        if self.b_use_knees:
            traversable_regions = [traversable_regions_dict['torso'],
                                   traversable_regions_dict['LF'],
                                   traversable_regions_dict['RF'],
                                   traversable_regions_dict['L_knee'],
                                   traversable_regions_dict['R_knee'],
                                   traversable_regions_dict['LH'],
                                   traversable_regions_dict['RH']]
        else:
            traversable_regions = [traversable_regions_dict['torso'],
                                   traversable_regions_dict['LF'],
                                   traversable_regions_dict['RF'],
                                   traversable_regions_dict['LH'],
                                   traversable_regions_dict['RH']]
        frame_planner = LocomanipulationFramePlanner(traversable_regions,
                                                     aux_frames_path=self.aux_frames_path,
                                                     fixed_frames=fixed_frames_seq,
                                                     motion_frames_seq=motion_frames_seq,
                                                     sca_robot_geom=sca_geometry)

        # set planner
        weights_rigid_link = np.array([1., 0., 10.])
        ik_cfree_planner.set_planner(frame_planner)
        ik_cfree_planner.set_plan_to_model_frames(plan_to_model_frames)
        ik_cfree_planner.plan(p_init, T, alpha, weights_rigid_link, visualizer)

        self.assertEqual(True, True)
        return ik_cfree_planner

    def test_self_collision_avoidance(self):
        ik_cfree_planner = self.test_stairs_plan_one_hand_at_a_time()

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
            lh_targets[idx] = targets_dict['LH']
            rh_targets[idx] = targets_dict['RH']
            if self.b_use_knees:
                lkn_targets[idx] = targets_dict['L_knee']
                rkn_targets[idx] = targets_dict['R_knee']
            idx += 1

        if b_visualize:
            package_dir = self.package_dir
            robot_urdf_file = self.robot_urdf_file
            rob_model, col_model, vis_model, rob_data, col_data, vis_data = self.load_robot_model(package_dir, robot_urdf_file)

            save_freq = 1
            display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                                                          rob_data, vis_data, col_data,
                                                          ctrl_freq=N_knots / (n_contacts * T), save_freq=save_freq)
            # load stair environment to visualizer
            display.add_shapes_from(self.stairs.obstacles)

            # start animation
            display.start_animation()
            for i in range(N_knots):
                display.animate_single_collision(ik_cfree_planner.plan_to_model_frames['torso'] + '_0', base_targets[i])
                # display.animate_single_collision(ik_cfree_planner.plan_to_model_frames['L_knee'] + '_0', lkn_targets[i])
                # display.animate_single_collision(ik_cfree_planner.plan_to_model_frames['R_knee'] + '_0', rkn_targets[i])
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
        b_visualize = False
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
            save_filename = self.robot_name + 'sca_five_stage_plan_box_sphere_clean.pkl'
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
            data_saver.advance()
            data_saver.close()

        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
