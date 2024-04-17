import unittest

import os
import sys
import numpy as np
import scipy as sp
import meshcat
import pinocchio as pin
import util.util
from collections import OrderedDict
from pinocchio.visualize import MeshcatVisualizer

from pnc.planner.multicontact.frame_traversable_region import FrameTraversableRegion
from pnc.planner.multicontact.locomanipulation_frame_planner import LocomanipulationFramePlanner
from pnc.planner.multicontact.planner_surface_contact import MotionFrameSequencer, PlannerSurfaceContact
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
# IRIS
from vision.iris.iris_geom_interface import *
from vision.iris.iris_regions_manager import IrisRegionsManager

cwd = os.getcwd()
sys.path.append(cwd)

b_visualize = True


class TestLocomanipulationFramePlanner(unittest.TestCase):
    def setUp(self):
        self.torso_frame_name = 'torso'
        self.ee_offsets_path = cwd + '/pnc/reachability_map/output/draco3_ee_offsets.yaml'
        self.aux_frames_path = cwd + '/pnc/reachability_map/output/draco3_aux_frames.yaml'

        # create navy door environment
        self.door_pos = np.array([0.3, 0., 0.])
        door_width = np.array([0.025, 0., 0.])
        dom_ubody_lb = np.array([-1.6, -0.8, 0.5])
        dom_ubody_ub = np.array([1.6, 0.8, 2.1])
        dom_lbody_lb = np.array([-1.6, -0.8, -0.])
        dom_lbody_ub = np.array([1.6, 0.8, 1.1])
        floor = mut.HPolyhedron.MakeBox(
                                np.array([-2, -0.9, -0.05]) + self.door_pos + door_width,
                                np.array([2, 0.9, -0.001]) + self.door_pos + door_width)
        knee_knocker_base = mut.HPolyhedron.MakeBox(
                                    np.array([-0.025, -0.9, 0.0]) + self.door_pos + door_width,
                                    np.array([0.025, 0.9, 0.4]) + self.door_pos + door_width)
        knee_knocker_lwall = mut.HPolyhedron.MakeBox(
                                    np.array([-0.025, 0.9-0.518, 0.0]) + self.door_pos + door_width,
                                    np.array([0.025, 0.9, 2.2]) + self.door_pos + door_width)
        knee_knocker_rwall = mut.HPolyhedron.MakeBox(
                                    np.array([-0.025, -0.9, 0.0]) + self.door_pos + door_width,
                                    np.array([0.025, -(0.9-0.518), 2.2]) + self.door_pos + door_width)
        knee_knocker_top = mut.HPolyhedron.MakeBox(
                                    np.array([-0.025, -0.9, 1.85]) + self.door_pos + door_width,
                                    np.array([0.025, 0.9, 2.25]) + self.door_pos + door_width)
        knee_knocker_llip = mut.HPolyhedron.MakeBox(
                                    np.array([-0.035, 0.9-0.518, 0.25]) + self.door_pos + door_width,
                                    np.array([0.035, 0.9-0.518+0.15, 2.0]) + self.door_pos + door_width)
        knee_knocker_rlip = mut.HPolyhedron.MakeBox(
                                    np.array([-0.035, -(0.9-0.518+0.15), 0.25]) + self.door_pos + door_width,
                                    np.array([0.035,  -(0.9-0.518), 2.0]) + self.door_pos + door_width)
        self.obstacles = [floor,
                          knee_knocker_base,
                          knee_knocker_lwall,
                          knee_knocker_rwall,
                          knee_knocker_llip,
                          knee_knocker_rlip,
                          knee_knocker_top]
        self.domain_ubody = mut.HPolyhedron.MakeBox(dom_ubody_lb, dom_ubody_ub)
        self.domain_lbody = mut.HPolyhedron.MakeBox(dom_lbody_lb, dom_lbody_ub)

    def load_reachability_paths(self):
        self.frame_stl_paths, self.poly_halfspace_paths = OrderedDict(), OrderedDict()
        for fr in self.frame_names:
            self.frame_stl_paths[fr] = (cwd +
                                        '/pnc/reachability_map/output/draco3_' + fr + '.stl')
            self.poly_halfspace_paths[fr] = (cwd +
                                         '/pnc/reachability_map/output/draco3_' + fr + '.yaml')


    def get_draco3_default_initial_pose(self):
        q0 = np.zeros(35, )
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
        q0[13] = 0.  # left_ezgripper_knuckle_palm_L1_1
        q0[14] = 0.  # left_ezgripper_knuckle_L1_L2_1
        q0[15] = 0.  # left_ezgripper_knuckle_palm_L1_2
        q0[16] = 0.  # left_ezgripper_knuckle_L1_L2_2
        q0[17] = 0.  # neck pitch
        q0[18] = 0.  # r_hip_ie
        q0[19] = np.radians(-hip_yaw_angle)  # r_hip_aa
        q0[20] = -np.pi / 4  # r_hip_fe
        q0[21] = np.pi / 4  # r_knee_fe_jp
        q0[22] = np.pi / 4  # r_knee_fe_jd
        q0[23] = -np.pi / 4  # r_ankle_fe
        q0[24] = np.radians(hip_yaw_angle)  # r_ankle_ie
        q0[25] = 0.  # r_shoulder_fe
        q0[26] = -np.pi / 6  # r_shoulder_aa
        q0[27] = 0.  # r_shoulder_ie
        q0[28] = -np.pi / 2  # r_elbow_fe
        q0[29] = np.pi/3.   # r_wrist_ps
        q0[30] = 0.  # r_wrist_pitch
        q0[31] = 0.  # right_ezgripper_knuckle_palm_L1_1
        q0[32] = 0.  # right_ezgripper_knuckle_L1_L2_1
        q0[33] = 0.  # right_ezgripper_knuckle_palm_L1_2
        q0[34] = 0.  # right_ezgripper_knuckle_L1_L2_2

        floating_base = np.array([0., 0., 0.741, 0., 0., 0., 1.])
        return np.concatenate((floating_base, q0))

    def get_navy_door_default_initial_pose(self):
        # rotates, then translates
        pos = self.door_pos
        R = util.util.euler_to_rot([0., 0., np.pi / 2.])
        quat = util.util.rot_to_quat(R)
        return np.concatenate((pos, quat))

    def test_use_fixed_and_motion_iris_paths(self):
        self.frame_names = ['torso', 'LF', 'RF', 'LH', 'RH']
        self.load_reachability_paths()
        # load navy door visualization
        door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/ground/navy_door.urdf",
            cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

        # load robot
        draco3 = PinocchioRobotSystem(
            cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
            cwd + "/robot_model/draco3", False, False)
        vis_q = self.get_draco3_default_initial_pose()
        cmd = draco3.create_cmd_ordered_dict(vis_q[7:], np.zeros(len(vis_q[7:])), np.zeros(len(vis_q[7:])))
        draco3.update_system(None, None, None, None,
                             vis_q[:3], vis_q[3:7], np.zeros(3), np.zeros(3),
                             cmd["joint_pos"], cmd["joint_vel"])
        # standing height
        standing_pos = vis_q[:3]

        # ------------------- IRIS -------------------
        step_length = 0.35
        # load obstacle, domain, and start / end seed for IRIS
        obstacles = self.obstacles
        domain_ubody = self.domain_ubody
        domain_lbody = self.domain_lbody
        # shift (feet) iris seed to get nicer IRIS region
        iris_lf_shift = np.array([0.1, 0., 0.])
        iris_rf_shift = np.array([0.1, 0., 0.])
        # get end effector positions via fwd kin
        starting_torso_pos = standing_pos
        final_torso_pos = starting_torso_pos + np.array([step_length, 0., 0.])
        starting_lf_pos = draco3.get_link_iso("l_foot_contact")[:3, 3]
        final_lf_pos = starting_lf_pos + np.array([step_length, 0., 0.])
        starting_lh_pos = draco3.get_link_iso("l_hand_contact")[:3, 3] - np.array([0.01, 0., 0.])
        final_lh_pos = starting_lh_pos + np.array([step_length, 0., 0.])
        starting_rf_pos = draco3.get_link_iso("r_foot_contact")[:3, 3]
        final_rf_pos = starting_rf_pos + np.array([step_length, 0., 0.])
        starting_rh_pos = draco3.get_link_iso("r_hand_contact")[:3, 3] - np.array([0.01, 0., 0.])
        final_rh_pos = starting_rh_pos + np.array([step_length, 0., 0.])

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
        safe_regions_mgr_dict['torso'].computeIris()
        safe_regions_mgr_dict['LF'].computeIris()
        safe_regions_mgr_dict['LH'].computeIris()
        safe_regions_mgr_dict['RF'].computeIris()
        safe_regions_mgr_dict['RH'].computeIris()

        # check that Iris regions cover from start to goal points
        safe_regions_mgr_dict['torso'].connectIrisSeeds()
        safe_regions_mgr_dict['LF'].connectIrisSeeds()
        safe_regions_mgr_dict['LH'].connectIrisSeeds()
        safe_regions_mgr_dict['RF'].connectIrisSeeds()
        safe_regions_mgr_dict['RH'].connectIrisSeeds()

        if b_visualize:
            visualizer = MeshcatVisualizer(draco3._model, draco3._collision_model, draco3._visual_model)

            try:
                visualizer.initViewer(open=True)
                visualizer.viewer.wait()
            except ImportError as err:
                print(
                    "Error while initializing the viewer. It seems you should install Python meshcat"
                )
                print(err)
                sys.exit(0)
            visualizer.loadViewerModel(rootNodeName="draco3")
            visualizer.display(vis_q)

            # load door to visualizer
            door_vis = MeshcatVisualizer(door_model, door_collision_model, door_visual_model)
            door_vis.initViewer(visualizer.viewer)
            door_vis.loadViewerModel(rootNodeName="door")
            door_vis_q = self.get_navy_door_default_initial_pose()
            door_vis.display(door_vis_q)

        else:
            visualizer = None

        # generate all frame traversable regions
        traversable_regions_dict = OrderedDict()
        for fr in self.frame_names:
            if fr == 'torso':
                traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                      b_visualize_reach=b_visualize,
                                                                      b_visualize_safe=b_visualize,
                                                                      visualizer=visualizer)
            else:
                traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                      self.frame_stl_paths[fr],
                                                                      self.poly_halfspace_paths[fr],
                                                                      b_visualize_reach=b_visualize,
                                                                      b_visualize_safe=b_visualize,
                                                                      visualizer=visualizer)
                traversable_regions_dict[fr].update_origin_pose(standing_pos)
            traversable_regions_dict[fr].load_iris_regions(safe_regions_mgr_dict[fr])
        self.assertEqual(True, True)

        # set fixed and motion frame sets
        fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

        # initial and desired final positions for each frame
        p_init = {}
        # torso
        p_init['torso'] = standing_pos
        # left foot
        p_init['LF'] = starting_lf_pos
        # right foot
        p_init['RF'] = starting_rf_pos
        # left hand
        p_init['LH'] = starting_lh_pos
        # right hand
        p_init['RH'] = starting_rh_pos

        # ---- Step 1: L hand to frame
        fixed_frames.append(['LF', 'RF'])   # frames that must not move
        motion_frames_seq.add_motion_frame({'LH': starting_lh_pos + np.array([0.08, 0.07, 0.15])})
        lh_contact_front = PlannerSurfaceContact('LH', np.array([-1, 0, 0]))
        lh_contact_front.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
        motion_frames_seq.add_contact_surface(lh_contact_front)

        # ---- Step 2: step through door with left foot
        fixed_frames.append(['RF', 'LH'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'LF': final_lf_pos})
        lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(lf_contact_over)

        # ---- Step 3: re-position L/R hands for more stability
        fixed_frames.append(['LF', 'RF'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'LH': starting_lh_pos + np.array([0.09, 0.06, 0.18]),
                            'RH': starting_rh_pos + np.array([0.09, -0.06, 0.18])})
        lh_contact_inside = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
        lh_contact_inside.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
        rh_contact_inside = PlannerSurfaceContact('RH', np.array([0, 1, 0]))
        motion_frames_seq.add_contact_surface([lh_contact_inside, rh_contact_inside])

        # ---- Step 4: step through door with right foot
        fixed_frames.append(['LF', 'LH', 'RH'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'RF': final_rf_pos,
                            'torso': final_torso_pos})
                            # 'R_knee': final_rkn_pos
        rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(rf_contact_over)

        # ---- Step 5: square up
        fixed_frames.append(['torso', 'LF', 'RF', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({})


        # make multi-trajectory planner
        T = 3
        alpha = [0, 0, 1]
        traversable_regions = [traversable_regions_dict['torso'],
                               traversable_regions_dict['LF'],
                               traversable_regions_dict['RF'],
                               traversable_regions_dict['LH'],
                               traversable_regions_dict['RH']]

        frame_planner = LocomanipulationFramePlanner(traversable_regions,
                                                     self.ee_offsets_path,
                                                     aux_frames_path=None,  #self.aux_frames_path
                                                     fixed_frames=fixed_frames,
                                                     motion_frames_seq=motion_frames_seq)
        frame_planner.plan_iris(p_init, T, alpha)
        frame_planner.plot(visualizer=visualizer)
        self.assertEqual(True, True)

    def test_use_fixed_and_motion_kn_iris_paths(self):
        self.frame_names = ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']
        # load navy door visualization
        door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/ground/navy_door.urdf",
            cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

        # load robot
        draco3 = PinocchioRobotSystem(
            cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
            cwd + "/robot_model/draco3", False, False)
        vis_q = self.get_draco3_default_initial_pose()
        cmd = draco3.create_cmd_ordered_dict(vis_q[7:], np.zeros(len(vis_q[7:])), np.zeros(len(vis_q[7:])))
        draco3.update_system(None, None, None, None,
                             vis_q[:3], vis_q[3:7], np.zeros(3), np.zeros(3),
                             cmd["joint_pos"], cmd["joint_vel"])
        # standing height
        standing_pos = vis_q[:3]

        # ------------------- IRIS -------------------
        step_length = 0.35
        # load obstacle, domain, and start / end seed for IRIS
        obstacles = self.obstacles
        domain_ubody = self.domain_ubody
        domain_lbody = self.domain_lbody
        # shift (feet) iris seed to get nicer IRIS region
        iris_lf_shift = np.array([0.1, 0., 0.])
        iris_rf_shift = np.array([0.1, 0., 0.])
        # get end effector positions via fwd kin
        starting_torso_pos = standing_pos
        final_torso_pos = starting_torso_pos + np.array([step_length, 0., 0.])
        starting_lf_pos = draco3.get_link_iso("l_foot_contact")[:3, 3]
        final_lf_pos = starting_lf_pos + np.array([step_length, 0., 0.])
        starting_lh_pos = draco3.get_link_iso("l_hand_contact")[:3, 3] - np.array([0.01, 0., 0.])
        final_lh_pos = starting_lh_pos + np.array([step_length, 0., 0.])
        starting_rf_pos = draco3.get_link_iso("r_foot_contact")[:3, 3]
        final_rf_pos = starting_rf_pos + np.array([step_length, 0., 0.])
        starting_rh_pos = draco3.get_link_iso("r_hand_contact")[:3, 3] - np.array([0.01, 0., 0.])
        final_rh_pos = starting_rh_pos + np.array([step_length, 0., 0.])
        starting_lkn_pos = draco3.get_link_iso("l_knee_fe_ld")[:3, 3]
        final_lkn_pos = starting_lkn_pos + np.array([step_length, 0., 0.])
        starting_rkn_pos = draco3.get_link_iso("r_knee_fe_ld")[:3, 3]
        final_rkn_pos = starting_rkn_pos + np.array([step_length, 0., 0.])

        safe_torso_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_torso_pos)
        safe_torso_end_region = IrisGeomInterface(obstacles, domain_ubody, final_torso_pos)
        safe_lf_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_lf_pos + iris_lf_shift)
        safe_lf_end_region = IrisGeomInterface(obstacles, domain_lbody, final_lf_pos)
        safe_lk_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_lkn_pos)
        safe_lk_end_region = IrisGeomInterface(obstacles, domain_lbody, final_lkn_pos - iris_lf_shift)
        safe_lh_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_lh_pos)
        safe_lh_end_region = IrisGeomInterface(obstacles, domain_ubody, final_lh_pos)
        safe_rf_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_rf_pos + iris_rf_shift)
        safe_rf_end_region = IrisGeomInterface(obstacles, domain_lbody, final_rf_pos)
        safe_rk_start_region = IrisGeomInterface(obstacles, domain_lbody, starting_rkn_pos)
        safe_rk_end_region = IrisGeomInterface(obstacles, domain_lbody, final_rkn_pos - iris_rf_shift)
        safe_rh_start_region = IrisGeomInterface(obstacles, domain_ubody, starting_rh_pos)
        safe_rh_end_region = IrisGeomInterface(obstacles, domain_ubody, final_rh_pos)
        safe_regions_mgr_dict = {'torso': IrisRegionsManager(safe_torso_start_region, safe_torso_end_region),
                                 'LF': IrisRegionsManager(safe_lf_start_region, safe_lf_end_region),
                                 'L_knee': IrisRegionsManager(safe_lk_start_region, safe_lk_end_region),
                                 'LH': IrisRegionsManager(safe_lh_start_region, safe_lh_end_region),
                                 'RF': IrisRegionsManager(safe_rf_start_region, safe_rf_end_region),
                                 'R_knee': IrisRegionsManager(safe_rk_start_region, safe_rk_end_region),
                                 'RH': IrisRegionsManager(safe_rh_start_region, safe_rh_end_region)}
        safe_regions_mgr_dict['torso'].computeIris()
        safe_regions_mgr_dict['LF'].computeIris()
        safe_regions_mgr_dict['L_knee'].computeIris()
        safe_regions_mgr_dict['LH'].computeIris()
        safe_regions_mgr_dict['RF'].computeIris()
        safe_regions_mgr_dict['R_knee'].computeIris()
        safe_regions_mgr_dict['RH'].computeIris()

        # check that Iris regions cover from start to goal points
        safe_regions_mgr_dict['torso'].connectIrisSeeds()
        safe_regions_mgr_dict['LF'].connectIrisSeeds()
        safe_regions_mgr_dict['L_knee'].connectIrisSeeds()
        safe_regions_mgr_dict['LH'].connectIrisSeeds()
        safe_regions_mgr_dict['RF'].connectIrisSeeds()
        safe_regions_mgr_dict['R_knee'].connectIrisSeeds()
        safe_regions_mgr_dict['RH'].connectIrisSeeds()

        if b_visualize:
            visualizer = MeshcatVisualizer(draco3._model, draco3._collision_model, draco3._visual_model)

            try:
                visualizer.initViewer(open=True)
                visualizer.viewer.wait()
            except ImportError as err:
                print(
                    "Error while initializing the viewer. It seems you should install Python meshcat"
                )
                print(err)
                sys.exit(0)
            visualizer.loadViewerModel(rootNodeName="draco3")
            visualizer.display(vis_q)

            # load door to visualizer
            door_vis = MeshcatVisualizer(door_model, door_collision_model, door_visual_model)
            door_vis.initViewer(visualizer.viewer)
            door_vis.loadViewerModel(rootNodeName="door")
            door_vis_q = self.get_navy_door_default_initial_pose()
            door_vis.display(door_vis_q)

        else:
            visualizer = None

        # generate all frame traversable regions
        traversable_regions_dict = OrderedDict()
        for fr in self.frame_names:
            if fr == 'torso':
                traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                      b_visualize_reach=b_visualize,
                                                                      b_visualize_safe=b_visualize,
                                                                      visualizer=visualizer)
            else:
                traversable_regions_dict[fr] = FrameTraversableRegion(fr,
                                                                      self.frame_stl_paths[fr],
                                                                      self.poly_halfspace_paths[fr],
                                                                      b_visualize_reach=b_visualize,
                                                                      b_visualize_safe=b_visualize,
                                                                      visualizer=visualizer)
                traversable_regions_dict[fr].update_origin_pose(standing_pos)
            traversable_regions_dict[fr].load_iris_regions(safe_regions_mgr_dict[fr])
        self.assertEqual(True, True)

        # set fixed and motion frame sets
        fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

        # initial and desired final positions for each frame
        p_init = {}
        # torso
        p_init['torso'] = standing_pos
        # left foot
        p_init['LF'] = starting_lf_pos
        # right foot
        p_init['RF'] = starting_rf_pos
        # left knee
        p_init['L_knee'] = starting_lkn_pos
        # right knee
        p_init['R_knee'] = starting_rkn_pos
        # left hand
        p_init['LH'] = starting_lh_pos
        # right hand
        p_init['RH'] = starting_rh_pos

        # ---- Step 1: L hand to frame
        # fixed_frames.append(['LF', 'RF'])   # frames that must not move
        fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
        motion_frames_seq.add_motion_frame({'LH': starting_lh_pos + np.array([0.08, 0.07, 0.15])})
        lh_contact_front = PlannerSurfaceContact('LH', np.array([-1, 0, 0]))
        lh_contact_front.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
        motion_frames_seq.add_contact_surface(lh_contact_front)

        # ---- Step 2: step through door with left foot
        # fixed_frames.append(['RF', 'LH'])   # frames that must not move
        fixed_frames.append(['RF', 'R_knee', 'LH'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'LF': final_lf_pos,
                            'L_knee': final_lkn_pos})
        lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(lf_contact_over)

        # ---- Step 3: re-position L/R hands for more stability
        # fixed_frames.append(['LF', 'RF'])   # frames that must not move
        fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'LH': starting_lh_pos + np.array([0.09, 0.06, 0.18]),
                            'RH': starting_rh_pos + np.array([0.09, -0.06, 0.18])})
        lh_contact_inside = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
        lh_contact_inside.set_contact_breaking_velocity(np.array([-1, 0., 0.]))
        rh_contact_inside = PlannerSurfaceContact('RH', np.array([0, 1, 0]))
        motion_frames_seq.add_contact_surface([lh_contact_inside, rh_contact_inside])

        # ---- Step 4: step through door with right foot
        # fixed_frames.append(['LF', 'LH', 'RH'])   # frames that must not move
        fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'RF': final_rf_pos,
                            'torso': final_torso_pos,
                            'R_knee': final_rkn_pos})
        rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(rf_contact_over)

        # ---- Step 5: square up
        # fixed_frames.append(['torso', 'LF', 'RF', 'LH', 'RH'])
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({})


        # make multi-trajectory planner
        T = 3
        alpha = [0, 0, 1]
        # alpha = {1: 1, 2: 2, 3: 0.1}
        traversable_regions = [traversable_regions_dict['torso'],
                               traversable_regions_dict['LF'],
                               traversable_regions_dict['RF'],
                               traversable_regions_dict['L_knee'],
                               traversable_regions_dict['R_knee'],
                               traversable_regions_dict['LH'],
                               traversable_regions_dict['RH']]

        frame_planner = LocomanipulationFramePlanner(traversable_regions,
                                                     self.ee_offsets_path,
                                                     aux_frames_path=self.aux_frames_path,
                                                     fixed_frames=fixed_frames,
                                                     motion_frames_seq=motion_frames_seq)
        frame_planner.plan_iris(p_init, T, alpha)
        frame_planner.plot(visualizer=visualizer)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
