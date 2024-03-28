import unittest
from collections import OrderedDict

import util.util
from pnc.planner.multicontact.frame_traversable_region import FrameTraversableRegion
from pnc.planner.multicontact.locomanipulation_frame_planner import LocomanipulationFramePlanner

import os
import sys
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin

from pnc.planner.multicontact.planner_surface_contact import (
        PlannerSurfaceContact, MotionFrameSequencer)

cwd = os.getcwd()
sys.path.append(cwd)


class TestFrameTraversableRegion(unittest.TestCase):
    def setUp(self):
        self.torso_frame_name = 'torso'

        # self.frame_names = ['torso', 'LF', 'RF', 'L_knee', 'R_knee']
        self.frame_names = ['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']
        self.frame_stl_paths, self.poly_halfspace_paths = OrderedDict(), OrderedDict()
        for fr in self.frame_names:
            self.frame_stl_paths[fr] = (cwd +
                                        '/pnc/reachability_map/output/draco3_' + fr + '.stl')
            self.poly_halfspace_paths[fr] = (cwd +
                                         '/pnc/reachability_map/output/draco3_' + fr + '.yaml')

        self.ee_offsets_path = cwd + '/pnc/reachability_map/output/draco3_ee_offsets.yaml'

        self.aux_frames_path = cwd + '/pnc/reachability_map/output/draco3_aux_frames.yaml'

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

        floating_base = np.array([0., 0., 0.74, 0., 0., 0., 1.])
        return np.concatenate((floating_base, q0))

    def get_navy_door_default_initial_pose(self):
        # rotates, then translates
        pos = np.array([0.3, 0., 0.])
        R = util.util.euler_to_rot([0., 0., np.pi / 2.])
        quat = util.util.rot_to_quat(R)
        return np.concatenate((pos, quat))

    def get_sample_collision_free_boxes(self):
        box_llim, box_ulim = OrderedDict(), OrderedDict()
        box_llim['LF'] = np.array([
            [-0.1, -0.1, 0.0],  # prevent leg-crossing
            [-0.1, -0.1, 0.4],  # prevent leg-crossing
            [0.2, -0.1, 0.0]  # prevent leg-crossing
        ])
        box_llim['RF'] = np.array([
            [-0.1, -0.3, 0.0],  # prevent leg-crossing
            [-0.1, -0.3, 0.4],  # prevent leg-crossing
            [0.2, -0.3, 0.0]  # prevent leg-crossing
        ])
        box_llim['LH'] = np.array([
            [-0.1, -0.1, 0.7],  # prevent leg-crossing
            [-0.1, 0.0, 0.7],  # prevent leg-crossing
            [0.2, -0.1, 0.7]  # prevent leg-crossing
        ])
        box_llim['RH'] = np.array([
            [-0.1, -0.3, 0.7],  # prevent leg-crossing
            [-0.1, -0.3, 0.7],  # prevent leg-crossing
            [0.2, -0.3, 0.7]  # prevent leg-crossing
        ])

        # upper bounds of the safe boxes
        box_ulim['LF'] = np.array([
            [0.1, 0.3, 0.6],  # z stops at kin. limit
            [0.4, 0.3, 0.6],  # x stops at kin. limit
            [0.5, 0.3, 0.6]  # x stops at kin. limit
        ])
        box_ulim['RF'] = np.array([
            [0.1, 0.1, 0.6],  # prevent leg-crossing
            [0.4, 0.1, 0.6],  # prevent leg-crossing
            [0.5, 0.1, 0.6]  # prevent leg-crossing
        ])
        box_ulim['LH'] = np.array([
            [0.1, 0.3, 1.3],  # prevent leg-crossing
            [0.4, 0.3, 1.3],  # prevent leg-crossing
            [0.5, 0.3, 1.3]  # prevent leg-crossing
        ])
        box_ulim['RH'] = np.array([
            [0.1, 0.1, 1.3],  # prevent leg-crossing
            [0.4, 0.0, 1.3],  # prevent leg-crossing
            [0.5, 0.1, 1.3]  # prevent leg-crossing
        ])
        return box_llim, box_ulim

    def get_navy_door_collision_free_boxes(self):
        # save safe box regions
        box_llim, box_ulim = OrderedDict(), OrderedDict()

        # lower bounds of end-effectors safe boxes
        box_llim['torso'] = np.array([
            [-0.1, -0.3, 0.5],  # prevent leg-crossing
            [0.05, -0.3, 0.5],  # prevent leg-crossing
            [0.2, -0.3, 0.5]  # prevent leg-crossing
        ])
        box_llim['LF'] = np.array([
            [-0.2, 0.0, 0.0],  # prevent leg-crossing
            [-0.2, 0.0, 0.4],  # prevent leg-crossing
            [0.4, 0.0, 0.0]  # prevent leg-crossing
        ])
        box_llim['RF'] = np.array([
            [-0.2, -0.4, 0.0],  # prevent leg-crossing
            [-0.2, -0.35, 0.4],  # prevent leg-crossing
            [0.4, -0.4, 0.0]  # prevent leg-crossing
        ])
        box_llim['L_knee'] = np.array([
            [-0.2, 0.0, 0.0],  # prevent leg-crossing
            [-0.1, 0.0, 0.4],  # prevent leg-crossing
            [0.4, 0.0, 0.0]  # prevent leg-crossing
        ])
        box_llim['R_knee'] = np.array([
            [-0.2, -0.4, 0.0],  # prevent leg-crossing
            [-0.1, -0.35, 0.4],  # prevent leg-crossing
            [0.4, -0.4, 0.0]  # prevent leg-crossing
        ])
        box_llim['LH'] = np.array([
            [0.1, 0.0, 0.7],  # prevent leg-crossing
            [0.25, 0.0, 0.7],  # prevent leg-crossing
            [0.5, 0.0, 0.7]  # prevent leg-crossing
        ])
        box_llim['RH'] = np.array([
            [0.1, -0.4, 0.7],  # prevent leg-crossing
            [0.25, -0.35, 0.7],  # prevent leg-crossing
            [0.5, -0.4, 0.7]  # prevent leg-crossing
        ])

        # upper bounds of the safe boxes
        box_ulim['torso'] = np.array([
            [0.1, 0.3, 0.9],  # prevent leg-crossing
            [0.3, 0.3, 0.9],  # prevent leg-crossing
            [0.6, 0.3, 0.9]  # prevent leg-crossing
        ])
        box_ulim['LF'] = np.array([
            [0.25, 0.35, 0.6],  # z stops at kin. limit
            [0.6, 0.4, 0.6],  # x stops at kin. limit
            [0.8, 0.4, 0.6]  # x stops at kin. limit
        ])
        box_ulim['RF'] = np.array([
            [0.25, 0.0, 0.6],  # prevent leg-crossing
            [0.6, 0.0, 0.6],  # prevent leg-crossing
            [0.8, 0.0, 0.6]  # prevent leg-crossing
        ])
        box_ulim['L_knee'] = np.array([
            [0.25, 0.35, 0.6],  # z stops at kin. limit
            [0.8, 0.4, 0.6],  # x stops at kin. limit
            [0.8, 0.4, 0.6]  # x stops at kin. limit
        ])
        box_ulim['R_knee'] = np.array([
            [0.25, 0.0, 0.6],  # prevent leg-crossing
            [0.8, 0.0, 0.6],  # prevent leg-crossing
            [0.8, 0.0, 0.6]  # prevent leg-crossing
        ])
        box_ulim['LH'] = np.array([
            [0.25, 0.4, 1.3],  # prevent leg-crossing
            [0.55, 0.35, 1.3],  # prevent leg-crossing
            [0.8, 0.4, 1.3]  # prevent leg-crossing
        ])
        box_ulim['RH'] = np.array([
            [0.25, 0.0, 1.3],  # prevent leg-crossing
            [0.55, 0.0, 1.3],  # prevent leg-crossing
            [0.8, 0.0, 1.3]  # prevent leg-crossing
        ])
        return box_llim, box_ulim

    def get_semi_static_collision_free_boxes(self):
        # save safe box regions
        box_llim, box_ulim = OrderedDict(), OrderedDict()

        # lower bounds of end-effectors safe boxes
        box_llim['torso'] = np.array([
            [-0.1, -0.3, 0.5],  # prevent leg-crossing
            [0.05, -0.3, 0.5],  # prevent leg-crossing
            [0.2, -0.3, 0.5]  # prevent leg-crossing
        ])
        box_llim['LF'] = np.array([
            [-0.2, 0.0, 0.0],  # prevent leg-crossing
            [-0.1, 0.0, 0.4],  # prevent leg-crossing
            [0.4, 0.0, 0.0]  # prevent leg-crossing
        ])
        box_llim['RF'] = np.array([
            [-0.2, -0.4, 0.0],  # prevent leg-crossing
            [-0.1, -0.35, 0.4],  # prevent leg-crossing
            [0.4, -0.4, 0.0]  # prevent leg-crossing
        ])
        box_llim['L_knee'] = np.array([
            [-0.2, 0.0, 0.0],  # prevent leg-crossing
            [-0.1, 0.0, 0.4],  # prevent leg-crossing
            [0.4, 0.0, 0.0]  # prevent leg-crossing
        ])
        box_llim['R_knee'] = np.array([
            [-0.2, -0.4, 0.0],  # prevent leg-crossing
            [-0.1, -0.35, 0.4],  # prevent leg-crossing
            [0.4, -0.4, 0.0]  # prevent leg-crossing
        ])
        box_llim['LH'] = np.array([
            [-0.1, 0.0, 0.7],  # prevent leg-crossing
            [0.1, 0.0, 0.7],  # prevent leg-crossing
            [0.5, 0.0, 0.7]  # prevent leg-crossing
        ])
        box_llim['RH'] = np.array([
            [-0.1, -0.4, 0.7],  # prevent leg-crossing
            [0.1, -0.38, 0.7],  # prevent leg-crossing
            [0.5, -0.4, 0.7]  # prevent leg-crossing
        ])

        # upper bounds of the safe boxes
        box_ulim['torso'] = np.array([
            [0.1, 0.3, 0.9],  # prevent leg-crossing
            [0.3, 0.3, 0.9],  # prevent leg-crossing
            [0.6, 0.3, 0.9]  # prevent leg-crossing
        ])
        box_ulim['LF'] = np.array([
            [0.25, 0.4, 0.6],  # z stops at kin. limit
            [0.6, 0.35, 0.6],  # x stops at kin. limit
            [0.8, 0.4, 0.6]  # x stops at kin. limit
        ])
        box_ulim['RF'] = np.array([
            [0.25, 0.0, 0.6],  # prevent leg-crossing
            [0.6, 0.0, 0.6],  # prevent leg-crossing
            [0.8, 0.0, 0.6]  # prevent leg-crossing
        ])
        box_ulim['L_knee'] = np.array([
            [0.25, 0.4, 0.6],  # z stops at kin. limit
            [0.6, 0.35, 0.6],  # x stops at kin. limit
            [0.8, 0.4, 0.6]  # x stops at kin. limit
        ])
        box_ulim['R_knee'] = np.array([
            [0.25, 0.0, 0.6],  # prevent leg-crossing
            [0.6, 0.0, 0.6],  # prevent leg-crossing
            [0.8, 0.0, 0.6]  # prevent leg-crossing
        ])
        box_ulim['LH'] = np.array([
            [0.15, 0.45, 1.3],  # prevent leg-crossing
            [0.55, 0.38, 1.3],  # prevent leg-crossing
            [0.8, 0.45, 1.3]  # prevent leg-crossing
        ])
        box_ulim['RH'] = np.array([
            [0.15, 0.0, 1.3],  # prevent leg-crossing
            [0.55, 0.0, 1.3],  # prevent leg-crossing
            [0.8, 0.0, 1.3]  # prevent leg-crossing
        ])
        return box_llim, box_ulim

    def get_long_semi_static_collision_free_boxes(self):
        # save safe box regions
        box_llim, box_ulim = OrderedDict(), OrderedDict()

        # lower bounds of end-effectors safe boxes
        box_llim['torso'] = np.array([
            [-0.2, -0.3, 0.5],  # prevent leg-crossing
            [0.05, -0.3, 0.5],  # prevent leg-crossing
            [0.2, -0.3, 0.5]  # prevent leg-crossing
        ])
        box_llim['LF'] = np.array([
            [-0.2, 0.0, 0.0],  # prevent leg-crossing
            [-0.1, 0.0, 0.4],  # prevent leg-crossing
            [0.4, 0.0, 0.0]  # prevent leg-crossing
        ])
        box_llim['RF'] = np.array([
            [-0.2, -0.4, 0.0],  # prevent leg-crossing
            [-0.1, -0.35, 0.4],  # prevent leg-crossing
            [0.4, -0.4, 0.0]  # prevent leg-crossing
        ])
        box_llim['L_knee'] = np.array([
            [-0.2, 0.0, 0.0],  # prevent leg-crossing
            [-0.1, 0.0, 0.4],  # prevent leg-crossing
            [0.4, 0.0, 0.0]  # prevent leg-crossing
        ])
        box_llim['R_knee'] = np.array([
            [-0.2, -0.4, 0.0],  # prevent leg-crossing
            [-0.1, -0.35, 0.4],  # prevent leg-crossing
            [0.5, -0.4, 0.0]  # prevent leg-crossing
        ])
        box_llim['LH'] = np.array([
            [-0.2, 0.0, 0.7],  # prevent leg-crossing
            [0.1, 0.0, 0.7],  # prevent leg-crossing
            [0.5, 0.0, 0.7]  # prevent leg-crossing
        ])
        box_llim['RH'] = np.array([
            [-0.2, -0.4, 0.7],  # prevent leg-crossing
            [0.1, -0.38, 0.7],  # prevent leg-crossing
            [0.5, -0.4, 0.7]  # prevent leg-crossing
        ])

        # upper bounds of the safe boxes
        box_ulim['torso'] = np.array([
            [0.25, 0.3, 0.9],  # prevent leg-crossing
            [0.5, 0.3, 0.9],  # prevent leg-crossing
            [0.6, 0.3, 0.9]  # prevent leg-crossing
        ])
        box_ulim['LF'] = np.array([
            [0.25, 0.4, 1.0],  # z stops at kin. limit
            [0.6, 0.35, 1.0],  # x stops at kin. limit
            [0.8, 0.4, 1.0]  # x stops at kin. limit
        ])
        box_ulim['RF'] = np.array([
            [0.25, 0.0, 1.0],  # prevent leg-crossing
            [0.6, 0.0, 1.0],  # prevent leg-crossing
            [0.8, 0.0, 1.0]  # prevent leg-crossing
        ])
        box_ulim['L_knee'] = np.array([
            [0.25, 0.4, 1.0],  # z stops at kin. limit
            [0.6, 0.35, 1.0],  # x stops at kin. limit
            [0.8, 0.4, 1.0]  # x stops at kin. limit
        ])
        box_ulim['R_knee'] = np.array([
            [0.25, 0.0, 1.0],  # prevent leg-crossing
            [0.6, 0.0, 1.0],  # prevent leg-crossing
            [0.8, 0.0, 1.0]  # prevent leg-crossing
        ])
        box_ulim['LH'] = np.array([
            [0.15, 0.45, 1.3],  # prevent leg-crossing
            [0.55, 0.38, 1.3],  # prevent leg-crossing
            [0.8, 0.45, 1.3]  # prevent leg-crossing
        ])
        box_ulim['RH'] = np.array([
            [0.15, 0.0, 1.3],  # prevent leg-crossing
            [0.55, 0.0, 1.3],  # prevent leg-crossing
            [0.8, 0.0, 1.3]  # prevent leg-crossing
        ])
        return box_llim, box_ulim

    def test_visualizing_reachable_region(self):
        frame_name = 'RF'
        test_region = FrameTraversableRegion(frame_name,
                                             self.frame_stl_paths[frame_name],
                                             self.poly_halfspace_paths[frame_name],
                                             b_visualize_reach=True)

        # translate to new origin
        new_origin = np.array([0., 0., 0.6])
        test_region.update_origin_pose(new_origin)
        self.assertEqual(True, True)

    def test_visualize_region_with_robot_model(self):
        frame_name = 'RF'
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
            cwd + "/robot_model/draco3", pin.JointModelFreeFlyer())
        visualizer = MeshcatVisualizer(model, collision_model, visual_model)

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
        vis_q = pin.neutral(model)
        visualizer.display(vis_q)

        test_region = FrameTraversableRegion(frame_name,
                                             self.frame_stl_paths[frame_name],
                                             self.poly_halfspace_paths[frame_name],
                                             visualizer=visualizer,
                                             b_visualize_reach=True,
                                             b_visualize_safe=True)
        self.assertEqual(True, True)

        # move to standing height
        standing_pos = np.array([0., 0., 0.6])
        vis_q[:3] = standing_pos
        visualizer.display(vis_q)
        test_region.update_origin_pose(standing_pos)
        self.assertEqual(True, True)

    def test_visualize_region_and_ee_paths(self):
        frame_name = 'RF'
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
            cwd + "/robot_model/draco3", pin.JointModelFreeFlyer())
        visualizer = MeshcatVisualizer(model, collision_model, visual_model)

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
        vis_q = self.get_draco3_default_initial_pose()
        visualizer.display(vis_q)

        test_region = FrameTraversableRegion(frame_name,
                                             self.frame_stl_paths[frame_name],
                                             self.poly_halfspace_paths[frame_name],
                                             visualizer=visualizer,
                                             b_visualize_safe=True,
                                             b_visualize_reach=True)
        self.assertEqual(True, True)

        # move to standing height
        standing_pos = vis_q[:3]
        test_region.update_origin_pose(standing_pos)
        self.assertEqual(True, True)

        # collision-free boxes
        box_llim, box_ulim = self.get_sample_collision_free_boxes()
        test_region.load_collision_free_boxes(box_llim[frame_name], box_ulim[frame_name])

    def test_visualize_navy_door_paths(self):
        b_visualize = True
        # load robot
        rob_model, rob_collision_model, rob_visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
            cwd + "/robot_model/draco3", pin.JointModelFreeFlyer())
        vis_q = self.get_draco3_default_initial_pose()

        door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/ground/navy_door.urdf",
            cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

        if b_visualize:
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

        # standing height
        standing_pos = vis_q[:3]

        # collision-free boxes
        box_llim, box_ulim = self.get_navy_door_collision_free_boxes()

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
            traversable_regions_dict[fr].load_collision_free_boxes(box_llim[fr], box_ulim[fr])
        self.assertEqual(True, True)

        step_length = 0.45
        # initial and desired final positions for each frame
        p_init, p_end = {}, {}
        #
        # torso
        #
        p_init['torso'] = standing_pos
        p_end['torso'] = standing_pos + np.array([step_length, 0., 0.])

        #
        # left foot
        #
        p_init['LF'] = np.array([0.06, 0.14, 0.])  # TODO: use fwd kin
        p_end['LF'] = p_init['LF'] + np.array([step_length, 0., 0.])

        #
        # right foot
        #
        p_init['RF'] = np.array([0.06, -0.14, 0.])  # TODO: use fwd kin
        p_end['RF'] = p_init['RF'] + np.array([step_length, 0., 0.])

        #
        # left knee
        #
        R = util.util.euler_to_rot([0., np.pi / 6, 0.])
        lknee_lf_offset = R @ np.array([0., -0.00599, 0.324231])  # rotate by 30deg about y
        p_init['L_knee'] = p_init['LF'] + lknee_lf_offset  # TODO: use fwd kin
        p_end['L_knee'] = p_end['LF'] + lknee_lf_offset

        #
        # right knee
        #
        rknee_rf_offset = R @ np.array([0., 0.006, 0.324231])
        p_init['R_knee'] = p_init['RF'] + rknee_rf_offset  # TODO: use fwd kin
        p_end['R_knee'] = p_end['RF'] + rknee_rf_offset

        #
        # left hand
        #
        p_init['LH'] = np.array([0.22, 0.3, standing_pos[2]])  # TODO: use fwd kin
        p_end['LH'] = p_init['LH'] + np.array([step_length, 0., 0.])

        #
        # right hand
        #
        p_init['RH'] = np.array([0.22, -0.3, standing_pos[2]])  # TODO: use fwd kin
        p_end['RH'] = p_init['RH'] + np.array([step_length, 0., 0.])

        # make multi-trajectory planner
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
                                                     self.ee_offsets_path,
                                                     aux_frames_path=self.aux_frames_path)
        frame_planner.plan(p_init, p_end, T, alpha)
        frame_planner.plot(visualizer=visualizer)
        self.assertEqual(True, True)

        # add auxiliary collision-free frames
        # aux_frame = 'R_knee'
        # aux_frame = 'R_knee'
        # aux_frame_region = 'RF'
        # frame_planner.add_reachable_frame_constraint(aux_frame, aux_frame_region)

    def test_use_fixed_and_motion_paths(self):
        b_visualize = True
        # load robot
        rob_model, rob_collision_model, rob_visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
            cwd + "/robot_model/draco3", pin.JointModelFreeFlyer())
        vis_q = self.get_draco3_default_initial_pose()

        door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/ground/navy_door.urdf",
            cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

        if b_visualize:
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

        # standing height
        standing_pos = vis_q[:3]

        # collision-free boxes
        box_llim, box_ulim = self.get_long_semi_static_collision_free_boxes()

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
            traversable_regions_dict[fr].load_collision_free_boxes(box_llim[fr], box_ulim[fr])
        self.assertEqual(True, True)

        # set fixed and motion frame sets
        fixed_frames, motion_frames_seq = [], MotionFrameSequencer()

        step_length = 0.35
        # initial and desired final positions for each frame
        p_init, p_end = {}, {}
        #
        # torso
        #
        p_init['torso'] = standing_pos
        p_end['torso'] = standing_pos + np.array([step_length, 0., 0.])

        #
        # left foot
        #
        p_init['LF'] = np.array([0.06, 0.14, 0.])  # TODO: use fwd kin
        p_end['LF'] = p_init['LF']

        #
        # right foot
        #
        p_init['RF'] = np.array([0.06, -0.14, 0.])  # TODO: use fwd kin
        p_end['RF'] = p_init['RF']

        #
        # left knee
        #
        R = util.util.euler_to_rot([0., np.pi / 6, 0.])
        lknee_lf_offset = R @ np.array([0., -0.00599, 0.324231])  # rotate by 30deg about y
        p_init['L_knee'] = p_init['LF'] + lknee_lf_offset  # TODO: use fwd kin
        p_end['L_knee'] = p_end['LF'] + lknee_lf_offset

        #
        # right knee
        #
        rknee_rf_offset = R @ np.array([0., 0.006, 0.324231])
        p_init['R_knee'] = p_init['RF'] + rknee_rf_offset  # TODO: use fwd kin
        p_end['R_knee'] = p_end['RF'] + rknee_rf_offset

        #
        # left hand
        #
        p_init['LH'] = np.array([0.22, 0.3, standing_pos[2]])  # TODO: use fwd kin
        p_end['LH'] = p_init['LH'] + np.array([0.08, 0.07, 0.15])

        #
        # right hand
        #
        p_init['RH'] = np.array([0.22, -0.3, standing_pos[2]])  # TODO: use fwd kin
        p_end['RH'] = p_init['RH']

        # ---- Step 1: L hand to frame
        fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
        motion_frames_seq.add_motion_frame({'LH': p_init['LH'] + np.array([0.08, 0.07, 0.15])})
        lh_contact_front = PlannerSurfaceContact('LH', np.array([-1, 0, 0]))
        motion_frames_seq.add_contact_surface(lh_contact_front)

        # ---- Step 2: step through door with left foot
        fixed_frames.append(['RF', 'R_knee', 'LH'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'LF': p_init['LF'] + np.array([step_length, 0., 0.]),
                            'L_knee': p_init['L_knee'] + np.array([step_length, 0., 0.])})
        lf_contact_over = PlannerSurfaceContact('LF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(lf_contact_over)

        # ---- Step 3: re-position L/R hands for more stability
        fixed_frames.append(['LF', 'RF', 'L_knee', 'R_knee'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'LH': p_init['LH'] + np.array([0.09, 0.06, 0.18]),
                            'RH': p_init['RH'] + np.array([0.09, -0.06, 0.18])})
        lh_contact_inside = PlannerSurfaceContact('LH', np.array([0, -1, 0]))
        rh_contact_inside = PlannerSurfaceContact('RH', np.array([0, 1, 0]))
        motion_frames_seq.add_contact_surface([lh_contact_inside, rh_contact_inside])

        # ---- Step 4: step through door with right foot
        fixed_frames.append(['LF', 'L_knee', 'LH', 'RH'])   # frames that must not move
        motion_frames_seq.add_motion_frame({
                            'RF': p_init['RF'] + np.array([step_length, 0., 0.]),
                            'R_knee': p_init['R_knee'] + np.array([step_length, 0., 0.]),
                            'torso': p_init['torso'] + np.array([step_length, 0., 0.])})
        rf_contact_over = PlannerSurfaceContact('RF', np.array([0, 0, 1]))
        motion_frames_seq.add_contact_surface(rf_contact_over)

        # ---- Step 5: square up
        fixed_frames.append(['torso', 'LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH'])
        motion_frames_seq.add_motion_frame({})


        # make multi-trajectory planner
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
                                                     self.ee_offsets_path,
                                                     aux_frames_path=self.aux_frames_path,
                                                     fixed_frames=fixed_frames,
                                                     motion_frames_seq=motion_frames_seq)
        frame_planner.plan(p_init, p_end, T, alpha, verbose=False)
        frame_planner.plot(visualizer=visualizer, static_html=False)
        self.assertEqual(True, True)

    def test_polytope_offset(self):
        b_visualize = True
        # load robot
        rob_model, rob_collision_model, rob_visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
            cwd + "/robot_model/draco3", pin.JointModelFreeFlyer())
        vis_q = self.get_draco3_default_initial_pose()

        door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/ground/navy_door.urdf",
            cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

        if b_visualize:
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
            visualizer.loadViewerModel(rootNodeName="draco3")
            visualizer.display(vis_q)
        else:
            visualizer = None

        #
        # left foot
        #
        lf_init = np.array([0.06, 0.14, 0.])  # TODO: use fwd kin
        lf_end = lf_init + np.array([0.45, 0., 0.])
        lf_traversable_region = FrameTraversableRegion(self.lf_frame_name,
                                                       self.lf_frame_stl_path, self.lf_poly_halfspace_path,
                                                       b_visualize_reach=b_visualize,
                                                       b_visualize_safe=b_visualize,
                                                       visualizer=visualizer)
        self.assertEqual(True, True)

        # move to standing height
        standing_pos = vis_q[:3]
        lf_traversable_region.update_origin_pose(standing_pos)
        self.assertEqual(True, True)

        #
        # right foot
        #
        rf_init = np.array([0.06, -0.14, 0.])  # TODO: use fwd kin
        rf_end = rf_init + np.array([0.45, 0., 0.])
        rf_traversable_region = FrameTraversableRegion(self.rf_frame_name,
                                                       self.rf_frame_stl_path, self.rf_poly_halfspace_path,
                                                       b_visualize_reach=b_visualize,
                                                       b_visualize_safe=b_visualize,
                                                       visualizer=visualizer)
        self.assertEqual(True, True)

        # move to standing height
        standing_pos = vis_q[:3]
        rf_traversable_region.update_origin_pose(standing_pos)
        self.assertEqual(True, True)

        #
        # torso
        #
        torso_traversable_region = FrameTraversableRegion(self.torso_frame_name,
                                                          b_visualize_reach=b_visualize,
                                                          b_visualize_safe=b_visualize,
                                                          visualizer=visualizer)
        self.assertEqual(True, True)

        traversable_regions = [torso_traversable_region, lf_traversable_region, rf_traversable_region]
        frame_planner = LocomanipulationFramePlanner(traversable_regions,
                                                     self.ee_offsets_path)
        frame_planner.debug_sample_points(visualizer, self.torso_frame_name)
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
