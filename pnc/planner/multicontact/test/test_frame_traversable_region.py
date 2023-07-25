import unittest

import util.util
from pnc.planner.multicontact.frame_traversable_region import FrameTraversableRegion

import os
import sys
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin

cwd = os.getcwd()
sys.path.append(cwd)


class TestFrameTraversableRegion(unittest.TestCase):
    def setUp(self):
        self.torso_frame_name = 'torso'
        self.rf_frame_name = 'RF'
        self.rf_frame_stl_path = cwd + '/pnc/reachability_map/output/draco3_' + \
                                 self.rf_frame_name + '.stl'
        self.rf_poly_halfspace_path = cwd + '/pnc/reachability_map/output/draco3_' + \
                                      self.rf_frame_name + '.yaml'

        self.lf_frame_name = 'LF'
        self.lf_frame_stl_path = cwd + '/pnc/reachability_map/output/draco3_' + \
                                 self.lf_frame_name + '.stl'
        self.lf_poly_halfspace_path = cwd + '/pnc/reachability_map/output/draco3_' + \
                                      self.lf_frame_name + '.yaml'

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
        q0[11] = 0.  # l_wrist_ps
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
        q0[29] = 0.  # r_wrist_ps
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
        R = util.util.euler_to_rot([0., 0., np.pi/2.])
        quat = util.util.rot_to_quat(R)
        return np.concatenate((pos, quat))

    def get_sample_collision_free_boxes(self):
        L_lf = np.array([
            [-0.1, -0.1, 0.0],  # prevent leg-crossing
            [-0.1, -0.1, 0.4],  # prevent leg-crossing
            [0.2, -0.1, 0.0]  # prevent leg-crossing
        ])
        L_rf = np.array([
            [-0.1, -0.3, 0.0],  # prevent leg-crossing
            [-0.1, -0.3, 0.4],  # prevent leg-crossing
            [0.2, -0.3, 0.0]  # prevent leg-crossing
        ])
        L_lh = np.array([
            [-0.1, -0.1, 0.7],  # prevent leg-crossing
            [-0.1, 0.0, 0.7],  # prevent leg-crossing
            [0.2, -0.1, 0.7]  # prevent leg-crossing
        ])
        L_rh = np.array([
            [-0.1, -0.3, 0.7],  # prevent leg-crossing
            [-0.1, -0.3, 0.7],  # prevent leg-crossing
            [0.2, -0.3, 0.7]  # prevent leg-crossing
        ])

        # upper bounds of the safe boxes
        U_lf = np.array([
            [0.1, 0.3, 0.6],  # z stops at kin. limit
            [0.4, 0.3, 0.6],  # x stops at kin. limit
            [0.5, 0.3, 0.6]  # x stops at kin. limit
        ])
        U_rf = np.array([
            [0.1, 0.1, 0.6],  # prevent leg-crossing
            [0.4, 0.1, 0.6],  # prevent leg-crossing
            [0.5, 0.1, 0.6]  # prevent leg-crossing
        ])
        U_lh = np.array([
            [0.1, 0.3, 1.3],  # prevent leg-crossing
            [0.4, 0.3, 1.3],  # prevent leg-crossing
            [0.5, 0.3, 1.3]  # prevent leg-crossing
        ])
        U_rh = np.array([
            [0.1, 0.1, 1.3],  # prevent leg-crossing
            [0.4, 0.0, 1.3],  # prevent leg-crossing
            [0.5, 0.1, 1.3]  # prevent leg-crossing
        ])
        box_llim=[L_lf, L_rf, L_lh, L_rh]
        box_ulim=[U_lf, U_rf, U_lh, U_rh]
        return box_llim, box_ulim

    def get_navy_door_collision_free_boxes(self):
        # lower bounds of end-effectors safe boxes
        L_lf = np.array([
            [-0.2, 0.0, 0.0],  # prevent leg-crossing
            [-0.1, 0.0, 0.4],  # prevent leg-crossing
            [0.4, 0.0, 0.0]  # prevent leg-crossing
        ])
        L_rf = np.array([
            [-0.2, -0.4, 0.0],  # prevent leg-crossing
            [-0.1, -0.35, 0.4],  # prevent leg-crossing
            [0.4, -0.4, 0.0]  # prevent leg-crossing
        ])
        L_lh = np.array([
            [-0.2, 0.0, 0.7],  # prevent leg-crossing
            [-0.1, 0.0, 0.7],  # prevent leg-crossing
            [0.4, 0.0, 0.7]  # prevent leg-crossing
        ])
        L_rh = np.array([
            [-0.2, -0.4, 0.7],  # prevent leg-crossing
            [-0.1, -0.4, 0.7],  # prevent leg-crossing
            [0.4, -0.4, 0.7]  # prevent leg-crossing
        ])
        L_torso = np.array([
            [-0.2, -0.3, 0.5],  # prevent leg-crossing
            [-0.1, -0.3, 0.5],  # prevent leg-crossing
            [0.1, -0.3, 0.5]  # prevent leg-crossing
        ])

        # upper bounds of the safe boxes
        U_lf = np.array([
            [0.25, 0.35, 0.6],  # z stops at kin. limit
            [0.6, 0.4, 0.6],  # x stops at kin. limit
            [0.6, 0.4, 0.6]  # x stops at kin. limit
        ])
        U_rf = np.array([
            [0.25, 0.0, 0.6],  # prevent leg-crossing
            [0.6, 0.0, 0.6],  # prevent leg-crossing
            [0.6, 0.0, 0.6]  # prevent leg-crossing
        ])
        U_lh = np.array([
            [0.25, 0.4, 1.3],  # prevent leg-crossing
            [0.6, 0.4, 1.3],  # prevent leg-crossing
            [0.6, 0.4, 1.3]  # prevent leg-crossing
        ])
        U_rh = np.array([
            [0.25, 0.0, 1.3],  # prevent leg-crossing
            [0.6, 0.0, 1.3],  # prevent leg-crossing
            [0.6, 0.0, 1.3]  # prevent leg-crossing
        ])
        U_torso = np.array([
            [0.2, 0.3, 0.9],  # prevent leg-crossing
            [0.4, 0.3, 0.9],  # prevent leg-crossing
            [0.6, 0.3, 0.9]  # prevent leg-crossing
        ])
        box_llim=[L_torso, L_lf, L_rf, L_lh, L_rh]
        box_ulim=[U_torso, U_lf, U_rf, U_lh, U_rh]
        return box_llim, box_ulim

    def test_visualizing_reachable_region(self):
        test_region = FrameTraversableRegion(self.rf_frame_name,
                                             self.rf_frame_stl_path, self.rf_poly_halfspace_path,
                                             b_visualize=True)

        # translate to new origin
        new_origin = np.array([0., 0., 0.6])
        test_region.update_origin_pose(new_origin)
        self.assertEqual(True, True)

    def test_visualize_region_with_robot_model(self):
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

        test_region = FrameTraversableRegion(self.rf_frame_name,
                                             self.rf_frame_stl_path, self.rf_poly_halfspace_path,
                                             visualizer=visualizer)
        self.assertEqual(True, True)

        # move to standing height
        standing_pos = np.array([0., 0., 0.6])
        vis_q[:3] = standing_pos
        visualizer.display(vis_q)
        test_region.update_origin_pose(standing_pos)
        self.assertEqual(True, True)

    def test_visualize_region_and_ee_paths(self):
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

        test_region = FrameTraversableRegion(self.rf_frame_name,
                                             self.rf_frame_stl_path, self.rf_poly_halfspace_path,
                                             visualizer=visualizer)
        self.assertEqual(True, True)

        # move to standing height
        standing_pos = vis_q[:3]
        test_region.update_origin_pose(standing_pos)
        self.assertEqual(True, True)

        # collision-free boxes
        box_llim, box_ulim = self.get_sample_collision_free_boxes()
        test_region.load_collision_free_boxes(box_llim[1], box_ulim[1])

    def test_visualize_navy_door_paths(self):
        # load robot
        rob_model, rob_collision_model, rob_visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
            cwd + "/robot_model/draco3", pin.JointModelFreeFlyer())
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
        vis_q = self.get_draco3_default_initial_pose()
        visualizer.display(vis_q)

        # load door to visualizer
        door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
            cwd + "/robot_model/ground/navy_door.urdf",
            cwd + "/robot_model/ground", pin.JointModelFreeFlyer())
        door_vis = MeshcatVisualizer(door_model, door_collision_model, door_visual_model)
        door_vis.initViewer(visualizer.viewer)
        door_vis.loadViewerModel(rootNodeName="door")
        door_vis_q = self.get_navy_door_default_initial_pose()
        door_vis.display(door_vis_q)

        #
        # right foot
        #
        rf_traversable_region = FrameTraversableRegion(self.rf_frame_name,
                                             self.rf_frame_stl_path, self.rf_poly_halfspace_path,
                                             visualizer=visualizer)
        self.assertEqual(True, True)

        # move to standing height
        standing_pos = vis_q[:3]
        rf_traversable_region.update_origin_pose(standing_pos)
        self.assertEqual(True, True)

        # collision-free boxes
        box_llim, box_ulim = self.get_navy_door_collision_free_boxes()
        rf_traversable_region.load_collision_free_boxes(box_llim[2], box_ulim[2])

        #
        # torso
        #
        torso_traversable_region = FrameTraversableRegion(self.torso_frame_name, visualizer=visualizer)
        self.assertEqual(True, True)

        # collision-free boxes
        box_llim, box_ulim = self.get_navy_door_collision_free_boxes()
        torso_traversable_region.load_collision_free_boxes(box_llim[0], box_ulim[0])


if __name__ == '__main__':
    unittest.main()
