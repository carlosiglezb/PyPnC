import unittest
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
        self.frame_name = 'RF'
        self.frame_stl_path = cwd + '/pnc/reachability_map/output/draco3_' +\
                    self.frame_name + '.stl'
        self.poly_halfspace_path = cwd + '/pnc/reachability_map/output/draco3_' +\
                    self.frame_name + '.yaml'

    def test_visualizing_reachable_region(self):
        test_region = FrameTraversableRegion(self.frame_name,
                              self.frame_stl_path, self.poly_halfspace_path,
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

        test_region = FrameTraversableRegion(self.frame_name,
                              self.frame_stl_path, self.poly_halfspace_path,
                              visualizer=visualizer)
        self.assertEqual(True, True)

        # move to standing height
        standing_pos = np.array([0., 0., 0.6])
        vis_q[:3] = standing_pos
        visualizer.display(vis_q)
        test_region.update_origin_pose(standing_pos)


if __name__ == '__main__':
    unittest.main()
