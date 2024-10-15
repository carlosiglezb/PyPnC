import copy
import unittest

import os, sys

from pnc.planner.multicontact.self_collision_avoidance.self_collision_checker import SelfCollisionChecker

cwd = os.getcwd()
sys.path.append(cwd)

import pinocchio as pin

B_VISUALIZE = False

class TestCollisionConfigurations(unittest.TestCase):
    def test_g1_configurations_in_collision(self):
        # Initialize self-collision checker
        robot_model_path = cwd + "/robot_model/g1_description/"
        urdf_path = robot_model_path + "g1_simple_collisions.urdf"
        srdf_path = robot_model_path + "config/g1.srdf"

        self_collision_checker = SelfCollisionChecker(robot_model_path,
                                                      urdf_path,
                                                      srdf_path)
        robot, geom_model = self_collision_checker.get_rob_geom_models()
        q_current = copy.copy(robot.q0)

        if B_VISUALIZE:
            viz = pin.visualize.MeshcatVisualizer(
                robot.model, geom_model, robot.visual_model
            )
            robot.setVisualizer(viz, init=False)
            viz.initViewer(open=True)
            viz.loadViewerModel(rootNodeName="g1")
            viz.display_collisions = True
            viz.display(q_current)

        # Check for collisions in zero configuration position
        b_collision_detected = self_collision_checker.check_collisions(q_current)

        # no collisions in zero configuration position
        self.assertFalse(b_collision_detected)

        # Check for collisions when right arm collides (inwards) with torso
        rshoulder_aaId = robot.model.getJointId('right_shoulder_roll_joint')
        q_current[rshoulder_aaId + 7 - 2] = 0.3
        if B_VISUALIZE:
            viz.display(q_current)
        # Check for collisions in zero configuration position
        b_collision_detected = self_collision_checker.check_collisions(q_current)
        self.assertTrue(b_collision_detected)

        # check for no collision if moving arm in opposite direction (outwards)
        q_current[rshoulder_aaId + 7 - 2] = -0.3
        if B_VISUALIZE:
            viz.display(q_current)
        # Check for collisions in zero configuration position
        b_collision_detected = self_collision_checker.check_collisions(q_current)
        self.assertFalse(b_collision_detected)



if __name__ == '__main__':
    unittest.main()
