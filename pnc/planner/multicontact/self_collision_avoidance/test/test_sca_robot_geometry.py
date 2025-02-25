import unittest
import numpy as np
import os

from pnc.planner.multicontact.self_collision_avoidance.sca_robot_geometry import SCARobotGeometry

cwd = os.getcwd()


class ScaRobotGeometryTestCase(unittest.TestCase):
    def test_init(self):
        robot_model_path = cwd + "/robot_model/g1_description/"
        urdf_path = robot_model_path + "g1_cube_collisions.urdf"

        sca_link_names = ['torso', 'left_knee', 'right_knee']
        plan_to_model_frames = {
            'torso': 'torso_link',
            'LF': 'left_ankle_roll_link',
            'RF': 'right_ankle_roll_link',
            'L_knee': 'left_knee_link',
            'R_knee': 'right_knee_link',
            'LH': 'left_palm_link',
            'RH': 'right_palm_link'
        }
        sca_geom = SCARobotGeometry(robot_model_path, urdf_path, plan_to_model_frames)
        torso_rep = sca_geom.get_box_representation('torso')
        lknee_rep = sca_geom.get_box_representation('L_knee')
        rknee_rep = sca_geom.get_box_representation('R_knee')

        expected_torso_half_size = np.array([[0.07], [0.105], [0.175]])
        expected_knee_half_size = np.array([[0.025], [0.025], [0.025]])
        self.assertTrue(np.linalg.norm(torso_rep['A'][:3] - np.eye(3)) < 1e-3)
        self.assertTrue(np.linalg.norm(torso_rep['A'][3:] + np.eye(3)) < 1e-3)
        self.assertTrue(np.linalg.norm(torso_rep['b'][:3] - expected_torso_half_size) < 1e-3)
        self.assertTrue(np.linalg.norm(torso_rep['b'][3:] - expected_torso_half_size) < 1e-3)
        self.assertTrue(np.linalg.norm(lknee_rep['A'][:3] - np.eye(3)) < 1e-3)
        self.assertTrue(np.linalg.norm(lknee_rep['A'][3:] + np.eye(3)) < 1e-3)
        self.assertTrue(np.linalg.norm(lknee_rep['b'][:3] - expected_knee_half_size) < 1e-3)
        self.assertTrue(np.linalg.norm(lknee_rep['b'][3:] - expected_knee_half_size) < 1e-3)
        self.assertTrue(np.linalg.norm(rknee_rep['A'][:3] - np.eye(3)) < 1e-3)
        self.assertTrue(np.linalg.norm(rknee_rep['A'][3:] + np.eye(3)) < 1e-3)
        self.assertTrue(np.linalg.norm(rknee_rep['b'][:3] - expected_knee_half_size) < 1e-3)
        self.assertTrue(np.linalg.norm(rknee_rep['b'][3:] - expected_knee_half_size) < 1e-3)


if __name__ == '__main__':
    unittest.main()
