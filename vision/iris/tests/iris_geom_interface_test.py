import unittest

import numpy as np
import meshcat

from vision.iris.iris_geom_interface import *
from vision.iris.iris_regions_manager import IrisRegionsManager

sphere_viz = Sphere(0.05)


class IrisGeomInterfaceTest(unittest.TestCase):
    def test_cube_obstacle(self):
        dom_lb = np.array([-2, -2, -2])
        dom_ub = np.array([2, 2, 2])
        obstacles = [mut.HPolyhedron.MakeUnitBox(3)]
        domain = mut.HPolyhedron.MakeBox(dom_lb, dom_ub)
        starting_pos = np.array([1.2, 1.4, 1.5])

        # create safe region using IRIS
        safe_region = IrisGeomInterface(obstacles, domain, starting_pos)
        safe_region.computeIris()

        # visualize IRIS region
        vis = meshcat.Visualizer()

        # default settings
        vis.open()
        vis.wait()
        vis["/Background"].set_property("visible", False)

        safe_region.visualize(vis)

        # test if random points are inside of safe region
        n_dim = 3
        n_samples = 50
        tp_lst = [np.zeros(n_dim)] * n_samples
        for i in range(n_samples):
            tp_lst[i] = np.random.uniform(low=-2, high=2, size=n_dim)
        tp_in_iris = [safe_region.isPointSafe(tp) for tp in tp_lst]

        # Test particles
        for i, tp in enumerate(tp_lst):
            if tp_in_iris[i]:
                vis[f"tp/{i}"].set_object(sphere_viz, meshcat_safe_obj())
            else:
                vis[f"tp/{i}"].set_object(sphere_viz, meshcat_collision_obj())
            vis[f"tp/{i}"].set_transform(tf.translation_matrix(tp))

        self.assertTrue(True, "Check visualization")

    def test_knee_knocker_obstacle(self):
        dom_lb = np.array([-1.6, -0.8, -0.])
        dom_ub = np.array([1.6, 0.8, 2.1])
        floor = mut.HPolyhedron.MakeBox(
                                np.array([-2, -0.9, -0.05]),
                                np.array([2, 0.9, -0.001]))
        knee_knocker_base = mut.HPolyhedron.MakeBox(
                                    np.array([-0.035, -0.9, 0.0]),
                                    np.array([0.035, 0.9, 0.4]))
        knee_knocker_lwall = mut.HPolyhedron.MakeBox(
                                    np.array([-0.035, 0.9-0.518, 0.0]),
                                    np.array([0.035, 0.9, 2.2]))
        knee_knocker_rwall = mut.HPolyhedron.MakeBox(
                                    np.array([-0.035, -0.9, 0.0]),
                                    np.array([0.035, -(0.9-0.518), 2.2]))
        knee_knocker_top = mut.HPolyhedron.MakeBox(
                                    np.array([-0.035, -0.9, 1.85]),
                                    np.array([0.035, 0.9, 2.2]))
        obstacles = [floor,
                     knee_knocker_base,
                     knee_knocker_lwall,
                     knee_knocker_rwall,
                     knee_knocker_top]
        domain = mut.HPolyhedron.MakeBox(dom_lb, dom_ub)
        starting_pos = np.array([-0.2, -0.1, 0.001])
        ending_pos = np.array([0.2, -0.1, 0.001])

        # create safe region using IRIS
        safe_start_region = IrisGeomInterface(obstacles, domain, starting_pos)
        safe_end_region = IrisGeomInterface(obstacles, domain, ending_pos)
        safe_regions_mgr = IrisRegionsManager(safe_start_region, safe_end_region)
        safe_regions_mgr.computeIris()

        # if start-to-end regions not connected, sample points in between
        if not safe_regions_mgr.areIrisSeedsContained():
            safe_regions_mgr.connectIrisSeeds()

        # visualize IRIS region
        vis = meshcat.Visualizer()

        # open Visualizer and use default settings
        vis.open()
        vis.wait()
        vis["/Background"].set_property("visible", False)

        # Visualize IRIS regions for "start" and "end" seeds
        safe_regions_mgr.visualize(vis)

        # test if random points are inside of safe region
        n_dim = 3
        n_samples = 50
        tp_lst = [np.zeros(n_dim)] * n_samples
        for i in range(n_samples):
            tp_lst[i] = np.random.uniform(low=dom_lb, high=dom_ub, size=n_dim)
        tp_in_iris_start = [safe_start_region.isPointSafe(tp) for tp in tp_lst]
        tp_in_iris_end = [safe_end_region.isPointSafe(tp) for tp in tp_lst]

        # Test particles
        for i, tp in enumerate(tp_lst):
            if tp_in_iris_start[i] or tp_in_iris_end[i]:
                vis[f"tp/{i}"].set_object(sphere_viz, meshcat_safe_obj())
            else:
                vis[f"tp/{i}"].set_object(sphere_viz, meshcat_collision_obj())
            vis[f"tp/{i}"].set_transform(tf.translation_matrix(tp))

        # hide clutter
        vis["tp"].set_property("visible", False)
        vis["iris/0"].set_property("visible", False)
        vis["iris/1"].set_property("visible", False)
        self.assertTrue(True, "Check visualization")


if __name__ == '__main__':
    unittest.main()
