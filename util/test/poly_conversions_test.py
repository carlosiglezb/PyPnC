import unittest

import numpy as np
import hppfcl as fcl
from pinocchio import StdVec_Vector3
from pydrake.geometry.optimization import HPolyhedron
from scipy.spatial import ConvexHull

from util.environment_creator import TiltedStairs
from util.pydrake_meshcat_interface import hpoly_to_fcl_collision


class TestDrakeToCoalConversions(unittest.TestCase):
    def test_hpoly_to_fcl(self):
        # example usage:
        hpoly_vertices = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]
        ])
        hull = ConvexHull(hpoly_vertices)
        fcl_faces_indices = hull.simplices

        triangles_lst = fcl.StdVec_Triangle()
        for indices in fcl_faces_indices:
            i1, i2, i3 = indices
            triangle = fcl.Triangle(int(i1), int(i2), int(i3))
            triangles_lst.append(triangle)

        vertices_list = fcl.StdVec_Vec3s()
        for v in hpoly_vertices:
            vertices_list.append(v)

        convex_geometry = fcl.Convex(
            vertices_list,
            triangles_lst
        )

        self.assertEqual(True, False)  # add assertion here

    def test_hpoly_to_fcl_collision(self):
        stairs = TiltedStairs()
        for col_obj in stairs.obstacles:
            if isinstance(col_obj, HPolyhedron):
                obstacle_geom = hpoly_to_fcl_collision(col_obj)
            else:
                raise NotImplementedError("Only HPolyhedron obstacles are supported for stairs environment")

        self.assertEqual(True, True)  # add assertion here

if __name__ == '__main__':
    unittest.main()
