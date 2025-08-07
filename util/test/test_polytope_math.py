import unittest

import numpy as np
from util.polytope_math import get_closest_distance_to_polytope_surface


class TestPointToPolytopeDistance(unittest.TestCase):

    def test_point_inside_2d_square(self):
        # Define a square: -1 <= x <= 1, -1 <= y <= 1
        # A x <= b
        # x >= -1  => -x <= 1
        # x <= 1
        # y >= -1  => -y <= 1
        # y <= 1
        A = np.array([
            [1, 0],  # x <= 1
            [-1, 0],  # -x <= 1  => x >= -1
            [0, 1],  # y <= 1
            [0, -1]  # -y <= 1  => y >= -1
        ])
        b = np.array([1, 1, 1, 1])

        point = np.array([0.5, 0.5])  # Inside the square
        expected_distance = 0.5  # Closest to x=1 or y=1
        distance = get_closest_distance_to_polytope_surface(point, A, b)
        assert np.isclose(distance, expected_distance, atol=1e-6)
        print(f"Test 'point_inside_2d_square' passed. Distance: {distance}")

    def test_point_outside_2d_square(self):
        # Same square as above
        A = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
        ])
        b = np.array([1, 1, 1, 1])

        point = np.array([2.0, 0.0])  # Outside the square, closest to x=1
        expected_distance = 1.0
        distance = get_closest_distance_to_polytope_surface(point, A, b)
        assert np.isclose(distance, expected_distance, atol=1e-6)
        print(f"Test 'point_outside_2d_square' passed. Distance: {distance}")

    def test_point_on_surface_2d_square(self):
        # Same square as above
        A = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
        ])
        b = np.array([1, 1, 1, 1])

        point = np.array([1.0, 0.5])  # On the surface (right edge)
        expected_distance = 0.0
        distance = get_closest_distance_to_polytope_surface(point, A, b)
        assert np.isclose(distance, expected_distance, atol=1e-6)
        print(f"Test 'point_on_surface_2d_square' passed. Distance: {distance}")

    def test_point_at_vertex_2d_square(self):
        # Same square as above
        A = np.array([
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1]
        ])
        b = np.array([1, 1, 1, 1])

        point = np.array([1.0, 1.0])  # At a vertex
        expected_distance = 0.0
        distance = get_closest_distance_to_polytope_surface(point, A, b)
        assert np.isclose(distance, expected_distance, atol=1e-6)
        print(f"Test 'point_at_vertex_2d_square' passed. Distance: {distance}")

    def test_point_inside_3d_cube(self):
        # Define a cube: -1 <= x,y,z <= 1
        A = np.array([
            [1, 0, 0],  # x <= 1
            [-1, 0, 0],  # -x <= 1
            [0, 1, 0],  # y <= 1
            [0, -1, 0],  # -y <= 1
            [0, 0, 1],  # z <= 1
            [0, 0, -1]  # -z <= 1
        ])
        b = np.array([1, 1, 1, 1, 1, 1])

        point = np.array([0.2, -0.3, 0.7])  # Inside the cube
        expected_distance = 1.0 - 0.7  # Closest to z=1 face
        distance = get_closest_distance_to_polytope_surface(point, A, b)
        assert np.isclose(distance, expected_distance, atol=1e-6)
        print(f"Test 'point_inside_3d_cube' passed. Distance: {distance}")

    def test_point_outside_3d_cube_corner(self):
        # Same cube as above
        A = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1]
        ])
        b = np.array([1, 1, 1, 1, 1, 1])

        point = np.array([2.0, 2.0, 2.0])  # Outside, closest to vertex (1,1,1)
        expected_distance = np.sqrt((2 - 1) ** 2 + (2 - 1) ** 2 + (2 - 1) ** 2)  # Distance to (1,1,1)
        distance = get_closest_distance_to_polytope_surface(point, A, b)
        assert np.isclose(distance, expected_distance, atol=1e-6)
        print(f"Test 'point_outside_3d_cube_corner' passed. Distance: {distance}")

if __name__ == '__main__':
    unittest.main()
