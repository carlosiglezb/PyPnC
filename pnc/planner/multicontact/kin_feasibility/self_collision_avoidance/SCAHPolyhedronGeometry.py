import numpy as np
from pydrake.geometry.optimization import HPolyhedron
import pinocchio as pin
import hppfcl
from scipy.spatial import HalfspaceIntersection, ConvexHull

class SCAHPolyhedronGeometry:
    def __init__(self,
                 obstacles: list[HPolyhedron],
                 obstacle_names: list[str]):

        self._geometry_primitives = {}

        for name, poly in zip(obstacle_names, obstacles):
            # 1. Calculate the Chebyshev Center (the center of the polytope)
            # This solves a linear program to find the largest inscribed ball
            center = poly.ChebyshevCenter().reshape(-1, 1)

            # 2. Extract original A and b
            A_orig = poly.A()
            b_orig = poly.b().reshape(-1, 1)

            # 3. Transform b so the origin is at the center
            # New constraint: A * (x_local + center) <= b_orig
            # A * x_local <= b_orig - A * center
            b_centered = b_orig - (A_orig @ center)

            # 4. Store the centered representation
            params = {
                'A': A_orig,
                'b': b_centered,
                'polytope_origin': center,  # The old global center is now the "origin"
                'polytope_rotation': np.eye(3)
            }
            self._geometry_primitives[name] = params

    def get_box_representation(self, link_name: str) -> dict[str, np.ndarray]:
        """Returns the A and b matrices for the named obstacle."""
        return self._geometry_primitives.get(link_name)

    def get_primitive_shape_type(self, link_name: str) -> str:
        """
        Since HPolyhedrons are general polytopes, we treat them as 'box'
        (polytope) types for SCA compatibility.
        """
        if link_name in self._geometry_primitives:
            return 'box'
        return 'unknown'

    def is_link_in_sca_list(self, link_name: str) -> bool:
        """Checks if the obstacle name exists in our geometry set."""
        return link_name in self._geometry_primitives

    def get_shape_origin(self, link_name: str) -> np.ndarray:
        return self._geometry_primitives[link_name]['polytope_origin']

    def get_shape_rotation(self, link_name: str) -> np.ndarray:
        return self._geometry_primitives[link_name]['polytope_rotation']

    def get_geometry_primitive_names(self) -> list[str]:
            """Return the names of stored geometry primitives."""
            return list(self._geometry_primitives.keys())

    @classmethod
    def from_wall_scene(cls, hole_scene_instance):
        """
        Factory method to instantiate using the HoleInWallObstructed class.
        """
        names = [
            "floor", "bottom", "knee_knocker_lwall",
            "knee_knocker_rwall", "knee_knocker_top", "triangle_obstacle"
        ]
        return cls(hole_scene_instance.obstacles, names)