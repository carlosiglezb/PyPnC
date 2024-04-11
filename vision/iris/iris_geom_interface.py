import pydrake.geometry.optimization as mut

# Convex Hull description
from scipy.spatial import ConvexHull
from pypoman import compute_polytope_vertices
from meshcat.geometry import TriangularMeshGeometry
import meshcat.transformations as tf
from visualizer.meshcat_tools.meshcat_palette import *


def pydrake_geom_to_meshcat(mut_polyhedron):
    poly_A = mut_polyhedron.A()
    poly_b = mut_polyhedron.b()
    poly_vertices = compute_polytope_vertices(poly_A, poly_b)
    poly_chull = ConvexHull(poly_vertices, qhull_options='QJ')
    return TriangularMeshGeometry(poly_chull.points, poly_chull.simplices)


# TODO: Move obstacles & domain to a separate file to reduce memory size
class IrisGeomInterface:
    def __init__(self, mut_obstacles, mut_domain, seed=None):
        # geometries in PyDrake format
        self.domain_mut = mut_domain        # only one domain specified
        self.obstacles_mut = mut_obstacles  # list of obstacles allowed

        # convert PyDrake geometry to MeshCat format
        self.domain_mcat = pydrake_geom_to_meshcat(mut_domain)
        self.sphere_mcat = Sphere(0.05)
        self.obstacles_mcat_lst = []
        for obs in mut_obstacles:
            self.obstacles_mcat_lst.append(pydrake_geom_to_meshcat(obs))

        # if seed not specified, assume the origin
        if seed is None:
            seed = [0, 0, 0]
        self.seed_pos = seed
        self.iris_region = None
        self.iris_mcat = None

        # default settings
        self.options = mut.IrisOptions()
        self.options.require_sample_point_is_contained = True
        self.options.iteration_limit = 10
        self.options.termination_threshold = 0.05
        self.options.relative_termination_threshold = 0.01

    def computeIris(self):
        self.iris_region = mut.Iris(
            obstacles=self.obstacles_mut, sample=self.seed_pos,
            domain=self.domain_mut, options=self.options)
        self.iris_mcat = pydrake_geom_to_meshcat(self.iris_region)

    def isPointSafe(self, point):
        return self.iris_region.PointInSet(point)

    def irisIntersects(self, other: mut.ConvexSet):
        return self.iris_region.Intersect(other)

    def visualize(self, meshcat_viewer, iris_name=0):
        if self.iris_region is None:
            raise ValueError("IRIS region not computed yet")

        # Domain
        meshcat_viewer["domain"].set_object(self.domain_mcat, meshcat_domain_obj())
        meshcat_viewer["domain"].set_property("visible", False)

        # Obstacle
        for i, obs in enumerate(self.obstacles_mcat_lst):
            meshcat_viewer[f"obstacle/{i}"].set_object(obs, meshcat_obstacle_obj())
        # IRIS region
        meshcat_viewer[f"iris/{iris_name}"].set_object(self.iris_mcat, meshcat_iris_obj())
        # Seed
        meshcat_viewer[f"seed/{iris_name}"].set_object(self.sphere_mcat, meshcat_point_obj())
        meshcat_viewer[f"seed/{iris_name}"].set_transform(tf.translation_matrix(self.seed_pos))

        # default some viewer settings to False to avoid clutter
        meshcat_viewer["obstacle"].set_property("visible", False)
        meshcat_viewer["seed"].set_property("visible", False)

