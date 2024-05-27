# Convex Hull description
from scipy.spatial import ConvexHull
from pypoman import compute_polytope_vertices
from meshcat.geometry import TriangularMeshGeometry
from pydrake.geometry.optimization import HPolyhedron


def pydrake_geom_to_meshcat(mut_polyhedron: HPolyhedron):
    poly_A = mut_polyhedron.A()
    poly_b = mut_polyhedron.b()
    poly_vertices = compute_polytope_vertices(poly_A, poly_b)
    poly_chull = ConvexHull(poly_vertices, qhull_options='QJ')
    return TriangularMeshGeometry(poly_chull.points, poly_chull.simplices)
