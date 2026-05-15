import numpy as np
from itertools import combinations

# Convex Hull description
from scipy.spatial import ConvexHull
from pypoman import compute_polytope_vertices
from meshcat.geometry import TriangularMeshGeometry
from pydrake.geometry.optimization import HPolyhedron
import hppfcl as fcl
import pinocchio as pin


def _vertices_from_hpoly_3d(A: np.ndarray, b: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Enumerate the vertices of a 3D H-polytope {x : Ax <= b} by intersecting
    every triple of bounding planes and keeping the feasible intersections.

    Avoids pycddlib (and its GMP requirement) entirely; uses only numpy.
    Complexity is O(n^3) in the number of halfspaces, which is fine for the
    small obstacle polytopes used here (typically 5-15 faces).
    """
    n_faces = A.shape[0]
    vertices = []
    for i, j, k in combinations(range(n_faces), 3):
        A_sub = A[[i, j, k], :]
        if abs(np.linalg.det(A_sub)) < 1e-10:
            continue  # planes don't meet at a unique point
        x = np.linalg.solve(A_sub, b[[i, j, k]])
        if np.all(A @ x <= b + tol):
            vertices.append(x)
    if len(vertices) < 4:
        raise ValueError("Fewer than 4 feasible vertices found — polytope may be unbounded or degenerate.")
    vertices = np.array(vertices)
    # deduplicate near-coincident vertices
    _, unique_idx = np.unique(np.round(vertices, 7), axis=0, return_index=True)
    return vertices[unique_idx]


def pydrake_geom_to_meshcat(mut_polyhedron: HPolyhedron):
    poly_A = mut_polyhedron.A()
    poly_b = mut_polyhedron.b()
    poly_vertices = _vertices_from_hpoly_3d(poly_A, poly_b)
    poly_chull = ConvexHull(poly_vertices, qhull_options='QJ')
    return TriangularMeshGeometry(poly_chull.points, poly_chull.simplices)

def scipy_hull_to_meshcat(convex_hull: ConvexHull):
    return TriangularMeshGeometry(convex_hull.points, convex_hull.simplices)

def polytope_intersections_to_meshcat(intersections: np.ndarray):
    """
    Convert an array of points to a Meshcat TriangularMeshGeometry.
    """
    poly_chull = ConvexHull(intersections)
    return TriangularMeshGeometry(poly_chull.points, poly_chull.simplices)


def hpoly_to_fcl_collision(drake_hpoly: HPolyhedron) -> fcl.Convex:
    """
    Converts a 3D Drake HPolyhedron (H-representation) into a
    hppfcl/coal CollisionObject (V-representation via Convex).

    Args:
        drake_hpoly: The Drake HPolyhedron object (must be 3D).

    Returns:
        A hppfcl.CollisionObject containing the fcl.Convex geometry.

    Raises:
        ValueError: If the HPolyhedron is not 3-dimensional.
        RuntimeError: If necessary FCL binding classes are not found.
    """

    A = drake_hpoly.A()
    b = drake_hpoly.b()
    hpoly_vertices = _vertices_from_hpoly_3d(A, b)

    # Compute the Convex Hull to get the Faces (Triangulation)
    hull = ConvexHull(hpoly_vertices, qhull_options='QJ')
    fcl_faces_indices = hull.simplices  # M x 3 array of vertex indices

    # Create the required list of 3D Vectors (Vertices)
    try:
        StdVec_Vector3_Type = fcl.StdVec_Vec3s
    except AttributeError:
        raise RuntimeError("FCL binding StdVec_Vec3s not found. Check for equivalent class.")

    vertices_list = StdVec_Vector3_Type()
    for v in hpoly_vertices:
        vertices_list.append(v)

    # Create the required list of `fcl.Triangle` objects (Faces)
    try:
        StdVec_Triangle_Type = fcl.StdVec_Triangle
    except AttributeError:
        raise RuntimeError("FCL binding StdVec_Triangle not found. Check for equivalent class.")

    triangles_list_wrapper = StdVec_Triangle_Type()
    for indices in fcl_faces_indices:
        i1, i2, i3 = indices
        # Construct the fcl.Triangle object using vertex indices
        triangle = fcl.Triangle(int(i1), int(i2), int(i3))
        triangles_list_wrapper.append(triangle)

    # Create the fcl.Convex Geometry Object
    convex_geometry = fcl.Convex(
        vertices_list,
        triangles_list_wrapper
    )

    return convex_geometry

def create_convex_geom_from_copy(pin_geom_to_copy: pin.GeometryObject,
                          geometry: fcl.Convex,
                          name: str) -> pin.GeometryObject:
    """
    Create a new Pinocchio `GeometryObject` by copying an existing one and
    replacing its geometry with an `fcl.Convex` mesh.

    Args:
        pin_geom_to_copy: Source `pin.GeometryObject` to copy metadata from
                          (frame, parent, material, etc.).
        geometry: An `hppfcl.Convex` instance containing the convex mesh data.
        name: The name to assign to the created geometry object.

    Returns:
        A new `pin.GeometryObject` with:
          - `geometry` set to the provided `hppfcl.Convex`,
          - `name` set to `name`,
          - `meshPath` set to `'CONVEX'`,
          - placement reset to identity (zero translation, identity rotation).
    """
    # Copy the input geometry object to preserve non-geometry metadata
    geom_out = pin.GeometryObject.copy(pin_geom_to_copy)

    # Replace geometry with the provided fcl convex mesh
    geom_out.geometry = geometry

    # Set identifying fields
    geom_out.name = name
    geom_out.meshPath = 'CONVEX'

    # Assume the provided geometry is already in the world frame:
    # reset placement to identity to avoid duplicating transforms.
    geom_out.placement.translation = np.zeros(3)
    geom_out.placement.rotation = np.eye(3)
    return geom_out
