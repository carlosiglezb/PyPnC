import numpy as np

# Convex Hull description
from scipy.spatial import ConvexHull
from pypoman import compute_polytope_vertices
from meshcat.geometry import TriangularMeshGeometry
from pydrake.geometry.optimization import HPolyhedron
import hppfcl as fcl
import pinocchio as pin


def pydrake_geom_to_meshcat(mut_polyhedron: HPolyhedron):
    poly_A = mut_polyhedron.A()
    poly_b = mut_polyhedron.b()
    poly_vertices = compute_polytope_vertices(poly_A, poly_b)
    # poly_chull = ConvexHull(poly_vertices, qhull_options='QbB')
    poly_chull = ConvexHull(poly_vertices, qhull_options='QJ')    # qhull_options='QJ', 'QbB', 'QR0', 'Qs', 'En'
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

    # Validation and H-rep to V-rep

    # In a real environment, you would use a library like pypoman here:
    A = drake_hpoly.A()
    b = drake_hpoly.b()
    hpoly_vertices = compute_polytope_vertices(A, b)

    # Compute the Convex Hull to get the Faces (Triangulation)
    hull = ConvexHull(hpoly_vertices, qhull_options='QJ')    # qhull_options='QJ', 'QbB', 'QR0', 'Qs', 'En'
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
