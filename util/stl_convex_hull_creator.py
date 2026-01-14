import trimesh
import os, sys

cwd = os.getcwd()
sys.path.append(cwd)

def create_convex_hull_stl(input_path, output_path):
    """Loads an STL, computes its convex hull, and saves the result."""
    mesh = trimesh.load_mesh(input_path)
    if not mesh.is_convex:
        # Compute the convex hull
        convex_hull = mesh.convex_hull
        # Save the new convex mesh
        convex_hull.export(output_path)
        print(f"Created convex hull: {output_path}")
    else:
        print(f"Mesh at {input_path} is already convex.")

links_to_make_convex = [
    "torso_link",
    "pelvis_contour_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "left_knee_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]
stl_path = cwd + "/robot_model/g1_description/meshes/"

for link in links_to_make_convex:
    link_stl = link + ".STL"
    link_chull_stl = link + "_chull.STL"
    create_convex_hull_stl(stl_path + link_stl, stl_path + link_chull_stl)
