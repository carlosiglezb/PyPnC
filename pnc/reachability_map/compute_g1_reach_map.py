"""
Constructs the reachable space of the end effectors of G1
assuming the base is fixed. The reachable space is randomly sampled,
then a convex hull is created using these points, and finally
the polytopes are simplified using a decimation process.

This follows more or less the approach from
Tonneau, Steve, et al. "An efficient acyclic contact planner for
multiped robots." IEEE Transactions on Robotics 34.3 (2018): 586-601.
"""
import os
import sys
import numpy as np
import open3d as o3d
from collections import OrderedDict

cwd = os.getcwd()
sys.path.append(cwd)

# Robot model libraries
from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem

# YAML
from ruamel.yaml import YAML
from copy import deepcopy

b_visualize_hulls = False
decimation_triangles = 30
N_samples = 10000


def generate_random_joint_dict(joint_id):
    q_rand = np.random.uniform(low=q_llim, high=q_ulim, size=nq)
    for k, v in joint_id.items():
        q_dict[k] = q_rand[v]
        zero_qdot_dict[k] = 0.


def generate_t_pose_joint_dict(joint_id):
    for k, v in joint_id.items():
        if k == 'left_shoulder_pitch_link':
            q_t_pose[k] = np.pi/2.
        elif k == 'right_shoulder_pitch_link':
            q_t_pose[k] = -np.pi/2.
        else:
            q_t_pose[k] = 0.


# Load Draco3 robot model
robot = PinocchioRobotSystem(
                cwd + "/robot_model/g1_description/g1.urdf",
                cwd + "/robot_model/g1_description", b_fixed_base=True)

joint_id = robot.joint_id
nq = robot.n_q
q_dict, q_t_pose = OrderedDict(), OrderedDict()
zero_qdot_dict = OrderedDict()
q_lims = robot.joint_pos_limit
q_llim, q_ulim = q_lims[:, 0],  q_lims[:, 1]

# Loop to generate end effector points
reach_space = OrderedDict()
# ee_links = OrderedDict()
end_effectors = ['LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']

# change below according to the robot model
connected_frames_list = []
end_effector_names = ['left_ankle_roll_link',
                      'right_ankle_roll_link',
                      'left_knee_link',
                      'right_knee_link',
                      'left_palm_link',
                      'right_palm_link']
connected_frames_list.append(
                    {'parent_frame': 'left_knee_link',
                     'child_frame': 'left_ankle_roll_link',
                     'length': 0.})    # <-- this will be computed and saved later
connected_frames_list.append(
                    {'parent_frame': 'right_knee_link',
                     'child_frame': 'right_ankle_roll_link',
                     'length': 0.})    # <-- this will be computed and saved later

for ee in end_effectors:
    reach_space[ee] = np.zeros((N_samples, 3))      # [x,y,z] for all samples

for n in range(N_samples):

    # generate new random state, q_dict
    generate_random_joint_dict(joint_id)

    # update robot model and get corresponding end effector position
    robot.update_system(None, None, None, None, None, None, None, None,
                        q_dict, zero_qdot_dict)
    for ee, ee_link in zip(end_effectors, end_effector_names):
        reach_space[ee][n] = robot.get_link_iso(ee_link)[:3, 3]

    # Print progress every 10% increment
    if (n != 0) and n % (0.1*N_samples) == 0:
        print(f'Reachability space generation complete: {(n/N_samples)*100}%')

# Define path to save convex hull
save_loc = cwd + '/pnc/reachability_map/output/g1/'

# get zero position of robot to get reachable space of torso
generate_t_pose_joint_dict(q_dict)
robot.update_system(None, None, None, None,
                    None, None, None, None,
                    q_t_pose, zero_qdot_dict)

# Create convex hulls and save plane equation coefficients
yaml = YAML()
plane_dict = {'a': 0., 'b': 0., 'c': 0., 'd': 0.}
mesh_simplified = OrderedDict()
for ee_name, rs in reach_space.items():
    plane_coeffs_list = []

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(rs)
    hull, _ = pcl.compute_convex_hull()

    # Decimation
    mesh_simplified[ee_name] = hull.simplify_quadric_decimation(decimation_triangles)
    if b_visualize_hulls:
        o3d.visualization.draw_geometries([mesh_simplified[ee_name]],
                                          mesh_show_wireframe=True)

    # compute triangle normals and save meshes as STL
    filename = 'g1_' + ee_name
    mesh_simplified[ee_name] = o3d.geometry.TriangleMesh.\
        compute_triangle_normals(mesh_simplified[ee_name])
    o3d.io.write_triangle_mesh(save_loc + filename + '.stl', mesh_simplified[ee_name])

    # get plane information
    triangles = np.asarray(mesh_simplified[ee_name].triangles)
    normals = np.asarray(mesh_simplified[ee_name].triangle_normals)
    vertices = np.asarray(mesh_simplified[ee_name].vertices)

    # construct equation coefficients (offset was missing)
    for tr in range(len(triangles)):
        coeffs = np.zeros((4,))
        coeffs[:3] = normals[tr]
        x_plane = vertices[triangles[tr][0]]        # [x,y,z] point at vertex of triangle/plane
        coeffs[3] = -coeffs[:3] @ x_plane
        plane_dict['a'], plane_dict['b'] = float(coeffs[0]), float(coeffs[1])
        plane_dict['c'], plane_dict['d'] = float(coeffs[2]), float(coeffs[3])
        plane_coeffs_list.append(deepcopy(plane_dict))

    with open(save_loc + filename + '.yaml', 'w') as f:
        yaml.dump(plane_coeffs_list, f)

p_ee_offsets_dict = {}
xyz_offset_dict = {'x': 0., 'y': 0., 'z': 0.}
# save reachability of torso w.r.t. end effector
for (ee_name, rs), contact_name in zip(reach_space.items(), end_effector_names):
    xyz_offset_dict['x'] = float(robot.get_link_iso(contact_name)[0, 3])
    xyz_offset_dict['y'] = float(robot.get_link_iso(contact_name)[1, 3])
    xyz_offset_dict['z'] = float(robot.get_link_iso(contact_name)[2, 3])
    p_ee_offsets_dict[ee_name] = deepcopy(xyz_offset_dict)

with open(save_loc + 'g1_ee_offsets.yaml', 'w') as f:
    yaml.dump(p_ee_offsets_dict, f)

# save auxiliary frames
it = 0
for con_frame in connected_frames_list:
    parent_frame = con_frame['parent_frame']
    child_frame = con_frame['child_frame']
    offset = robot.get_link_iso(parent_frame)[:3, -1] - robot.get_link_iso(child_frame)[:3, -1]
    connected_frames_list[it]['length'] = float(np.linalg.norm(offset))
    it += 1

with open(save_loc + 'g1_aux_frames.yaml', 'w') as f:
    yaml.dump(connected_frames_list, f)

