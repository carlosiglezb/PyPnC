import os
import sys

import numpy as np

cwd = os.getcwd()
sys.path.append(cwd)

from pinocchio.visualize import MeshcatVisualizer
import pinocchio as pin

import meshcat.geometry as g
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

# Display Robot in Meshcat Visualizer
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    cwd + "/robot_model/draco3/draco3_gripper_mesh_updated.urdf",
    cwd + "/robot_model/draco3", pin.JointModelFreeFlyer())
viz = MeshcatVisualizer(model, collision_model, visual_model)
try:
    viz.initViewer(open=True)
    viz.viewer.wait()
except ImportError as err:
    print(
        "Error while initializing the viewer. It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)
viz.loadViewerModel(rootNodeName="draco3")
vis_q = pin.neutral(model)
viz.display(vis_q)

# Display Reachable spaces
ee_list = ['LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']
for ee_name in ee_list:
    meshPath = cwd + "/pnc/reachability_map/output/draco3_" + ee_name + ".stl"
    filename = os.path.join(meshPath)
    obj = g.Mesh(g.StlMeshGeometry.from_file(filename))
    obj.material.transparent = True
    obj.material.opacity = 0.6
    viz.viewer["reachability"][ee_name].set_object(obj)

# Display convex hull (in matplotlib) using plane equations
plane_eqn_path = cwd + '/pnc/reachability_map/output/draco3_'
for ee_name in ee_list:
    filename = ee_name + '.yaml'
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # get the right limits for each end effector
    if ee_name == 'LF':
        x, y = np.meshgrid(np.linspace(-0.6, 0.8), np.linspace(-0.4, 0.7))
        z_ulim = 0.1
        z_llim = -1.0
    elif ee_name == 'RF':
        x, y = np.meshgrid(np.linspace(-0.6, 0.8), np.linspace(-0.7, 0.4))
        z_ulim = 0.1
        z_llim = -1.0
    elif ee_name == 'LH':
        x, y = np.meshgrid(np.linspace(-0.6, 0.8), np.linspace(-0.6, 1.0))
        z_ulim = 1.0
        z_llim = -0.1
    elif ee_name == 'RH':
        x, y = np.meshgrid(np.linspace(-0.6, 0.8), np.linspace(-1.0, 0.6))
        z_ulim = 1.0
        z_llim = -0.1
    else:
        print(f'Frame {ee_name} does not have specified limits for plotting in matplotlib')

    with open(plane_eqn_path + filename, 'r') as f:
        yml = YAML().load(f)
        for i_plane in range(len(yml)):
            a, b = yml[i_plane]['a'], yml[i_plane]['b']
            c, d = yml[i_plane]['c'], yml[i_plane]['d']
            z = (1/c) * (-d - a*x - b*y)
            z[z > z_ulim] = np.nan
            z[z < z_llim] = np.nan
            ax.scatter(x, y, z, marker='.', alpha=0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    plt.show()
