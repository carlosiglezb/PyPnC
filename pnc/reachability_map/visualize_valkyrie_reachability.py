import os
import sys

from util.polytope_math import extract_plane_eqn_from_coeffs
from util.pydrake_meshcat_interface import pydrake_geom_to_meshcat
from visualizer.meshcat_tools.meshcat_palette import meshcat_reach_obj
from pydrake.geometry.optimization import HPolyhedron
from pinocchio.visualize import MeshcatVisualizer
from ruamel.yaml import YAML
import pinocchio as pin
cwd = os.getcwd()
sys.path.append(cwd)

# Display Robot in Meshcat Visualizer
robot_name = "valkyrie"
model, collision_model, visual_model = pin.buildModelsFromUrdf(
    cwd + "/robot_model/valkyrie/valkyrie_hands.urdf",
    cwd + "/robot_model/valkyrie", pin.JointModelFreeFlyer())
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
viz.loadViewerModel(rootNodeName=robot_name)
vis_q = pin.neutral(model)
viz.display(vis_q)

# Display Reachable spaces
ee_list = ['LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']
for ee_name in ee_list:
    plane_coeffs = []
    meshPath = cwd + "/pnc/reachability_map/output/valkyrie/" + robot_name + "_" + ee_name + ".yaml"

    with open(meshPath, 'r') as f:
        yml = YAML().load(f)
        for i_plane in range(len(yml)):
            plane_coeffs.append(yml[i_plane])

    # load the convex hull
    A, b = extract_plane_eqn_from_coeffs(plane_coeffs)
    polyhedron = HPolyhedron(A, -b)
    obj = pydrake_geom_to_meshcat(polyhedron)
    viz.viewer["reachability"][ee_name].set_object(obj, meshcat_reach_obj())

