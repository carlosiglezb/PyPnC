import os, sys
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

import numpy as np

from util import util

cwd = os.getcwd()
sys.path.append(cwd)


def get_navy_door_default_initial_pose(pos):
    # rotates, then translates
    R = util.euler_to_rot([0., 0., np.pi / 2.])
    quat = util.rot_to_quat(R)
    return np.concatenate((pos, quat))


def main():
    robots_names = ['valkyrie', 'ergoCub', 'g1']
    urdf_paths = [cwd + "/robot_model/valkyrie/valkyrie_hands.urdf",
                  cwd + "/robot_model/ergoCub/ergoCub.urdf",
                  cwd + "/robot_model/g1_description/g1.urdf"]
    z_offsets = {'valkyrie': 1.167, 'ergoCub': 0.774, 'g1': 0.75}

    # load (real) door to visualizer
    door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(
        cwd + "/robot_model/ground/navy_door.urdf",
        cwd + "/robot_model/ground", pin.JointModelFreeFlyer())

    viz = MeshcatVisualizer(door_model, door_collision_model, door_visual_model)
    try:
        viz.initViewer(open=True)
        viz.viewer.wait()
    except ImportError as err:
        print(
            "Error while initializing the viewer. It seems you should install Python meshcat"
        )
        print(err)
        sys.exit(0)
    viz.loadViewerModel(rootNodeName="door")
    door_pos = np.array([0., 0., 0.])
    door_vis_q = get_navy_door_default_initial_pose(door_pos)
    viz.display(door_vis_q)

    # load (real) robots to visualizer
    y_offset = 1.5
    for robot_name, urdf_path in zip(robots_names, urdf_paths):
        mesh_path = cwd + "/robot_model/" + robot_name
        if robot_name == 'g1':
            mesh_path += "_description"
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_path, mesh_path, pin.JointModelFreeFlyer())
        rob_viz = MeshcatVisualizer(model, collision_model, visual_model)
        rob_viz.initViewer(viz.viewer)
        rob_viz.loadViewerModel(rootNodeName=robot_name)
        rob_vis_q = pin.neutral(model)

        # for Valkyrie, force elbows down
        if robot_name == 'valkyrie':
            rob_vis_q[7+16] = -np.pi / 2        # "leftShoulderRoll",
            rob_vis_q[7+18] = -np.pi / 2        # "leftElbowPitch",
            rob_vis_q[7+23] = np.pi / 2        # "rightShoulderRoll",
            rob_vis_q[7+25] = np.pi / 2        # "rightElbowPitch",
        rob_vis_q[1] += y_offset
        rob_vis_q[2] = z_offsets[robot_name]
        rob_viz.display(rob_vis_q)
        y_offset += 1.5


if __name__ == "__main__":
    main()
