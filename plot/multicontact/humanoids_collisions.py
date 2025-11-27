import os, sys

import coal
import pinocchio as pin
from meshcat.geometry import MeshLambertMaterial, Cylinder
from pinocchio.visualize import MeshcatVisualizer

import numpy as np
from pinocchio.visualize.meshcat_visualizer import hasMeshFileInfo

from plot.meshcat_utils import MeshcatPinocchioAnimation, coal_geom_to_meshcat
from util import util
from visualizer.meshcat_tools.meshcat_palette import meshcat_obstacle_obj, YELLOW

cwd = os.getcwd()
sys.path.append(cwd)


def get_navy_door_default_initial_pose(pos):
    # rotates, then translates
    R = util.euler_to_rot([0., 0., np.pi / 2.])
    quat = util.rot_to_quat(R)
    return np.concatenate((pos, quat))


def main():
    # robots_names = ['valkyrie', 'ergoCub', 'g1']
    # urdf_paths = [cwd + "/robot_model/valkyrie/valkyrie_hands.urdf",
    #               cwd + "/robot_model/ergoCub/ergoCub.urdf",
    #               cwd + "/robot_model/g1_description/g1.urdf"]
    robots_names = ['g1']
    # urdf_paths = [cwd + "/robot_model/g1_description/g1_29dof_simple_collisions.urdf"]  # g1.urdf
    urdf_paths = [cwd + "/robot_model/g1_description/g1_29dof_lock_waist_modified.urdf"]  # g1.urdf
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
        rob_data, col_data, vis_data = pin.createDatas(model, collision_model, visual_model)

        rob_viz = MeshcatVisualizer(model, collision_model, visual_model)
        rob_viz.initViewer(viz.viewer)
        rob_viz.loadViewerModel(rootNodeName=robot_name)
        rob_viz.display_collisions = True

        # get current joint positions
        rob_vis_q = pin.neutral(model)
        # for Valkyrie, force elbows down
        if robot_name == 'valkyrie':
            rob_vis_q[7 + 16] = -np.pi / 2  # "leftShoulderRoll",
            rob_vis_q[7 + 18] = -np.pi / 2  # "leftElbowPitch",
            rob_vis_q[7 + 23] = np.pi / 2  # "rightShoulderRoll",
            rob_vis_q[7 + 25] = np.pi / 2  # "rightElbowPitch",
        elif robot_name == 'g1':
            rob_vis_q[7 + 14] = np.pi / 6  # left_shoulder_roll_joint
            rob_vis_q[7 + 16] = np.pi / 2  # left_elbow_joint
            rob_vis_q[7 + 21] = -np.pi / 6  # left_shoulder_roll_joint
            rob_vis_q[7 + 23] = np.pi / 2  # left_shoulder_roll_joint
        rob_vis_q[1] += y_offset
        rob_vis_q[2] = z_offsets[robot_name]
        rob_viz.display(rob_vis_q)

        # update kinematics
        pin.forwardKinematics(model, rob_data, rob_vis_q)
        pin.updateGeometryPlacements(model, rob_data, collision_model, col_data)

        # set color of collision shapes
        for visual in collision_model.geometryObjects:
            cylinder_object = coal_geom_to_meshcat(visual.geometry) # cylinder aligned with y-axis
            rob_viz.viewer[f'{robot_name}/collisions'][visual.name].set_object(cylinder_object,
                                                               MeshLambertMaterial(color=YELLOW, opacity=0.4))
            # rob_viz.viewer[robot_name][visual.name].set_transform(visual.placement.homogeneous)

            # Get mesh pose.
            M = col_data.oMg[collision_model.getGeometryId(visual.name)]
            # Manage scaling
            if hasMeshFileInfo(visual):
                scale = np.asarray(visual.meshScale).flatten()
                S = np.diag(np.concatenate((scale, [1.0])))
                # S = visual.placement.homogeneous
                T = np.array(M.homogeneous).dot(S)
            else:
                T = M.homogeneous
            if isinstance(cylinder_object, Cylinder):
                # Correct cylinder alignment from y-axis to z-axis
                R_corr = pin.Quaternion(
                    np.sqrt(2) / 2, -np.sqrt(2) / 2, 0.0, 0.0).toRotationMatrix()
                T_corr = np.eye(4)
                T_corr[0:3, 0:3] = R_corr
                T = T.dot(T_corr)
            # Update viewer configuration.
            rob_viz.viewer[f'{robot_name}/collisions'][visual.name].set_transform(T)

        y_offset += 1.5


if __name__ == "__main__":
    main()
