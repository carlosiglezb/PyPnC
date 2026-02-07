import copy

import os, sys
cwd = os.getcwd()
sys.path.append(cwd)

import pinocchio as pin
import numpy as np
import pickle

from pinocchio.visualize import MeshcatVisualizer
from pydrake.geometry.optimization import HPolyhedron

import plot.meshcat_utils as vis_tools
import matplotlib.pyplot as plt

from util.environment_creator import TiltedStairs
from util.pydrake_meshcat_interface import hpoly_to_fcl_collision, create_convex_geom_from_copy


# --- Configuration ---
# ROBOT_MODEL = "g1_29dof_simple_collisions"
ROBOT_MODEL = "g1_29dof_lock_waist_chull"
ROBOT_URDF_PATH = cwd + "/robot_model/g1_description/"
ROBOT_URDF = ROBOT_URDF_PATH + ROBOT_MODEL + ".urdf"
ROBOT_SRDF = ROBOT_URDF_PATH + ROBOT_MODEL + ".srdf"
ROBOT_PACKAGE_DIRS = [cwd + "/robot_model/g1_description"]
ENV_URDF = cwd + "/robot_model/ground/navy_door_fixed.urdf"
ENV_NAME = 'door'   # {door, stairs} selects either door URDF or Stairs Collision objects

B_VISUALIZE_DOOR = False
B_ANIMATE = False
B_PRINT_COLLISION = True

def load_simulated_models(robot_urdf_path, env_urdf_path):
    """
    Loads a robot model, environment geometry, and a floating-base trajectory.
    """

    print(f"Loading models from: {robot_urdf_path} and {env_urdf_path}")

    # Load Robot Model with reduced collisions from SRDF
    robot_model, robot_col_model, robot_vis_model = pin.buildModelsFromUrdf(robot_urdf_path,
                                                                      ROBOT_PACKAGE_DIRS[0],
                                                                      pin.JointModelFreeFlyer())

    robot_geom_model = pin.buildGeomFromUrdf(robot_model, robot_urdf_path, pin.GeometryType.COLLISION, package_dirs=ROBOT_PACKAGE_DIRS)
    load_hull_collisions(robot_model, robot_geom_model, ROBOT_SRDF)

    # Load Environment Model
    env_model_fixed = pin.buildModelFromUrdf(env_urdf_path)  # Load model to get its frames
    env_geom_model = pin.buildGeomFromUrdf(env_model_fixed, env_urdf_path, pin.GeometryType.COLLISION)

    if B_VISUALIZE_DOOR:
        # visualize for debugging
        door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(env_urdf_path, cwd + "/robot_model/ground")
        door_vis = MeshcatVisualizer(door_model, door_collision_model, door_visual_model)
        door_vis.initViewer()
        door_vis.viewer.wait()
        door_vis.loadViewerModel(rootNodeName="door")
        door_vis.display_collisions = True
        door_vis.display()

    return robot_model, robot_col_model, robot_vis_model, robot_geom_model, env_geom_model


def load_trajectory(trajectory_pkl):
    """
    Loads a joint trajectory from a pickle file.
    """

    with open(trajectory_pkl, 'rb') as file:
        try:
            d = pickle.load(file)
            joint_pos = d['joint_pos']
            time = d['time']
        except EOFError:
            raise NotImplementedError

    print(f"Loaded trajectory with {len(joint_pos)} time steps.")

    return joint_pos, time

def merge_and_define_collision_pairs(robot_geom_model, env_geom_model):
    """
    Merges environment geometries into the robot's geometry model and defines
    the necessary robot-environment collision pairs.
    """

    print(f"Robot Geometry Model has {len(robot_geom_model.geometryObjects)} geometries.")
    print(f"Environment Geometry Model has {len(env_geom_model.geometryObjects)} geometries.")
    combined_geom_model = copy.copy(robot_geom_model)

    # --- merge environment collisions into robot model ---
    if 'door' in ENV_NAME:
        for i in range(env_geom_model.ngeoms):
            combined_geom_model.addGeometryObject(env_geom_model.geometryObjects[i])
        combined_geom_model.addAllCollisionPairs()
        pin.removeCollisionPairs(robot_model, combined_geom_model, ROBOT_SRDF)
    elif 'stairs' in ENV_NAME:
        stairs = TiltedStairs()
        for i_col, col_obj in enumerate(stairs.obstacles_vis):
            if i_col <= 2:  # skip floor and side walls collisions
                continue
            if isinstance(col_obj, HPolyhedron):
                obstacle_geom = hpoly_to_fcl_collision(col_obj)
            else:
                raise NotImplementedError("Only HPolyhedron obstacles are supported for stairs environment")
            # get from door: env_geom_model
            new_geom = create_convex_geom_from_copy(env_geom_model.geometryObjects[0],
                                                    obstacle_geom,
                                                    'stairs_obstacle_' + str(i_col))
            combined_geom_model.addGeometryObject(new_geom)
        combined_geom_model.addAllCollisionPairs()
    print(f"\n--- Geometries Merged. Total: {len(combined_geom_model.geometryObjects)} geometries. ---")

    return combined_geom_model


def check_trajectory_collisions(robot_model, robot_geom_model, joint_pos, time):
    """
    Replays the joint trajectory, checks for collisions at each step, and
    calculates the total penetration depth.
    """

    robot_data = robot_model.createData()
    robot_geom_data = pin.GeometryData(robot_geom_model)

    if B_ANIMATE:
        door_model, door_col_model, door_vis_model = pin.buildModelsFromUrdf(ENV_URDF,
                                                                          cwd + "/robot_model/ground")
        rob_data, col_data, vis_data = pin.createDatas(robot_model, robot_col_model, robot_vis_model)
        display = vis_tools.MeshcatPinocchioAnimation(robot_model, robot_col_model, robot_vis_model,
                          robot_data, vis_data, col_data, save_freq=10)
        if ENV_NAME == 'door':
            display.add_robot("door", door_model, door_col_model, door_vis_model)
        elif ENV_NAME == 'stairs':
            stairs = TiltedStairs()
            display.add_shapes_from(stairs.obstacles_vis)
        display.start_animation()

    penetration_depths, sum_penetration_depths = [], []

    idx_offset = 0
    for i, q in enumerate(joint_pos):
        # Skip data of impulse dynamics (if any)
        if i > 0 and (np.linalg.norm(np.array(joint_pos[i]) - np.array(joint_pos[i - 1])) < 0.001):
            idx_offset += 1
            continue

        # Update robot kinematics
        q = np.array(q)
        if B_ANIMATE:
            display.animate_frame_with_collisions(q)
            display.animation_step()

        pin.forwardKinematics(robot_model, robot_data, q)

        # Update geometry placements (MUST be called after forwardKinematics)
        pin.updateGeometryPlacements(robot_model, robot_data, robot_geom_model, robot_geom_data, q)

        # Compute all collisions (stopAtFirstCollision = False)
        pin.computeCollisions(robot_model, robot_data, robot_geom_model, robot_geom_data, q, False)

        # metrics to save
        min_distance_overall, sum_penetrations = 0.0, 0.0

        # Check for collisions among all collision pairs (including environment)
        for k, cr in enumerate(robot_geom_data.collisionResults):
            is_in_collision = cr.isCollision()

            if is_in_collision:
                # If any collision is detected, compute minimum distance (which is negative
                # and represents penetration depth)
                pin.computeDistances(robot_model, robot_data, robot_geom_model, robot_geom_data, q)

                res = robot_geom_data.distanceResults[k]
                sum_penetrations += abs(res.min_distance) if res.min_distance < 0.0 else 0.0

                # If this pair has the deepest penetration so far, store colliding body names
                if res.min_distance < min_distance_overall:
                    first_id = robot_geom_model.collisionPairs[k].first
                    second_id = robot_geom_model.collisionPairs[k].second
                    collision_pair_from = robot_geom_model.geometryObjects[first_id].name
                    collision_pair_to = robot_geom_model.geometryObjects[second_id].name

                    # Update the deepest penetration among all pairs at this time step
                    min_distance_overall = 1000*min(min_distance_overall, res.min_distance)

        # Store the absolute value of the minimum (deepest) penetration for each body
        max_penetration_at_step = abs(min_distance_overall)

        if B_PRINT_COLLISION and max_penetration_at_step > 0:
            print(
                f"Time {time[i-idx_offset]}: COLLISION ({collision_pair_from}, {collision_pair_to}). Max Penetration: {max_penetration_at_step:.4f} m"
            )

        penetration_depths.append(max_penetration_at_step)
        sum_penetration_depths.append(sum_penetrations)

    # --- Total Penetration Calculation ---
    total_penetration = np.sum(penetration_depths)

    print("\n--- Summary ---")
    print(f"Total time steps checked: {len(joint_pos)}")
    print(f"Number of time steps with collision: {np.sum(np.array(penetration_depths) > 0)}")
    print(f"Total Penetration (Sum of Max Penetration per Colliding Step): {total_penetration:.4f} m")

    if B_ANIMATE:
        display.finish_animation()

    return total_penetration, penetration_depths, sum_penetration_depths


def check_trajectory_env_robot_collisions(robot_model, robot_geom_model, joint_pos, time):
    """
    Replays the joint trajectory, checks for collisions at each step, and
    calculates the total penetration depth.
    """

    robot_data = robot_model.createData()
    robot_geom_data = pin.GeometryData(robot_geom_model)

    penetration_depths = []

    print("\n--- Replaying Trajectory and Checking Collisions ---")
    idx_offset = 0
    for i, q in enumerate(joint_pos):
        # Skip data of impulse dynamics (if any)
        if i > 0 and (np.linalg.norm(np.array(joint_pos[i]) - np.array(joint_pos[i - 1])) < 0.001):
            idx_offset += 1
            continue

        # Check up until the end of the time vector
        if i >= len(time):
            break

        # Update robot kinematics
        q = np.array(q)
        pin.forwardKinematics(robot_model, robot_data, q)

        # Update geometry placements (MUST be called after forwardKinematics)
        pin.updateGeometryPlacements(robot_model, robot_data, robot_geom_model, robot_geom_data, q)

        # Compute all collisions (stopAtFirstCollision = False)
        pin.computeCollisions(robot_model, robot_data, robot_geom_model, robot_geom_data, q, False)

        min_distance_overall = 0.0

        # Check for collisions among all collision pairs (including environment)
        for k, cr in enumerate(robot_geom_data.collisionResults):
            is_in_collision = cr.isCollision()

            if is_in_collision:
                # If any collision is detected, compute minimum distance (which is negative
                # and represents penetration depth)
                pin.computeDistances(robot_model, robot_data, robot_geom_model, robot_geom_data, q)

                res = robot_geom_data.distanceResults[k]

                # If this pair has the deepest penetration so far, store colliding body names
                first_id = robot_geom_model.collisionPairs[k].first
                second_id = robot_geom_model.collisionPairs[k].second
                collision_from = robot_geom_model.geometryObjects[first_id].name
                collision_to = robot_geom_model.geometryObjects[second_id].name
                # update penetration distance only if it involves the environment
                if res.min_distance < min_distance_overall and ((ENV_NAME in collision_from) or (ENV_NAME in collision_to)):
                    collision_pair_from = robot_geom_model.geometryObjects[first_id].name
                    collision_pair_to = robot_geom_model.geometryObjects[second_id].name
                    # Update the deepest penetration among all pairs at this time step
                    min_distance_overall = min(min_distance_overall, res.min_distance)

        # Store the absolute value of the minimum (deepest) penetration for each body
        max_penetration_at_step = 1000*abs(min_distance_overall)

        if B_PRINT_COLLISION and max_penetration_at_step > 0:
            print(
                f"Time {time[i-idx_offset]}: COLLISION ({collision_pair_from}, {collision_pair_to}). Max Penetration: {max_penetration_at_step:.4f} m"
            )

        penetration_depths.append(max_penetration_at_step)

    # --- Total Penetration Calculation ---
    total_penetration = np.sum(penetration_depths)

    print("\n--- Summary ---")
    print(f"Total time steps checked: {len(joint_pos)}")
    print(f"Number of time steps with collision: {np.sum(np.array(penetration_depths) > 0)}")
    print(f"Total Penetration (Sum of Max Penetration per Colliding Step): {total_penetration:.4f} m")

    return total_penetration, penetration_depths


def plot_self_collision_distances():
    plt.figure()
    plt.plot(kin_sca_time, scol_mfpp_penetration_depths, 'r--', label='no-SCA')
    plt.plot(kin_sca_time, scol_kin_sca_penetration_depths, 'b', label='kin-SCA')
    # plt.plot(time, scol_sum_penetrations, 'r', alpha=0.4, label='kin-SCA (sum)')
    plt.plot(sca_time, scol_sca_penetration_depths, 'ko--', label='full-SCA')
    # plt.plot(sca_time, scol_sca_sum_penetrations, 'c', alpha=0.4, label='full-SCA (sum)')
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Penetration Depth (mm)', fontsize=14)
    plt.title('Max Self-Collision Penetration', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()


def plot_results():
    plt.figure()
    plt.plot(sca_time, mfpp_penetration_depths, 'r--', label='no-SCA')
    plt.plot(sca_time, kin_sca_penetration_depths, 'b', label='kin-SCA')
    plt.plot(sca_time, sca_penetration_depths, 'ko--', label='full-SCA')
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Penetration Depth (mm)', fontsize=14)
    plt.title('Max Env Penetration', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()


def load_hull_collisions(robot_model, robot_geom_model, ROBOT_SRDF):
    # Iterate through all geometry objects and replace meshes with their convex hulls
    for go in robot_geom_model.geometryObjects:
        # assume STL extension for collision file
        if "stl" in go.meshPath.lower():
            go.geometry.buildConvexRepresentation(True)  # Compute the convex hull
            go.geometry = go.geometry.convex  # Replace the mesh with its convex hull

    robot_geom_model.addAllCollisionPairs()
    pin.removeCollisionPairs(robot_model, robot_geom_model, ROBOT_SRDF)


if __name__ == '__main__':
    MFPP_TRAJECTORY_PKL = cwd + "/experiment_data/pelvis_box/g1_guided__no_imp__no_sca_refine_step_on_balanced_door_knees_diag.pkl"
    KIN_SCA_TRAJECTORY_PKL = cwd + "/experiment_data/pelvis_box/g1_guided__no_imp_kin_sca_no_sca_refine__step_on_balanced_door_knees_diag.pkl"
    SCA_TRAJECTORY_PKL = cwd + "/experiment_data/pelvis_box/g1_guided__no_imp_kin_sca_sca_refine__step_on_balanced_door_knees_diag.pkl"

    # Load models
    robot_model, robot_col_model, robot_vis_model, robot_geom_model, env_geom_model = load_simulated_models(ROBOT_URDF, ENV_URDF)

    # Load trajectories
    mfpp_joint_pos, mfpp_time = load_trajectory(MFPP_TRAJECTORY_PKL)
    kin_sca_joint_pos, kin_sca_time = load_trajectory(KIN_SCA_TRAJECTORY_PKL)
    sca_joint_pos, sca_time = load_trajectory(SCA_TRAJECTORY_PKL)

    # ---------
    # Check self-collisions
    # ---------
    print("\n--- Replaying MFPP Trajectory and Checking Collisions ---")
    scol_mfpp_total_penetration, scol_mfpp_penetration_depths, scol_mfpp_sum_penetrations = check_trajectory_collisions(
        robot_model, robot_geom_model, mfpp_joint_pos, mfpp_time
    )
    print("\n--- Replaying kin-SCA Trajectory and Checking Collisions ---")
    scol_kin_sca_total_penetration, scol_kin_sca_penetration_depths, scol_kin_sca_sum_penetrations = check_trajectory_collisions(
        robot_model, robot_geom_model, kin_sca_joint_pos, kin_sca_time
    )
    # Replay trajectory and check self-collisions in both nominal and SCA cases
    print("\n--- Replaying full-SCA Trajectory and Checking Collisions ---")
    scol_sca_total_penetration, scol_sca_penetration_depths, scol_sca_sum_penetrations = check_trajectory_collisions(
        robot_model, robot_geom_model, sca_joint_pos, sca_time
    )

    # Plot penetration depths over time
    plot_self_collision_distances()

    # ---------
    # Check env-robot collisions
    # ---------
    # Combine collision geometries to check env-robot collisions
    combined_geom_model = merge_and_define_collision_pairs(robot_geom_model, env_geom_model)

    # Replay trajectory and check collisions
    mfpp_total_penetration, mfpp_penetration_depths = check_trajectory_env_robot_collisions(
        robot_model, combined_geom_model, mfpp_joint_pos, mfpp_time
    )
    kin_sca_total_penetration, kin_sca_penetration_depths = check_trajectory_env_robot_collisions(
        robot_model, combined_geom_model, kin_sca_joint_pos, kin_sca_time
    )
    sca_total_penetration, sca_penetration_depths = check_trajectory_env_robot_collisions(
        robot_model, combined_geom_model, sca_joint_pos, sca_time
    )

    # Plot penetration depths over time
    plot_results()

    # Print maximum values in each case
    print('     |     MFPP     |     kin-SCA    |     full-SCA    ')
    print(f'self: {max(scol_mfpp_penetration_depths):.1f}  {max(scol_kin_sca_penetration_depths):.1f}  {max(scol_sca_penetration_depths):.1f}')
    print(f'env: {max(mfpp_penetration_depths):.1f}  {max(kin_sca_penetration_depths):.1f}  {max(sca_penetration_depths):.1f}')
