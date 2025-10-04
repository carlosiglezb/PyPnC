import pinocchio as pin
import numpy as np
import os, sys
import pickle

from pinocchio.visualize import MeshcatVisualizer
import plot.meshcat_utils as vis_tools

cwd = os.getcwd()
sys.path.append(cwd)

# --- Configuration ---
ROBOT_URDF = cwd + "/robot_model/g1_description/g1_29dof_simple_collisions.urdf"
ENV_URDF = cwd + "/robot_model/ground/navy_door_fixed.urdf"
TRAJECTORY_PKL = cwd + "/experiment_data/g1_sca_on_balanced_door_boxfddp.pkl"

B_VISUALIZE = False
B_ANIMATE = True

def load_simulated_models_and_trajectory(robot_urdf_path, env_urdf_path, traj_pkl_path):
    """
    Loads a robot model, environment geometry, and a floating-base trajectory.
    """

    print(f"Loading models from: {robot_urdf_path} and {env_urdf_path}")
    print(f"Loading trajectory from: {traj_pkl_path}")

    # Load Robot Model
    robot_model = pin.buildModelFromUrdf(robot_urdf_path, pin.JointModelFreeFlyer())
    robot_geom_model = pin.buildGeomFromUrdf(robot_model, robot_urdf_path, pin.GeometryType.COLLISION)

    # Load Environment Model
    env_model_fixed = pin.buildModelFromUrdf(env_urdf_path)  # Load model to get its frames
    env_geom_model = pin.buildGeomFromUrdf(env_model_fixed, env_urdf_path, pin.GeometryType.COLLISION)

    if B_VISUALIZE:
        # visualize for debugging
        door_model, door_collision_model, door_visual_model = pin.buildModelsFromUrdf(env_urdf_path, cwd + "/robot_model/ground")
        door_vis = MeshcatVisualizer(door_model, door_collision_model, door_visual_model)
        door_vis.initViewer()
        door_vis.viewer.wait()
        door_vis.loadViewerModel(rootNodeName="door")
        door_vis.display_collisions = True
        door_vis.display()

    # --- Load Trajectory ---
    with open(traj_pkl_path, 'rb') as f:
        d = pickle.load(f)
        joint_pos = d['joint_pos']
        time = d['time']

    print("Successfully loaded models and trajectory data.")
    return robot_model, robot_geom_model, env_geom_model, joint_pos, time


def merge_and_define_collision_pairs(robot_geom_model, env_geom_model):
    """
    Merges environment geometries into the robot's geometry model and defines
    the necessary robot-environment collision pairs.
    """

    print(f"Robot Geometry Model has {len(robot_geom_model.geometryObjects)} geometries.")
    print(f"Environment Geometry Model has {len(env_geom_model.geometryObjects)} geometries.")

    # --- merge environment collisions into robot model ---
    for i in range(env_geom_model.ngeoms):
        robot_geom_model.addGeometryObject(env_geom_model.geometryObjects[i])
    robot_geom_model.addAllCollisionPairs()
    print(f"\n--- Geometries Merged. Total: {len(robot_geom_model.geometryObjects)} geometries. ---")

    return robot_geom_model


def check_trajectory_collisions(robot_model, robot_geom_model, joint_pos, time):
    """
    Replays the joint trajectory, checks for collisions at each step, and
    calculates the total penetration depth.
    """

    robot_data = robot_model.createData()
    robot_geom_data = pin.GeometryData(robot_geom_model)

    if B_ANIMATE:
        rob_model, rob_col_model, rob_vis_model = pin.buildModelsFromUrdf(ROBOT_URDF,
                                                                          cwd + "/robot_model/g1_description",
                                                                          pin.JointModelFreeFlyer())
        door_model, door_col_model, door_vis_model = pin.buildModelsFromUrdf(ENV_URDF,
                                                                          cwd + "/robot_model/ground")
        rob_data, col_data, vis_data = pin.createDatas(rob_model, rob_col_model, rob_vis_model)
        display = vis_tools.MeshcatPinocchioAnimation(rob_model, rob_col_model, rob_vis_model,
                          robot_data, vis_data, col_data, save_freq=10)
        display.add_robot("door", door_model, door_col_model, door_vis_model)
        display.start_animation()

    penetration_depths = []

    print("\n--- Replaying Trajectory and Checking Collisions ---")
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
                if res.min_distance < min_distance_overall:
                    first_id = robot_geom_model.collisionPairs[k].first
                    second_id = robot_geom_model.collisionPairs[k].second
                    collision_pair_from = robot_geom_model.geometryObjects[first_id].name
                    collision_pair_to = robot_geom_model.geometryObjects[second_id].name

                # Find the deepest penetration among all pairs at this time step
                min_distance_overall = min(min_distance_overall, res.min_distance)

        # Store the absolute value of the minimum (deepest) penetration for each body
        max_penetration_at_step = abs(min_distance_overall)

        if max_penetration_at_step > 0:
            print(
                f"Time {time[i]}: COLLISION ({collision_pair_from}, {collision_pair_to}). Max Penetration: {max_penetration_at_step:.4f} m"
            )

        penetration_depths.append(max_penetration_at_step)

    # --- Total Penetration Calculation ---
    total_penetration = np.sum(penetration_depths)

    print("\n--- Summary ---")
    print(f"Total time steps checked: {len(joint_pos)}")
    print(f"Number of time steps with collision: {np.sum(np.array(penetration_depths) > 0)}")
    print(f"Total Penetration (Sum of Max Penetration per Colliding Step): {total_penetration:.4f} m")

    if B_ANIMATE:
        display.finish_animation()

    return total_penetration, penetration_depths


def plot_results():
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(time, penetration_depths, label='SCA')
    plt.xlabel('Time (s)')
    plt.ylabel('Penetration Depth (m)')
    plt.title('Penetration Depth Over Time')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # Load models and trajectory
    robot_model, robot_geom_model, env_geom_model, joint_pos, time = \
        load_simulated_models_and_trajectory(ROBOT_URDF, ENV_URDF, TRAJECTORY_PKL)

    # Combine collision geometries and define pairs
    combined_geom_model = merge_and_define_collision_pairs(robot_geom_model, env_geom_model)

    # Replay trajectory and check collisions
    total_penetration, penetration_depths = check_trajectory_collisions(
        robot_model, combined_geom_model, joint_pos, time
    )

    # Plot penetration depths over time
    plot_results()


