import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
import time, math
from collections import OrderedDict
import copy
import signal
import pickle

# import cv2
import pybullet as p
import numpy as np

np.set_printoptions(precision=2)

from config.g1_config import SimConfig, LowLevelConfig
from util import pybullet_util

gripper_joints = [
    "left_zero_joint", "left_one_joint", "left_two_joint", "left_three_joint",
    "left_four_joint", "left_five_joint", "left_six_joint",
    "right_zero_joint", "right_one_joint", "right_two_joint", "right_three_joint",
    "right_four_joint", "right_five_joint", "right_six_joint"
]


def set_initial_config(robot, joint_id):
    # Upperbody

    # Lowerbody
    p.resetJointState(robot, joint_id["left_hip_pitch_joint"], -np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["left_knee_joint"], np.pi / 3, 0.)
    p.resetJointState(robot, joint_id["left_ankle_pitch_joint"], -np.pi / 6, 0.)

    p.resetJointState(robot, joint_id["right_hip_pitch_joint"], -np.pi / 6, 0.)
    p.resetJointState(robot, joint_id["right_knee_joint"], np.pi / 3, 0.)
    p.resetJointState(robot, joint_id["right_ankle_pitch_joint"], -np.pi / 6, 0.)


def signal_handler(signal, frame):
    # if SimConfig.VIDEO_RECORD:
    #     pybullet_util.make_video(video_dir, False)
    p.disconnect()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":

    # Environment Setup
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.6,
        cameraYaw=180,
        cameraPitch=-6,
        cameraTargetPosition=[0, 0.5, 1.0])
    p.setGravity(0, 0, -9.8)
    p.setPhysicsEngineParameter(
        fixedTimeStep=SimConfig.CONTROLLER_DT, numSubSteps=SimConfig.N_SUBSTEP)
    # if SimConfig.VIDEO_RECORD: TODO implement with Meshcat?

    # Create Robot, Ground
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(cwd + "/robot_model/g1_description/g1.urdf",
                       SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
                       SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT)

    p.loadURDF(cwd + "/robot_model/ground/plane.urdf", [0, 0, 0])
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    nq, nv, na, joint_id, link_id, pos_basejoint_to_basecom, rot_basejoint_to_basecom = pybullet_util.get_robot_config(
        robot, SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
        SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT, SimConfig.PRINT_ROBOT_INFO)

    p.loadURDF(cwd + "/robot_model/ground/navy_door.urdf",
               SimConfig.NAVY_DOOR_POSITION,
               SimConfig.NAVY_DOOR_QUAT)

    # Initial Config
    set_initial_config(robot, joint_id)

    # Link Damping
    pybullet_util.set_link_damping(robot, link_id.values(), 0., 0.)

    # Joint Friction
    pybullet_util.set_joint_friction(robot, joint_id, 0.)

    # Load pkl file and save reference trajectories in lists
    (ref_time, ref_q_base, ref_q_joints, ref_qd_base, ref_qd_joints,
     ref_tau_joints) = [], [], [], [], [], []
    with open('experiment_data/g1_knee_knocker.pkl', 'rb') as f:
        while True:
            try:
                d = pickle.load(f)
                ref_time.append(d['time'])
                ref_q_base.append(d['q_base'])
                ref_q_joints.append(d['q_joints'])
                ref_qd_base.append(d['qd_base'])
                ref_qd_joints.append(d['qd_joints'])
                ref_tau_joints.append(d['tau_joints'])
            except EOFError:
                break

    # Run Sim
    t = 0
    dt = SimConfig.CONTROLLER_DT
    count = 0

    nominal_sensor_data = pybullet_util.get_sensor_data(
        robot, joint_id, link_id, pos_basejoint_to_basecom,
        rot_basejoint_to_basecom)

    kp, kd = pybullet_util.get_gains_dicts(joint_id,
                                           LowLevelConfig.KP_JOINT,
                                           LowLevelConfig.KD_JOINT)

    gripper_command = dict()
    for gripper_joint in gripper_joints:
        gripper_command[gripper_joint] = nominal_sensor_data['joint_pos'][
            gripper_joint]

    initial_jpos_dict = copy.deepcopy(nominal_sensor_data['joint_pos'])
    b_start_plan = False
    command = {}
    while True:
        # Get SensorData
        sensor_data = pybullet_util.get_sensor_data(robot, joint_id, link_id,
                                                    pos_basejoint_to_basecom,
                                                    rot_basejoint_to_basecom)

        rf_height = pybullet_util.get_link_iso(robot,
                                               link_id['right_ankle_pitch_link'])[2, 3]
        lf_height = pybullet_util.get_link_iso(robot,
                                               link_id['left_ankle_pitch_link'])[2, 3]
        sensor_data['b_rf_contact'] = True if rf_height <= 0.03 else False
        sensor_data['b_lf_contact'] = True if lf_height <= 0.03 else False

        # Get Keyboard Event
        keys = p.getKeyboardEvents()
        if pybullet_util.is_key_triggered(keys, 'c'):
            for k, v in gripper_command.items():
                gripper_command[k] += 1.94 / 3.
        elif pybullet_util.is_key_triggered(keys, 'o'):
            for k, v in gripper_command.items():
                gripper_command[k] -= 1.94 / 3.
        elif pybullet_util.is_key_triggered(keys, 'p'):
            b_start_plan = True

        # choose command from either planner or MPC balance controller
        if b_start_plan:
            # Get next reference trajectories

            # Interpolate references


            # Perform MPC


            # Set commands
            command['joint_pos'] = [0.0] * len(joint_id)
            command['joint_vel'] = [0.0] * len(joint_id)
            command['joint_trq'] = [0.0] * len(joint_id)
            pybullet_util.set_motor_impedance(robot, joint_id, command, kp, kd)

            # if last step, reset flags and balance

        else:
            # balance
            pybullet_util.set_motor_pos(robot, joint_id, initial_jpos_dict)

        # # Save Image
        # if (SimConfig.VIDEO_RECORD) and (count % SimConfig.RECORD_FREQ == 0):
        #     frame = pybullet_util.get_camera_image(
        #         [1., 0.5, 1.], 1.0, 120, -15, 0, 60., 1920, 1080, 0.1, 100.)
        #     frame = frame[:, :, [2, 1, 0]]  # << RGB to BGR
        #     filename = video_dir + '/step%06d.jpg' % jpg_count
        #     cv2.imwrite(filename, frame)
        #     jpg_count += 1

        p.stepSimulation()

        time.sleep(dt)
        t += dt
        count += 1
