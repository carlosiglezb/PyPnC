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
# planner tools
import crocoddyl
import pinocchio as pin
from pnc.planner.multicontact.dyn_feasibility.humanoid_action_models import (createMultiFrameActionModel,
                                                                             createSequence,
                                                                             createDoubleSupportActionModel)
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


def get_plan_to_model_frames():
    plan_to_model_frames = OrderedDict()
    plan_to_model_frames['torso'] = 'torso_link'
    plan_to_model_frames['LF'] = 'left_ankle_roll_link'
    plan_to_model_frames['RF'] = 'right_ankle_roll_link'
    plan_to_model_frames['L_knee'] = 'left_knee_link'
    plan_to_model_frames['R_knee'] = 'right_knee_link'
    plan_to_model_frames['LH'] = 'left_palm_link'
    plan_to_model_frames['RH'] = 'right_palm_link'
    return plan_to_model_frames


def get_plan_to_model_ids(rob_model):
    plan_to_model_frames = get_plan_to_model_frames()

    # Getting the frame ids
    plan_to_model_ids = {}
    plan_to_model_ids['RF'] = rob_model.getFrameId(plan_to_model_frames['RF'])
    plan_to_model_ids['LF'] = rob_model.getFrameId(plan_to_model_frames['LF'])
    plan_to_model_ids['R_knee'] = rob_model.getFrameId(plan_to_model_frames['R_knee'])
    plan_to_model_ids['L_knee'] = rob_model.getFrameId(plan_to_model_frames['L_knee'])
    plan_to_model_ids['LH'] = rob_model.getFrameId(plan_to_model_frames['LH'])
    plan_to_model_ids['RH'] = rob_model.getFrameId(plan_to_model_frames['RH'])
    plan_to_model_ids['torso'] = rob_model.getFrameId(plan_to_model_frames['torso'])

    return plan_to_model_ids


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
    package_dir = cwd + "/robot_model/g1_description"
    robot_urdf_file = package_dir + "/g1.urdf"
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    robot = p.loadURDF(robot_urdf_file,
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

    # Build pinocchio robot model
    rob_model, col_model, vis_model = pin.buildModelsFromUrdf(robot_urdf_file,
                                                              package_dir,
                                                              pin.JointModelFreeFlyer())
    rob_data, _, __ = pin.createDatas(rob_model, col_model, vis_model)

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

    # update pinocchio model
    q0_joints = np.array(list(nominal_sensor_data['joint_pos'].values()))
    q0_base = np.concatenate([SimConfig.INITIAL_POS_WORLD_TO_BASEJOINT,
                             SimConfig.INITIAL_QUAT_WORLD_TO_BASEJOINT])
    q0 = np.concatenate([q0_base, q0_joints])
    pin.forwardKinematics(rob_model, rob_data, q0)
    pin.updateFramePlacements(rob_model, rob_data)
    # Initialize Planner
    plan_to_model_ids = get_plan_to_model_ids(rob_model)
    state = crocoddyl.StateMultibody(rob_model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    ee_rpy = {'LH': [0., 0., 0.], 'RH': [0., 0., 0.]}
    # Solver settings
    T, N_horizon = 0.5, 50
    max_iter = 100

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

        # update current state
        x_current = np.concatenate((
                                    sensor_data['base_joint_pos'],
                                    sensor_data['base_joint_quat'],
                                    np.array(list(sensor_data['joint_pos'].values())),
                                    sensor_data['base_joint_lin_vel'],
                                    sensor_data['base_joint_ang_vel'],
                                    np.array(list(sensor_data['joint_vel'].values())),
        ))

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
            ref_start_time = t

        # choose command from either planner or MPC balance controller
        if b_start_plan:
            # Get next reference trajectories (ideally, we'd get these from the
            # Bezier curve functions. For now, we just get them from the pkl file)


            # Interpolate references to deal with different time steps


            # Perform MPC


            # Set commands
            command['joint_pos'] = [0.0] * len(joint_id)
            command['joint_vel'] = [0.0] * len(joint_id)
            command['joint_trq'] = [0.0] * len(joint_id)
            pybullet_util.set_motor_impedance(robot, joint_id, command, kp, kd)

            # if last step, reset flags and balance

        else:               # balance with MPC
            if t < 1:       # balance with fixed position control for the first second or so
                pybullet_util.set_motor_pos(robot, joint_id, initial_jpos_dict)
            else:           # balance with torque control
                # Set commands to current
                dmodel = createDoubleSupportActionModel(state,
                                                        actuation,
                                                        x_current,
                                                        plan_to_model_ids['LF'],
                                                        plan_to_model_ids['RF'],
                                                        None,
                                                        None,
                                                        None)
                model_seqs = createSequence([dmodel], T / N_horizon, N_horizon-1)
                problem = crocoddyl.ShootingProblem(x_current, sum(model_seqs, [])[:-1], model_seqs[-1][-1])
                fddp = crocoddyl.SolverFDDP(problem)
                fddp.setCallbacks([crocoddyl.CallbackLogger()])
                fddp.th_stop = 1e-3

                # Set initial guess and solve
                xs = [x_current] * (fddp.problem.T + 1)
                us = fddp.problem.quasiStatic([x_current] * fddp.problem.T)
                if not fddp.solve(xs, us, max_iter):
                    print("Failed to solve")

                # balance with torque control
                tau_des = list(fddp.us[0])
                p.setJointMotorControlArray(robot,
                                            list(joint_id.values()),
                                            controlMode=p.TORQUE_CONTROL,
                                            forces=tau_des)

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
        print(f't: {t}')
