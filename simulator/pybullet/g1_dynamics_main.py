import os
import sys

from util.interpolation import interpolate_linearly
from util.util import trajectory_scaler

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
    p.connect(p.DIRECT)     #p.DIRECT
    p.resetDebugVisualizerCamera(
        cameraDistance=1.6,
        cameraYaw=180,
        cameraPitch=-6,
        cameraTargetPosition=[0, 0.5, 1.2])
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
    max_iter = 50

    initial_jpos_dict = copy.deepcopy(nominal_sensor_data['joint_pos'])
    b_replay_plan = False
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
        # keys = p.getKeyboardEvents()
        # if pybullet_util.is_key_triggered(keys, 'c'):
        #     for k, v in gripper_command.items():
        #         gripper_command[k] += 1.94 / 3.
        # elif pybullet_util.is_key_triggered(keys, 'o'):
        #     for k, v in gripper_command.items():
        #         gripper_command[k] -= 1.94 / 3.
        # elif pybullet_util.is_key_triggered(keys, 'p'):
        if t > 0.5:
            b_replay_plan = True
            ref_start_time = t
            replan_call_idx = 0
            to_ref_idx = 0          # index of TO trajectory reference

        # choose command from either planner or MPC balance controller
        if b_replay_plan:
            # Get next reference trajectories (ideally, we'd get these from the
            # Bezier curve functions. For now, we just get them from the pkl file).
            # Get all the data we have up to the next horizon point
            # TODO can be made smaller than dim N_horizon
            qb_r, qj_r = np.zeros((10, 7)), np.zeros((10, nq-7))
            qdb_r, qdj_r = np.zeros((10, 6)), np.zeros((10, nv-6))
            tau_r = np.zeros((10, na))

            # check if we can move on to next index in TO trajectories
            if (ref_start_time + replan_call_idx * dt) > (ref_start_time + ref_time[to_ref_idx + 1]):
                to_ref_idx += 1

            # grab the reference points covering the next horizon
            t_ref_0, t_ref = ref_time[to_ref_idx], ref_time[replan_call_idx]
            r_idx = 0
            while t_ref < (t_ref_0 + T):
                qb_r[r_idx, :] = ref_q_base[r_idx]
                qj_r[r_idx, :] = ref_q_joints[r_idx]
                qdb_r[r_idx, :] = ref_qd_base[r_idx]
                qdj_r[r_idx, :] = ref_qd_joints[r_idx]
                tau_r[r_idx, :] = ref_tau_joints[r_idx]
                t_ref = ref_time[r_idx+1]
                r_idx += 1
            # check we get the last point covering the horizon up to T
            qb_r[r_idx, :] = interpolate_linearly(ref_q_base[r_idx-1], ref_q_base[r_idx],
                                                  ref_time[r_idx-1], ref_time[r_idx], t_ref_0 + T)
            qj_r[r_idx, :] = interpolate_linearly(ref_q_joints[r_idx-1], ref_q_joints[r_idx],
                                                  ref_time[r_idx-1], ref_time[r_idx], t_ref_0 + T)
            qdb_r[r_idx, :] = interpolate_linearly(ref_qd_base[r_idx-1], ref_qd_base[r_idx],
                                                   ref_time[r_idx-1], ref_time[r_idx], t_ref_0 + T)
            qdj_r[r_idx, :] = interpolate_linearly(ref_qd_joints[r_idx-1], ref_qd_joints[r_idx],
                                                   ref_time[r_idx-1], ref_time[r_idx], t_ref_0 + T)
            tau_r[r_idx, :] = interpolate_linearly(ref_tau_joints[r_idx-1], ref_tau_joints[r_idx],
                                                   ref_time[r_idx-1], ref_time[r_idx], t_ref_0 + T)

            # Interpolate references to deal with different time steps
            mpc_ref_time = np.concatenate((ref_time[:r_idx], [t_ref_0 + T]))
            qb_r_scaled = trajectory_scaler(qb_r, N_horizon, mpc_ref_time, t_ref_0, dt)
            qj_r_scaled = trajectory_scaler(qj_r, N_horizon, mpc_ref_time, t_ref_0, dt)
            qdb_r_scaled = trajectory_scaler(qdb_r, N_horizon, mpc_ref_time, t_ref_0, dt)
            qdj_r_scaled = trajectory_scaler(qdj_r, N_horizon, mpc_ref_time, t_ref_0, dt)
            tau_r_scaled = trajectory_scaler(tau_r, N_horizon, mpc_ref_time, t_ref_0, dt)

            # Set corresponding EE references (forgot to save them)
            all_frame_targets_dict = [None] * N_horizon
            model_seqs = []
            for r_idx in range(N_horizon):
                frame_targets_dict = {}
                q0 = np.concatenate([qb_r_scaled[r_idx], qj_r_scaled[r_idx]])
                v0 = np.concatenate([qdb_r_scaled[r_idx], qdj_r_scaled[r_idx]])
                x0 = np.concatenate((q0, v0))
                pin.forwardKinematics(rob_model, rob_data, q0)
                pin.updateFramePlacements(rob_model, rob_data)
                frame_targets_dict['torso'] = rob_data.oMf[plan_to_model_ids['torso']].translation
                frame_targets_dict['LF'] = rob_data.oMf[plan_to_model_ids['LF']].translation
                frame_targets_dict['RF'] = rob_data.oMf[plan_to_model_ids['RF']].translation
                frame_targets_dict['L_knee'] = rob_data.oMf[plan_to_model_ids['L_knee']].translation
                frame_targets_dict['R_knee'] = rob_data.oMf[plan_to_model_ids['R_knee']].translation
                frame_targets_dict['LH'] = rob_data.oMf[plan_to_model_ids['LH']].translation
                frame_targets_dict['RH'] = rob_data.oMf[plan_to_model_ids['RH']].translation
                all_frame_targets_dict[r_idx] = frame_targets_dict

                # Set commands to current
                dmodel = createMultiFrameActionModel(state,
                                                     actuation,
                                                     x_current,
                                                     plan_to_model_ids,
                                                     ['LF', 'RF'],
                                                     ee_rpy,
                                                     all_frame_targets_dict[r_idx],
                                                     None,
                                                     zero_config=x0)
                model_seqs += createSequence([dmodel], T / N_horizon, 0)
            problem = crocoddyl.ShootingProblem(x_current, sum(model_seqs, [])[:-1], model_seqs[-1][-1])
            fddp = crocoddyl.SolverFDDP(problem)
            fddp.setCallbacks([crocoddyl.CallbackLogger()])
            fddp.th_stop = 1e-3

            # Set initial guess and solve
            xs = [x_current] * (fddp.problem.T + 1)
            us = fddp.problem.quasiStatic([x_current] * fddp.problem.T)
            # us = tau_r_scaled[:-1]
            if not fddp.solve(xs, us, max_iter):
                print(f"Failed to solve MPC problem in {fddp.iter} iterations.")

            # balance with torque control
            pos_des = list(fddp.xs[0][7:nq])
            vel_des = list(fddp.xs[0][-(nv-6):])
            trq_des = list(fddp.us[0])
            trq_applied = (trq_des +
                           np.array(list(kp.values())) * (pos_des - x_current[7:nq]) +
                           np.array(list(kd.values())) * (vel_des - x_current[-(nv-6):]))
            p.setJointMotorControlArray(robot,
                                        joint_id.values(),
                                        controlMode=p.TORQUE_CONTROL,
                                        forces=list(trq_applied))
            # pybullet_util.set_motor_impedance(robot, joint_id, command, kp, kd)
            replan_call_idx += 1

            # if last step, reset flags and balance
            if t_ref >= (20 - 2*T):
                b_replay_plan = False
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
