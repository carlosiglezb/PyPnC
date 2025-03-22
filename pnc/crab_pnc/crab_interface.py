import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
import copy

import pybullet as p

from config.crab_config import PnCConfig
from pnc.interface import Interface
from pnc.crab_pnc.crab_interrupt_logic import CrabInterruptLogic
from pnc.crab_pnc.crab_state_provider import CrabStateProvider
from pnc.crab_pnc.crab_state_estimator import CrabStateEstimator
from pnc.crab_pnc.crab_control_architecture import CrabControlArchitecture
from pnc.data_saver import DataSaver


class CrabInterface(Interface):
    def __init__(self):
        super(CrabInterface, self).__init__()

        if PnCConfig.DYN_LIB == "dart":
            from pnc.robot_system.dart_robot_system import DartRobotSystem
            self._robot = DartRobotSystem(
                "/home/junhyeok/Repository/PnC/RobotModel/draco/draco_rel_path.urdf",
                False, False)
        elif PnCConfig.DYN_LIB == "pinocchio":
            from pnc.robot_system.pinocchio_robot_system import PinocchioRobotSystem
            self._robot = PinocchioRobotSystem(
                cwd + "/robot_model/crab/crab.urdf",
                cwd + "/robot_model/crab", False, PnCConfig.PRINT_ROBOT_INFO)
        else:
            raise ValueError("wrong dynamics library")

        self._sp = CrabStateProvider(self._robot)
        self._se = CrabStateEstimator(self._robot)
        self._control_architecture = CrabControlArchitecture(self._robot)
        self._interrupt_logic = CrabInterruptLogic(
            self._control_architecture)
        if PnCConfig.SAVE_DATA:
            self._data_saver = DataSaver()
            self._data_saver.add('joint_pos_limit',
                                 self._robot.joint_pos_limit)
            self._data_saver.add('joint_vel_limit',
                                 self._robot.joint_vel_limit)
            self._data_saver.add('joint_trq_limit',
                                 self._robot.joint_trq_limit)

    def get_command(self, sensor_data):
        if PnCConfig.SAVE_DATA:
            self._data_saver.add('time', self._running_time)
            self._data_saver.add('phase', self._control_architecture.state)

        # Update State Estimator
        if self._count == 0:
            print("=" * 80)
            print("Initialize")
            print("=" * 80)
            self._se.initialize(sensor_data)
        self._se.update(sensor_data)

        # Process Interrupt Logic
        self._interrupt_logic.process_interrupts()

        # Compute Cmd
        command = self._control_architecture.get_command()

        if PnCConfig.SAVE_DATA and (self._count % PnCConfig.SAVE_FREQ == 0):
            self._data_saver.add('joint_pos', self._robot.joint_positions)
            self._data_saver.add('joint_vel', self._robot.joint_velocities)
            self._data_saver.advance()

        # Increase time variables
        self._count += 1
        self._running_time += PnCConfig.CONTROLLER_DT
        self._sp.curr_time = self._running_time
        self._sp.prev_state = self._control_architecture.prev_state
        self._sp.state = self._control_architecture.state

        return copy.deepcopy(command)

    @property
    def interrupt_logic(self):
        return self._interrupt_logic
