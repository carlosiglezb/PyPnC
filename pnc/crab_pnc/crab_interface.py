import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import time, math
import copy
import numpy as np

import pybullet as p

from config.crab_config import PnCConfig
from pnc.interface import Interface
from pnc.crab_pnc.crab_interrupt_logic import CrabInterruptLogic
from pnc.crab_pnc.crab_state_provider import CrabStateProvider
from pnc.crab_pnc.crab_state_estimator import CrabStateEstimator
from pnc.crab_pnc.crab_control_architecture import CrabControlArchitecture
from pnc.data_saver import DataSaver

# ====================================================================== 
# class CrabInterface 
# ====================================================================== 

class CrabInterface(Interface):
    """
    Interface class for the Crab robot. Handles communication between the high-level
    controller and either simulation or hardware.
    
    This class:
    1. Updates the robot state based on sensor data
    2. Generates motor commands
    3. Handles conversion between different coordinate systems/units
    4. Implements safety checks and limits
    """
    # ====================================================================== 
    # __init__ 
    # ====================================================================== 

    def __init__(self):
        """
        Initialize the interface with default values and create state provider.
        The state provider maintains the current state of the robot.
        """
        super(CrabInterface, self).__init__()

        # ---------------------------------- 
        # Initialize Robot System
        # ----------------------------------    
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

        # ---------------------------------- 
        # Initialize State Provider
        # ---------------------------------- 
        self._sp = CrabStateProvider(self._robot)

        # ---------------------------------- 
        # Initialize State Estimator
        # ---------------------------------- 
        self._se = CrabStateEstimator(self._robot)

        # ---------------------------------- 
        # Initialize Control Architecture
        # ---------------------------------- 
        self._control_architecture = CrabControlArchitecture(self._robot)

        # ---------------------------------- 
        # Initialize Interrupt Logic
        # ---------------------------------- 
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

    # ====================================================================== 
    # get_command 
    # ====================================================================== 

    def get_command(self, sensor_data):
        """
        Generate commands for the robot's actuators based on the current state
        and control architecture.
        
        Returns:
            command (dict): Dictionary containing:
                - 'joint_pos': Desired joint positions
                - 'joint_vel': Desired joint velocities
                - 'joint_trq': Desired joint torques
        """
        if PnCConfig.SAVE_DATA:
            self._data_saver.add('time', self._running_time)
            self._data_saver.add('phase', self._control_architecture.state)

        # ---------------------------------- 
        # Update State Estimator
        # ---------------------------------- 
        if self._count == 0:
            print("=" * 80)
            print("Initialize")
            print("=" * 80)
            self._se.initialize(sensor_data)
        self._se.update(sensor_data)

        # ---------------------------------- 
        # Process Interrupt Logic
        # ---------------------------------- 
        self._interrupt_logic.process_interrupts()

        # ---------------------------------- 
        # Compute Command
        # ---------------------------------- 
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

    # ====================================================================== 
    # Property Getters
    # ====================================================================== 

    @property
    def interrupt_logic(self):
        return self._interrupt_logic

    # ====================================================================== 
    # update_robot_state 
    # ====================================================================== 

    def update_robot_state(self, sensor_data):
        """
        Update the robot's state based on new sensor data.
        
        This method:
        1. Updates joint states (position, velocity)
        2. Updates base position and orientation
        3. Updates the state provider with new data
        
        Args:
            sensor_data (dict): Dictionary containing:
                - 'base_joint_pos': Base joint positions
                - 'base_joint_vel': Base joint velocities
                - 'base_com_pos': Base COM position
                - 'base_com_quat': Base COM orientation (quaternion)
                - 'base_com_lin_vel': Base COM linear velocity
                - 'base_com_ang_vel': Base COM angular velocity
                - 'base_joint_pos': Actuated joint positions
                - 'base_joint_vel': Actuated joint velocities
        """
        # Update base (floating joint) states
        self._sp.q[:6] = sensor_data["base_joint_pos"]
        self._sp.qdot[:6] = sensor_data["base_joint_vel"]
        
        # Update actuated joint states
        self._sp.q[6:] = sensor_data["joint_pos"]
        self._sp.qdot[6:] = sensor_data["joint_vel"]
        
        # Update base pose and velocity
        self._sp.base_com_pos = sensor_data["base_com_pos"]
        self._sp.base_com_quat = sensor_data["base_com_quat"]
        self._sp.base_com_lin_vel = sensor_data["base_com_lin_vel"]
        self._sp.base_com_ang_vel = sensor_data["base_com_ang_vel"]
        
        # Update base joint position and velocity
        self._sp.base_joint_pos = sensor_data["base_joint_pos"]
        self._sp.base_joint_vel = sensor_data["base_joint_vel"]
        
        # Update time
        self._sp.time = sensor_data["time"]

    # ====================================================================== 
    # initialize 
    # ====================================================================== 

    def initialize(self, sensor_data):
        """
        Initialize the interface with initial sensor data.
        
        This method:
        1. Sets up initial robot state
        2. Initializes control architecture
        3. Performs any necessary calibration
        
        Args:
            sensor_data (dict): Initial sensor readings (same format as update_robot_state)
        """
        # Update robot state with initial sensor data
        self.update_robot_state(sensor_data)
        
        # Initialize control architecture with current state
        # This sets up the state machine and trajectory managers
        self._control_architecture.initialize()

    # ====================================================================== 
    # save_trajectory 
    # ====================================================================== 

    def save_trajectory(self):
        """
        Save the current trajectory data for analysis or debugging.
        Delegates to control architecture's save_trajectory method.
        """
        # Save trajectory data through control architecture
        self._control_architecture.save_trajectory()
