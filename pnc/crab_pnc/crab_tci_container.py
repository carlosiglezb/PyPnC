import numpy as np

from config.crab_config import WBCConfig, PnCConfig
# from pnc.crab_pnc.crab_rolling_joint_constraint import CrabRollingJointConstraint
from pnc.wbc.tci_container import TCIContainer
from pnc.wbc.basic_task import BasicTask
from pnc.wbc.basic_contact import SurfaceContact

# ====================================================================== 
# class CrabTCIContainer 
# ====================================================================== 

class CrabTCIContainer(TCIContainer):
    """
    Task-Contact-Internal Constraint Container for the Crab robot.
    
    This container:
    1. Defines all control tasks (COM, feet, torso orientation, etc.)
    2. Sets up contact constraints for the feet
    3. Manages task hierarchies through weights
    4. Provides interface between high-level control and whole-body controller
    
    The tasks are used by the whole-body controller to generate joint commands
    that achieve desired motions while respecting constraints.
    """
    
    # ====================================================================== 
    # __init__ 
    # ====================================================================== 

    def __init__(self, robot):
        """
        Initialize the TCI container with tasks and contacts.

        Args:
            robot: Robot instance containing model and state information
        """
        super(CrabTCIContainer, self).__init__(robot)

        # ---------------------------------- 
        # Initialize Tasks
        # ---------------------------------- 
        
        # COM (Center of Mass) Task
        # Controls the robot's overall balance and position
        self._com_task = BasicTask(robot, "COM", 3, 'com', PnCConfig.SAVE_DATA)
        self._com_task.kp = WBCConfig.KP_COM  # Position gain
        self._com_task.kd = WBCConfig.KD_COM  # Velocity gain
        self._com_task.w_hierarchy = WBCConfig.W_COM  # Task priority weight

        # Torso Orientation Task
        # Maintains desired orientation of the robot's main body
        self._torso_ori_task = BasicTask(robot, "LINK_ORI", 3,
                                       "base_link", PnCConfig.SAVE_DATA)
        self._torso_ori_task.kp = WBCConfig.KP_TORSO
        self._torso_ori_task.kd = WBCConfig.KD_TORSO
        self._torso_ori_task.w_hierarchy = WBCConfig.W_TORSO

        # Upperbody joints
        # upperbody_joint = [
        #     'neck_pitch', 'l_shoulder_fe', 'l_shoulder_aa', 'l_shoulder_ie',
        #     'l_elbow_fe', 'l_wrist_ps', 'l_wrist_pitch', 'r_shoulder_fe',
        #     'r_shoulder_aa', 'r_shoulder_ie', 'r_elbow_fe', 'r_wrist_ps',
        #     'r_wrist_pitch'
        # ]
        # self._upper_body_task = BasicTask(robot, "SELECTED_JOINT",
        #                                   len(upperbody_joint),
        #                                   upperbody_joint, PnCConfig.SAVE_DATA)
        # self._upper_body_task.kp = WBCConfig.KP_UPPER_BODY
        # self._upper_body_task.kd = WBCConfig.KD_UPPER_BODY
        # self._upper_body_task.w_hierarchy = WBCConfig.W_UPPER_BODY

        # Rfoot Pos Task
        self._rfoot_pos_task = BasicTask(robot, "LINK_XYZ", 3,
                                       "front_right__foot_link", PnCConfig.SAVE_DATA)
        self._rfoot_pos_task.kp = WBCConfig.KP_FOOT_POS
        self._rfoot_pos_task.kd = WBCConfig.KD_FOOT_POS
        self._rfoot_pos_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT

        # Lfoot Pos Task
        self._lfoot_pos_task = BasicTask(robot, "LINK_XYZ", 3,
                                       "front_left__foot_link", PnCConfig.SAVE_DATA)
        self._lfoot_pos_task.kp = WBCConfig.KP_FOOT_POS
        self._lfoot_pos_task.kd = WBCConfig.KD_FOOT_POS
        self._lfoot_pos_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT

        # Rfoot Ori Task
        self._rfoot_ori_task = BasicTask(robot, "LINK_ORI", 3,
                                        "front_right__foot_link", PnCConfig.SAVE_DATA)
        self._rfoot_ori_task.kp = WBCConfig.KP_FOOT_ORI
        self._rfoot_ori_task.kd = WBCConfig.KD_FOOT_ORI
        self._rfoot_ori_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT

        # Lfoot Ori Task
        self._lfoot_ori_task = BasicTask(robot, "LINK_ORI", 3,
                                       "front_left__foot_link", PnCConfig.SAVE_DATA)
        self._lfoot_ori_task.kp = WBCConfig.KP_FOOT_ORI
        self._lfoot_ori_task.kd = WBCConfig.KD_FOOT_ORI
        self._lfoot_ori_task.w_hierarchy = WBCConfig.W_CONTACT_FOOT

        # ---------------------------------- 
        # Initialize Contacts
        # ---------------------------------- 
            
        # Right Foot Contact
        # Defines contact constraints and friction cone for right foot
        self._rfoot_contact = SurfaceContact(robot, "font_right__foot_link", 0.115,
                                           0.065, 0.3, PnCConfig.SAVE_DATA)
        self._rfoot_contact.rf_z_max = 1e-3  # Maximum normal force

        # Left Foot Contact
        # Defines contact constraints and friction cone for left foot
        self._lfoot_contact = SurfaceContact(robot, "font_left__foot_link", 0.115,
                                           0.065, 0.3, PnCConfig.SAVE_DATA)
        self._lfoot_contact.rf_z_max = 1e-3  # Maximum normal force

        # ---------------------------------- 
        # Initialize Task Hierarchy
        # ---------------------------------- 
        
        # List of all tasks in order of priority
        self._task_list = [
            self._com_task, self._torso_ori_task, 
            # self._upper_body_task,
            self._rfoot_pos_task, self._lfoot_pos_task, self._rfoot_ori_task,
            self._lfoot_ori_task
        ]

        # List of all contact constraints
        self._contact_list = [self._rfoot_contact, self._lfoot_contact]

        # ---------------------------------- 
        # Initialize Internal Constraint
        # ---------------------------------- 
        # self._rolling_joint_constraint = CrabRollingJointConstraint(robot)
        # self._internal_constraint_list = [self._rolling_joint_constraint]

    # ======================================================================
    # Property Getters
    # These provide access to tasks and contacts for the controller
    # ======================================================================

    @property
    def com_task(self):
        """Center of Mass task for balance control"""
        return self._com_task

    @property
    def torso_ori_task(self):
        """Torso orientation task for body posture"""
        return self._torso_ori_task

    # @property
    # def upper_body_task(self):
    #     return self._upper_body_task

    @property
    def rfoot_pos_task(self):
        """Right foot position task"""
        return self._rfoot_pos_task

    @property
    def lfoot_pos_task(self):
        """Left foot position task"""
        return self._lfoot_pos_task

    @property
    def rfoot_ori_task(self):
        """Right foot orientation task"""
        return self._rfoot_ori_task

    @property
    def lfoot_ori_task(self):
        """Left foot orientation task"""
        return self._lfoot_ori_task

    @property
    def rfoot_contact(self):
        """Right foot contact constraint"""
        return self._rfoot_contact

    @property
    def lfoot_contact(self):
        """Left foot contact constraint"""
        return self._lfoot_contact

    @property
    def task_list(self):
        """List of all tasks in priority order"""
        return self._task_list

    @property
    def contact_list(self):
        """List of all contact constraints"""
        return self._contact_list

    @property
    def internal_constraint_list(self):
        """List of internal constraints (empty for Crab)"""
        return []
