import numpy as np

# from util import util
from pnc.data_saver import DataSaver
from config.crab_config import PnCConfig, WBCConfig
from pnc.wbc.ihwbc.ihwbc import IHWBC
# from pnc.wbc.ihwbc.ihwbc2 import IHWBC2
from pnc.wbc.ihwbc.joint_integrator import JointIntegrator

# ====================================================================== 
# class CrabController 
# ====================================================================== 

class CrabController(object):
    """
    Main controller class for the Crab robot. Coordinates the interface,
    robot system, and generates control commands.

    Control flow:
    1. Receive sensor data
    2. Update robot state
    3. Generate commands through interface
    4. Apply safety checks
    5. Send commands to actuators
    """
    
    def __init__(self, tci_container, robot):
        """
        Initialize the controller with robot system and interface.
        Sets up:
        - Robot model and state
        - Control interface
        - Command buffers
        - Safety parameters
        """
        
        self._tci_container = tci_container
        self._robot = robot 
        
        # Print to debug
        print(f"Robot DOFs: n_floating={robot.n_floating}, n_a={robot.n_a}, total={robot.n_q_dot}") 

        # Initialize WBC
        # l_jp_idx, l_jd_idx, r_jp_idx, r_jd_idx = self._robot.get_q_dot_idx(
        #     ['l_knee_fe_jp', 'l_knee_fe_jd', 'r_knee_fe_jp', 'r_knee_fe_jd'])
        # These are the wrist joints which might need special handling as passive joints
        l_jp_idx, l_jd_idx, r_jp_idx, r_jd_idx = self._robot.get_q_dot_idx(
            ['front_left__cluster_3_pitch', 'front_left__cluster_3_wrist', 
            'front_right__cluster_3_pitch', 'front_right__cluster_3_wrist'])
        act_list = [False] * robot.n_floating + [True] * robot.n_a
        # act_list[l_jd_idx] = False
        # act_list[r_jd_idx] = False

        # ---------------------------------- 
        # DOF breakdown
        # ---------------------------------- 
        n_q_dot   = len(act_list)
        n_active  = np.count_nonzero(np.array(act_list))
        n_passive = n_q_dot - n_active - 6 
        
        print(f"DOF breakdown: n_q_dot={n_q_dot}, n_active={n_active}, n_passive={n_passive}") 

        # ---------------------------------- 
        # Active / Passive Selection Matrices 
        # ---------------------------------- 
        
        # sa: active selection matrix
        self._sa = np.zeros((n_active, n_q_dot))
        
        # sv: passive selection matrix
        self._sv = np.zeros((n_passive, n_q_dot))
        
        # j: active index
        # k: passive index
        j, k = 0, 0
        for i in range(n_q_dot):
            if i >= 6:
                if act_list[i]:
                    self._sa[j, i] = 1.
                    j += 1
                else:
                    self._sv[k, i] = 1.
                    k += 1

        # sf: floating base selection matrix
        self._sf = np.zeros((6, n_q_dot))
        self._sf[0:6, 0:6] = np.eye(6)

        # ---------------------------------- 
        # Initialize IHWBC
        # ---------------------------------- 
        self._ihwbc = IHWBC(self._sf, self._sa, self._sv, PnCConfig.SAVE_DATA)
        if WBCConfig.B_TRQ_LIMIT:
            self._ihwbc.trq_limit = np.dot(self._sa[:, 6:],
                                           self._robot.joint_trq_limit)
        self._ihwbc.lambda_q_ddot = WBCConfig.LAMBDA_Q_DDOT
        self._ihwbc.lambda_rf     = WBCConfig.LAMBDA_RF

        # ---------------------------------- 
        # Initialize Joint Integrator
        # ---------------------------------- 
        self._joint_integrator = JointIntegrator(robot.n_a,
                                                 PnCConfig.CONTROLLER_DT)
        self._joint_integrator.pos_cutoff_freq = WBCConfig.POS_CUTOFF_FREQ
        self._joint_integrator.vel_cutoff_freq = WBCConfig.VEL_CUTOFF_FREQ
        self._joint_integrator.max_pos_err = WBCConfig.MAX_POS_ERR
        self._joint_integrator.joint_pos_limit = self._robot.joint_pos_limit
        self._joint_integrator.joint_vel_limit = self._robot.joint_vel_limit

        self._b_first_visit = True

        if PnCConfig.SAVE_DATA:
            self._data_saver = DataSaver()

    # ====================================================================== 
    # get_command 
    # ====================================================================== 

    def get_command(self):
        """
        Generate control commands for the current timestep.
        
        Process:
        1. Get raw commands from interface
        2. Apply safety checks and limits
        3. Store commands for next iteration
        
        Returns:
            dict: Safe control commands including positions, velocities, and torques
        """
        
        if self._b_first_visit:
            self.first_visit()

        # ---------------------------------- 
        # Dynamics properties
        # ---------------------------------- 
        mass_matrix     = self._robot.get_mass_matrix()
        mass_matrix_inv = np.linalg.inv(mass_matrix)
        coriolis        = self._robot.get_coriolis()
        gravity         = self._robot.get_gravity()
        
        self._ihwbc.update_setting(mass_matrix, mass_matrix_inv, coriolis, gravity)
        
        # ---------------------------------- 
        # Task, Contact, and Internal Constraint Setup
        # ---------------------------------- 
        w_hierarchy_list = []
        for task in self._tci_container.task_list:
            # display task name 
            print(f"Task: {task._target_id}")
            print(f"Task type: {task._task_type}")
            print(f"Task pos des: {task._pos_des}")
            task.update_jacobian()
            task.update_cmd()
            w_hierarchy_list.append(task.w_hierarchy)
        self._ihwbc.w_hierarchy = np.array(w_hierarchy_list)
        for contact in self._tci_container.contact_list:
            contact.update_contact()
        for internal_constraint in self._tci_container.internal_constraint_list:
            internal_constraint.update_internal_constraint()
            
        # ---------------------------------- 
        # WBC commands - solve IHWBC 
        # ---------------------------------- 
        joint_trq_cmd, joint_acc_cmd, rf_cmd = self._ihwbc.solve(
            self._tci_container.task_list, self._tci_container.contact_list,
            self._tci_container.internal_constraint_list)
        joint_trq_cmd = np.dot(self._sa[:, 6:].transpose(), joint_trq_cmd)
        joint_acc_cmd = np.dot(self._sa[:, 6:].transpose(), joint_acc_cmd)
        
        # Double integration
        joint_vel_cmd, joint_pos_cmd = self._joint_integrator.integrate(
            joint_acc_cmd, self._robot.joint_velocities,
            self._robot.joint_positions)

        # ---------------------------------- 
        # Save data
        # ---------------------------------- 
        if PnCConfig.SAVE_DATA:
            self._data_saver.add('joint_trq_cmd', joint_trq_cmd)

        command = self._robot.create_cmd_ordered_dict(
            joint_pos_cmd, joint_vel_cmd, joint_trq_cmd)
        return command
    
    # ====================================================================== 
    # first_visit 
    # ====================================================================== 

    def first_visit(self):
        """
        Initialize the controller states.
        
        Sets up:
        - Joint integrator initial states
        - First visit flag
        """
        
        joint_pos_ini = self._robot.joint_positions
        self._joint_integrator.initialize_states(
            np.zeros(self._robot.n_a), joint_pos_ini)

        self._b_first_visit = False
