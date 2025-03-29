import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from config.crab_config import WalkingState
from pnc.state_machine import StateMachine
from pnc.crab_pnc.crab_state_provider import CrabStateProvider

from util import util

# ====================================================================== 
# class Nothing 
# ====================================================================== 

class Nothing(StateMachine):
    """
    State machine for the Crab robot when it is doing absolutely nothing.
    """
    def __init__(self, id, tm, hm, fm, robot):
        super(Nothing, self).__init__(id, robot)
        self._trajectory_managers = tm
        self._hierarchy_managers = hm
        self._force_managers = fm
        self._robot = robot
        self._sp = CrabStateProvider()
        self._start_time = 0.
        self._duration = 3.0  # Duration in seconds before transitioning

    # ====================================================================== 
    # first_visit 
    # ====================================================================== 

    def first_visit(self):
        print("[WalkingState] NOTHING - Doing absolutely nothing")
        self._start_time = self._sp.curr_time 
        
        # Get current COM position
        com_pos_des = self._robot.get_com_pos()
        
        # Get current base orientation as quaternion
        base_iso = self._robot.get_link_iso("base_link")
        base_quat_des = util.rot_to_quat(base_iso[0:3, 0:3])
        
        # Initialize trajectory to maintain current position/orientation
        self._trajectory_managers[
            "floating_base"].initialize_floating_base_interpolation_trajectory(
                self._sp.curr_time, self._sp.curr_time + self._duration, 
                com_pos_des, base_quat_des)

    # ====================================================================== 
    # one_step 
    # ====================================================================== 

    def one_step(self):
        # Just track time, do absolutely nothing else
        self._state_machine_time = self._sp.curr_time - self._start_time
        
        # To prevent errors, ensure trajectory managers keep using current values
        # Update Floating Base Task
        self._trajectory_managers[
            "floating_base"].update_floating_base_desired(self._sp.curr_time)
        
        self._trajectory_managers["lfoot"].use_current()
        self._trajectory_managers["rfoot"].use_current()
        
        # # Update floating base task
        # self._trajectory_managers["dcm"].update_floating_base_task_desired(
        #     self._sp.curr_time)
                
    # ====================================================================== 
    # last_visit 
    # ====================================================================== 

    def last_visit(self):
        print("[WalkingState] NOTHING - Finished doing nothing")
        pass

    # ====================================================================== 
    # end_of_state 
    # ====================================================================== 

    def end_of_state(self):
        # End the state after duration seconds
        if self._state_machine_time > self._duration:
            return True
        else:
            return False

    # ====================================================================== 
    # get_next_state 
    # ====================================================================== 

    def get_next_state(self):
        # Transition to STAND state after doing nothing
        # return WalkingState.STAND
        return WalkingState.NOTHING 
