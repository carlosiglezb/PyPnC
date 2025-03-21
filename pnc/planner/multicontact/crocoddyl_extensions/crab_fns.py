import os 
import sys 

cwd = os.getcwd()
sys.path.append(cwd)

import pdb 

import time 
import math 
import numpy as np 
from numpy.linalg import norm 
import pinocchio as pin 
import example_robot_data as robex 
from scipy.optimize import fmin_bfgs 
from pinocchio.visualize import MeshcatVisualizer 

import crocoddyl 

# ---------------------------------- 
# Initial joint angles = 0   
# ---------------------------------- 

def get_initial_pose0():

    q0 = np.zeros(35,)

    q0[0] = 0.      # root_joint x 
    q0[1] = 0.      # root_joint y 
    q0[2] = 0.      # root_joint z 
    q0[3] = 0.      # root_joint q1
    q0[4] = 0.      # root_joint q2
    q0[5] = 0.      # root_joint q3
    q0[6] = 1.      # root_joint q4 
    q0[7] = 0.      # back_left__cluster_1_roll 
    q0[8] = 0.      # back_left__cluster_1_pitch 
    q0[9] = 0.      # back_left__cluster_2_roll 
    q0[10] = 0.     # back_left__cluster_2_pitch 
    q0[11] = 0.     # back_left__cluster_3_roll 
    q0[12] = 0.     # back_left__cluster_3_pitch 
    q0[13] = 0.     # back_left__cluster_3_wrist  
    q0[14] = 0.     # back_right__cluster_1_roll 
    q0[15] = 0.     # back_right__cluster_1_pitch 
    q0[16] = 0.     # back_right__cluster_2_roll 
    q0[17] = 0.     # back_right__cluster_2_pitch 
    q0[18] = 0.     # back_right__cluster_3_roll 
    q0[19] = 0.     # back_right__cluster_3_pitch 
    q0[20] = 0.     # back_right__cluster_3_wrist 
    q0[21] = 0.     # front_left__cluster_1_roll 
    q0[22] = 0.     # front_left__cluster_1_pitch 
    q0[23] = 0.     # front_left__cluster_2_roll 
    q0[24] = 0.     # front_left__cluster_2_pitch 
    q0[25] = 0.     # front_left__cluster_3_roll 
    q0[26] = 0.     # front_left__cluster_3_pitch 
    q0[27] = 0.     # front_left__cluster_3_wrist 
    q0[28] = 0.     # front_right__cluster_1_roll 
    q0[29] = 0.     # front_right__cluster_1_pitch 
    q0[30] = 0.     # front_right__cluster_2_roll 
    q0[31] = 0.     # front_right__cluster_2_pitch 
    q0[32] = 0.     # front_right__cluster_3_roll 
    q0[33] = 0.     # front_right__cluster_3_pitch 
    q0[34] = 0.     # front_right__cluster_3_wrist 

    # floating_base = np.array([0., 0., 0., 0., 0., 0., 1.])
    # return np.concatenate((floating_base, q0))
    return q0 

# ---------------------------------- 
# Initial joint angles = 10, -45   
# ---------------------------------- 

def get_initial_pose(
    roll_angle  = np.radians(10), 
    pitch_angle = np.radians(-45)
    ):

    q0 = np.zeros(35,)

    q0[0] = 0.                  # root_joint x 
    q0[1] = 0.                  # root_joint y 
    q0[2] = 0.                  # root_joint z 
    q0[3] = 0.                  # root_joint q1
    q0[4] = 0.                  # root_joint q2
    q0[5] = 0.                  # root_joint q3
    q0[6] = 1.                  # root_joint q4 
    q0[7] = roll_angle          # back_left__cluster_1_roll 
    q0[8] = pitch_angle         # back_left__cluster_1_pitch 
    q0[9] = roll_angle          # back_left__cluster_2_roll 
    q0[10] = pitch_angle        # back_left__cluster_2_pitch 
    q0[11] = roll_angle         # back_left__cluster_3_roll 
    q0[12] = pitch_angle        # back_left__cluster_3_pitch 
    q0[13] = 0.                 # back_left__cluster_3_wrist  
    q0[14] = roll_angle         # back_right__cluster_1_roll 
    q0[15] = pitch_angle        # back_right__cluster_1_pitch 
    q0[16] = roll_angle         # back_right__cluster_2_roll 
    q0[17] = pitch_angle        # back_right__cluster_2_pitch 
    q0[18] = roll_angle         # back_right__cluster_3_roll 
    q0[19] = pitch_angle        # back_right__cluster_3_pitch 
    q0[20] = 0.                 # back_right__cluster_3_wrist 
    q0[21] = roll_angle         # front_left__cluster_1_roll 
    q0[22] = pitch_angle        # front_left__cluster_1_pitch 
    q0[23] = 0.                 # front_left__cluster_2_roll 
    q0[24] = pitch_angle        # front_left__cluster_2_pitch 
    q0[25] = roll_angle         # front_left__cluster_3_roll 
    q0[26] = pitch_angle        # front_left__cluster_3_pitch 
    q0[27] = 0.                 # front_left__cluster_3_wrist 
    q0[28] = roll_angle         # front_right__cluster_1_roll 
    q0[29] = pitch_angle        # front_right__cluster_1_pitch 
    q0[30] = roll_angle         # front_right__cluster_2_roll 
    q0[31] = pitch_angle        # front_right__cluster_2_pitch 
    q0[32] = roll_angle         # front_right__cluster_3_roll 
    q0[33] = pitch_angle        # front_right__cluster_3_pitch 
    q0[34] = 0.                 # front_right__cluster_3_wrist 

    return q0  

