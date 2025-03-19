# ==================================================================== 
# Direct and inverse geometry of 3d robots 
# ==================================================================== 

# ---------------------------------- 
# Import necessary libraries 
# ---------------------------------- 

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

# load functions for crab robot 
from crab_fns import get_initial_pose0, get_initial_pose

# ---------------------------------- 
# Kinematic tree in Pinocchio 
# ---------------------------------- 

# Load robot
crab_urdf_file = cwd + "/robot_model/crab/crab.urdf"
package_dir    = cwd + "/robot_model/crab/"
rob_model, col_model, vis_model = pin.buildModelsFromUrdf(crab_urdf_file,
                                                          package_dir, pin.JointModelFreeFlyer())

# remove gravity
rob_model.gravity = pin.Motion.Zero()

# create data 
rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)

# initialize meshcat visualizer 
viz = MeshcatVisualizer(rob_model, col_model, vis_model)

viz.initViewer(open=True)
viz.loadViewerModel(rootNodeName="crab") 

# display initial configuration angles = 0 
q_config0 = get_initial_pose0()
viz.display(q_config0)

# config q can be displayed with different initial angles 
q_config = get_initial_pose() 
viz.display(q_config) 

# ==================================================================== 
# KEEP SCRIPT RUNNING 
# ==================================================================== 

print("Keep Meshcat server alive") 

while True: 
    time.sleep(1)
