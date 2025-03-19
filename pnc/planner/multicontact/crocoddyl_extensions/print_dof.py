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

# ---------------------------------- 
# Kinematic tree in Pinocchio 
# ---------------------------------- 

# load robot model 
robot = robex.load('crab') 

# show kinematic tree 
print(robot.model) 

# import class RbootWrapper and create an instance in terminal 
# /opt/openrobots/lib/python2.7/site-packages/pinocchio/robot_wrapper.py 
# idk where in tutorial they do this 

# # how to get index of a joint 
# joint_idx = robot.index('wrist_3_joint')
# # print(f"joint idx = {joint_idx}") 
# print("joint idx = ", joint_idx)
# print("joint name = " + robot.model.names[joint_idx]) 

# robot.model.names is a container for all the joint names 
print("Joints:") 
for i, n in enumerate(robot.model.names): 
    print(i, n) 

# robot.model.frames is a container for all the frames    
print("Frames:") 
for f in robot.model.frames: 
    print(f.name, "attached to joint #", f.parent) 
    
# ---------------------------------- 
# Print detailed information about each degree of freedom in the model 
# ---------------------------------- 
def print_dof_info(model):
    """Print detailed information about each degree of freedom in the model."""
    print("\n==== Degrees of Freedom Information ====")
    print(f"Total DoFs (model.nq): {model.nq}")

    # Track the current index in the configuration vector
    q_idx = 0

    # Loop through all joints
    for joint_id in range(model.njoints):
        joint_name = model.names[joint_id]
        joint = model.joints[joint_id]
        joint_nq = joint.nq  # Number of position variables for this joint
        joint_nv = joint.nv  # Number of velocity variables for this joint
        
        # Skip joints with no DoFs (like the "universe" joint)
        if joint_nq == 0:
            continue
            
        print(f"\nJoint {joint_id}: {joint_name}")
        print(f"  - Type: {joint.shortname()}")
        print(f"  - DoFs: {joint_nq} position variables, {joint_nv} velocity variables")
        
        # For free flyer (floating base), describe each DoF
        if joint.shortname() == "JointModelFreeFlyer":
            print("  - DoF mapping:")
            print("    [0-2]: Position (x, y, z)")
            print("    [3-6]: Orientation (quaternion)")
            q_range = list(range(q_idx, q_idx + joint_nq))
            print(f"  - Configuration indices: {q_range}")
        
        # For revolute and prismatic joints
        elif joint_nq == 1:
            print(f"  - Configuration index: {q_idx}")
            # print(f"  - Axis: {joint.axis}")  # Show rotation/translation axis
            
        # For other joint types (spherical, etc.)
        else:
            q_range = list(range(q_idx, q_idx + joint_nq))
            print(f"  - Configuration indices: {q_range}")
            
        # Increment index counter
        q_idx += joint_nq

print("\n==== Total: {0} degrees of freedom ====".format(robot.model.nq))

# Use it with your model
print_dof_info(robot.model)