# ==================================================================== 
# Direct and inverse geometry of 3d robots 
# ==================================================================== 

# ---------------------------------- 
# Import necessary libraries 
# ---------------------------------- 

import os 
import sys 
sys.path.append( os.getcwd() ) 
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

# load urdf (from apt paquet robotpkg-example-robot-data 
# stored in /opt/openrobots/share/example-robot-data) 
robot = robex.load('ur5') 

# show kinematic tree 
print(robot.model) 

# import class RbootWrapper and create an instance in terminal 
# /opt/openrobots/lib/python2.7/site-packages/pinocchio/robot_wrapper.py 
# idk where in tutorial they do this 

# how to get index of a joint 
joint_idx = robot.index('wrist_3_joint')
# print(f"joint idx = {joint_idx}") 
print("joint idx = ", joint_idx)
print("joint name = " + robot.model.names[joint_idx]) 

# robot.model.names is a container for all the joint names 
for i, n in enumerate(robot.model.names): 
    print(i, n) 

# robot.model.frames is a container for all the frames     
for f in robot.model.frames: 
    print(f.name, "attached to joint #", f.parent) 
    
# robot.placement(idx) and robot.framePlacement(idx) returns placement 
# placement is translation + rotation of joint / frame in argument 
a = robot.placement(robot.q0, 6)        # placement of end effector joint 
b = robot.framePlacement(robot.q0, 22)  # placement of end effector tip 

tool_axis = b.rotation[:, 2] # axis of the tool 

print('a = ', a) 
print('b = ', b) 
print('tool_axis = ', tool_axis) 

# dimension of the config space (i.e. the # of joints) is given in: 
NQ = robot.model.nq     # number of generalized coordinates (DOF) 
NV = robot.model.nv     # number of generalized velocities 

# for this simple robot, NQ = NV = 6 
print('NQ = ', NQ) 
print('NV = ', NV) 

# ---------------------------------- 
# Display simple geometries 
# ---------------------------------- 

# load viewer 
from utils.meshcat_viewer_wrapper import MeshcatVisualizer, colors 

viz = MeshcatVisualizer(robot) 

# config q can be displayed 
q_config = np.array( [ -1., -1.5, 2.1, -.5, -.5, 0 ] )
viz.display(q_config) 

# can display other geometries as well - add a red box 
ballID = "world/ball" 
radius = 0.1 
viz.addSphere(ballID, radius, colors.red) 

# place ball at (0.5, 0.1, 0.2) 
# viewer expects pos and rotation, append identity quaternion 
r_ball = np.array( [ 0.5, 0.1, 0.2 ] ) 
pose_ball = r_ball.tolist() + [ 1., 0., 0., 0. ] 
viz.applyConfiguration(ballID, pose_ball) 

print(pose_ball) 

# ---------------------------------- 
# Pick and place 3D 
# ---------------------------------- 

# use inverse geometry to find config of robot so that 
# end effector touches ball using unconstrained optimization 

# implement a cost function that takes config as argument and 
# returns squared distance between end effector tip (frame 22) 
# and ball, accounting for we want to touch the ball on boundary 
# in the natural direction (e_z axis of frame) 

frame_tip = robot.framePlacement(q_config, 22)  # SE(3) element frame of tip 
r_tip     = frame_tip.translation               # position of the tip 
e_z_axis  = frame_tip.rotation[:, 2]            # direction of the tip  
offset    = frame_tip.rotation[:, 2] * radius   # direction of the tip 

print("SE(3) element frame fo tip = ", frame_tip)
print("position of the tip = ", r_tip) 
print("direction of the tip = ", e_z_axis) 
print("offset = ", offset) 

# set target 
target = np.array(r_ball) 

# reset robot config 
q_config = np.array( [ -1., -1.5, 2.1, -.5, -.5, 0 ] ) 
viz.display(q_config) 

# define cost function 
def compute_cost(q): 
    '''Compute score from a configuration ''' 
    frame_tip = robot.framePlacement(q_config, 22)  # SE(3) element frame of tip 
    r_tip     = frame_tip.translation               # position of the tip 
    offset    = frame_tip.rotation[:, 2] * radius   # direction of the tip 
    return norm( r_tip + offset - target )**2 

def display_callback(q): 
    '''Display robot at each iteration''' 
    viz.display(q) 
    time.sleep(1e-1) 
    
# optimize 
q_touch  = fmin_bfgs( compute_cost, robot.q0 ) 
callback = display_callback(q_touch)   

# save copy of solution 
q_config = q_touch.copy() 

# get end effector placement 
robot.placement(q_config, 6).translation 

# now choose any trajectory you want in config space 
# make a for loop to display the robot at sampling positions 
# function sleep can be used to slow down loop 

# at each instant, recompute the position of the ball 
# and display it so it always "sticks" to the robot end effector 

vq  = np.array( [ 2., 0, 0, 4., 0, 0 ] ) 
idx = 6 

pose_end_eff = robot.placement(q_config, idx) 
r_end_eff    = pose_end_eff.translation     # position of end-eff in world frame 
r_ball       = pose_ball[:3]                # position of ball in world frame 
print("pose_end_eff = ", pose_end_eff) 
print("pose end eff rotation = ", pose_end_eff.rotation.T) 

r_end_ball_N = r_ball - r_end_eff       # Relative position of ball center wrt end effector position, express in world frame 

E_DCM_N      = pose_end_eff.rotation.T  # Rotation matrix from end effector to world frame 
r_end_ball_E = E_DCM_N @ r_end_ball_N   # Position of ball wrt eff in local coordinate

for i in range(200):
    # Chose new configuration of the robot
    q_config += vq / 40
    q_config[2] = 1.71 + math.sin(i * 0.05) / 2 

    # Gets the new position of the ball 
    pose_end_eff = robot.placement(q_config, idx)
    r_ball = pose_end_eff * r_end_ball_E  # Apply oMend to the relative placement of ball

    # Display new configuration for robot and ball
    viz.applyConfiguration(ballID, r_ball.tolist() + [1, 0, 0, 0])
    viz.display(q_config)
    time.sleep(1e-2)
    
# ---------------------------------- 
# Pick and place 6D 
# ---------------------------------- 

# let's say object is now rectangle and not a sphere 
boxID = "world/box"
try:
    viz.delete(ballID)
except:
    pass
viz.addBox(boxID, [0.1, 0.2, 0.1], colors.magenta)

# Place the box at the position (0.5, 0.1, 0.2) with no rotation
oMbox = pin.SE3(np.eye(3), np.array([0.5, 0.1, 0.2]))  # x,y,z 

viz.applyConfiguration(boxID, oMbox)

# 6D means translation and rotation 
# use SE(3) log function to score distance between two placements 

# Relative placement of the left facet with the z-axis orthogonal to the facet pointing inside the box
boxMtarget = pin.SE3(pin.utils.rotate('x', -np.pi / 2), np.array([0., -0.1, 0.]))
# Placement of the facet in the world
oMtarget = oMbox * boxMtarget 

viz.applyConfiguration(boxID, oMbox)

def compute_cost(q):
    '''Compute score from a configuration'''
    oMtip = robot.framePlacement(q, 22)
    # Align tip placement and facet placement
    return norm(pin.log(oMtip.inverse() * oMtarget).vector)

def callback(q):
    viz.display(q)
    time.sleep(1e-1)

qopt = fmin_bfgs(compute_cost, robot.q0, callback=callback)

print('The robot finally reached effector placement at\n', robot.placement(qopt, 6))

# move the box following the motion 
q_config = qopt.copy()
vq = np.array([2., 0, 0, 4., 0, 0])
idx = 6

pose_end_eff = robot.placement(q_config, idx)
# TODO: Compute the placement of the box wrt the end effector frame
endMbox = pin.SE3()  # Placement of the box wrt end effector

for i in range(100):
    # Chose new configuration of the robot
    q_config += vq / 40
    q_config[2] = 1.71 + math.sin(i * 0.05) / 2

    # TODO: replace with good calculation
    oMbox = oMbox

    # Display new configuration for robot and box
    viz.applyConfiguration(boxID, oMbox)
    viz.display(q_config)
    time.sleep(1e-2)

# ---------------------------------- 
# Inverse geometry while taking collisions into account 
# ---------------------------------- 

# use constrained optimization solver (just like 1st lesson) 

# ---------------------------------- 
# Optimizing in the quaternion space 
# ---------------------------------- 

robot = robex.load('solo12')
viz   = MeshcatVisualizer(robot)
viz.viewer.jupyter_cell()

viz.display(robot.q0) 
robot.feetIndexes = [robot.model.getFrameId(frameName) for frameName in ['HR_FOOT', 'HL_FOOT', 'FR_FOOT', 'FL_FOOT']]

# --- Add box to represent target
colors = ['red', 'blue', 'green', 'magenta']
for color in colors:
    viz.addSphere("world/%s" % color, .05, color)
    viz.addSphere("world/%s_des" % color, .05, color)

#
# OPTIM 6D #########################################################
#

targets = [
    np.array([-0.7, -0.2, 1.2]),
    np.array([-0.3, 0.5, 0.8]),
    np.array([0.3, 0.1, -0.1]),
    np.array([0.9, 0.9, 0.5])
]
for i in range(4):
    targets[i][2] += 1


def compute_cost(q):
    '''Compute score from a configuration'''
    cost = 0.
    for i in range(4):
        p_i = robot.framePlacement(q, robot.feetIndexes[i]).translation
        cost += norm(p_i - targets[i])**2
    return cost


def display_callback(q):
    viz.applyConfiguration('world/box', Mtarget)

    for i in range(4):
        p_i = robot.framePlacement(q, robot.feetIndexes[i])
        viz.applyConfiguration('world/%s' % colors[i], p_i)
        viz.applyConfiguration('world/%s_des' % colors[i], list(targets[i]) + [1, 0, 0, 0])

    viz.display(q)
    time.sleep(1e-2)


Mtarget = pin.SE3(pin.utils.rotate('x', 3.14 / 4), np.array([0.5, 0.1, 0.2]))  # x,y,z
qopt = fmin_bfgs(compute_cost, robot.q0, callback=display_callback)

# ---------------------------------- 
# configuration of parallel robots 
# ----------------------------------


# ==================================================================== 
# KEEP SCRIPT RUNNING 
# ==================================================================== 

print("Keep Meshcat server alive") 

while True: 
    time.sleep(1)

