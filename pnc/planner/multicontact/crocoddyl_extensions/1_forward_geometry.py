# Stand-alone program to optimize the configuration q_config=[q_config1,q_config2] of a 2-R robot with scipy BFGS.

# ---------------------------------- 
# Import necessary libraries 
# ---------------------------------- 

import os 
import sys 
sys.path.append( os.getcwd() ) 
import pdb 

import time
import numpy as np
from scipy.optimize import fmin_bfgs, fmin_slsqp
from robot_utils.meshcat_viewer_wrapper import MeshcatVisualizer, translation2d,planar
from numpy.linalg import norm, inv, pinv, svd, eig 

# ==================================================================== 
# REDUCED CONFIGURATION 
# ==================================================================== 

# ---------------------------------- 
# Displaying objects and create a 2 DOF robot 
# ---------------------------------- 

viz = MeshcatVisualizer() 
viz.viewer.jupyter_cell() 

# add objects 
ball_id = 'world/ball'; viz.addSphere( ball_id, .2, [1, 0, 0, 1] )
cyl_id  = 'world/cyl' ; viz.addCylinder( cyl_id, length = 1, radius  = .1, color = [0, 0, 1, 1] ) 
box_id  = 'world/box' ; viz.addBox( box_id, [.5, .2, .4], [1, 1, 0, 1] )  

# delete objects 
viz.delete(ball_id) 

# place objects using set_transform comand, specifying displacement of 7 values
# - first 3 values are position (x, y, z) 
# - last 4 values are q_configuaternion (q_config1, q_config2, q_config3, q_config4) 
pos  = [.1, .2, .3] 
quat = [1, 0, 0, 0] 
viz.applyConfiguration( cyl_id, pos + quat) 

# ---------------------------------- 
# Create a 2 DOF robot 
# ---------------------------------- 

# delete previous objects 
viz.delete(ball_id) 
viz.delete(cyl_id) 
viz.delete(box_id) 

# add robot joints 
joint1 = 'joint1'; viz.addSphere( joint1, .1, [1, 0, 0, 1] ) 
joint2 = 'joint2'; viz.addSphere( joint2, .1, [1, 0, 0, 1] ) 
joint3 = 'joint3'; viz.addSphere( joint3, .1, [1, 0, 0, 1] ) 

# add robot arms 
arm1 = 'arm1'; viz.addCylinder( arm1, .75, .05, [.65, .65, .65, 1] ) 
arm2 = 'arm2'; viz.addCylinder( arm2, .75, .05, [.65, .65, .65, 1] ) 

# add target 
target = 'target'; viz.addSphere( target, .1001, [0, .8, .1, 1] ) 
pos  = [.5, .5, 0] 
quat = [1, 0, 0, 0] 
viz.applyConfiguration( target, pos + quat ) 

# ---------------------------------- 
# Given configuration vector q_config of dimension 2, compute the position of the centers of each object, and correctly display the robot 
# ---------------------------------- 

# compute random configuration vector, each in [-pi, pi] 
q_config = np.random.rand(2) * 2 * np.pi - np.pi 

# compute the position of the centers of each object 
def display(q_config):
    '''
    Display the robot in Gepetto Viewer. 
    '''
    
    assert (q_config.shape == (2,))
    
    c0 = np.cos(q_config[0])
    s0 = np.sin(q_config[0])
    c1 = np.cos(q_config[0] + q_config[1])
    s1 = np.sin(q_config[0] + q_config[1])
    
    viz.applyConfiguration( 'joint1', planar( 0,         0,         0 ) ) 
    viz.applyConfiguration( 'arm1'  , planar( c0/2,      s0/2,      q_config[0] ) ) 
    viz.applyConfiguration( 'joint2', planar( c0,        s0,        q_config[0] ) ) 
    viz.applyConfiguration( 'arm2'  , planar( c0 + c1/2, s0 + s1/2, q_config[0] + q_config[1] ) ) 
    viz.applyConfiguration( 'joint3', planar( c0 + c1,   s0 + s1,   q_config[0] + q_config[1] ) ) 

# test function 
display(q_config) 

# get end effector position 
def get_end_eff(q_config):    
    '''
    Return the 2D position of the end effector of the robot at configuration q_config. 
    '''
    
    assert (q_config.shape == (2,))
    c0 = np.cos(q_config[0])
    s0 = np.sin(q_config[0])
    c1 = np.cos(q_config[0] + q_config[1])
    s1 = np.sin(q_config[0] + q_config[1])
    
    return np.array([c0 + c1, s0 + s1])

# test function 
pos_end_eff = get_end_eff(q_config) 

# ---------------------------------- 
# Optimize the configuration 
# ---------------------------------- 

pos_target = np.array([.5, .5]) 
viz.applyConfiguration( target, translation2d( pos_target[0], pos_target[1] ) ) 

# define cost function 
def compute_cost(q_config):
    '''
    Compute the cost of the robot at configuration q_config. 
    '''
    
    pos_end_eff = get_end_eff(q_config)
    
    return norm(pos_end_eff - pos_target)**2 

# SciPy BFGS also takes callback function - use it to display the robot at each iteration  
def display_callback(q_config): 
    display(q_config) 
    time.sleep(.1) 
    
# initial guess 
q0    = np.array( [0.0, 0.0] ) 

# optimize 
# fmin_bfgs input arguments: 
#   compute_cost = function to optimize 
#   q0 = initial guess 
#   display_callback = callback function 
q_opt = fmin_bfgs( compute_cost, q0, callback = display_callback ) 

print("Optimal configuration from BFGS: ", q_opt) 


# ==================================================================== 
# FULL CONFIGURATION 
# ==================================================================== 

# q_config = [ q1, q2 ] are the joint angles of the 2 DOF robot 
# full configuration is actually pose = [ x1, y1, th1, x2, y2, th2, x3, y3, th3 ] 
# total # parameters = 9 
# as for the constraints ... 
#   th1 = q1 
#   x2  = x1 + cos(th1) 
#   y2  = y1 + sin(th1) 
#   th2 = q1 + q2 
#   x3  = x2 + cos(th2) 
#   y3  = y2 + sin(th2) 
#   th3 = th2 

def get_end_eff_9(pose): 
    ''' 
    Return the 2D position of the end effector of the robot at configuration q_config 
    '''
    
    assert (pose.shape == (9, )) 
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = pose 
    pos_end_eff = np.array([x3, y3]) 
    
    return pos_end_eff 

def display_9(pose): 
    ''' 
    Display the robot in Gepetto Viewer 
    '''
    
    assert (pose.shape == (9, )) 
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = pose 
    
    dx1 = x2 - x1 ; dy1 = y2 - y1 
    dx2 = x3 - x2 ; dy2 = y3 - y2 
    
    a1 = [ x1 + dx1 / 2, y1 + dy1 / 2 ] 
    a2 = [ x2 + dx2 / 2, y2 + dy2 / 2 ] 
    
    viz.applyConfiguration( 'joint1', planar( x1,    y1,    t1 ) ) 
    viz.applyConfiguration( 'arm1',   planar( a1[0], a1[1], t1 ) )
    viz.applyConfiguration( 'joint2', planar( x2,    y2,    t2 ) ) 
    viz.applyConfiguration( 'arm2',   planar( a2[0], a2[1], t2 ) ) 
    viz.applyConfiguration( 'joint3', planar( x3,    y3,    t3 ) ) 
    
# ---------------------------------- 
# Compute costs and constraints 
# ---------------------------------- 
    
def compute_cost_9(pose): 
    ''' 
    Compute the cost of the robot at configuration q_config 
    '''
    
    pos_end_eff = get_end_eff_9(pose) 
    
    return norm(pos_end_eff - pos_target)**2 

# show that a random configuration looks like nonsense 
q_rand_9 = np.random.rand(9) 
display_9(q_rand_9) 

# define a function to compute the constraints 
def constraint_9(pose): 
    ''' 
    Compute the constraints of the robot at configuration q_config 
    '''
    
    assert (pose.shape == (9, )) 
    x1, y1, t1, x2, y2, t2, x3, y3, t3 = pose 
    
    constraints = np.zeros(7) 
    constraints[0] = x1 - 0 
    constraints[1] = y1 - 0 
    constraints[2] = x1 + np.cos(t1) - x2 
    constraints[3] = y1 + np.sin(t1) - y2 
    constraints[4] = x2 + np.cos(t2) - x3 
    constraints[5] = y2 + np.sin(t2) - y3 
    constraints[6] = t2 - t3 
    
    return constraints 

# test function 
print(compute_cost_9(q_rand_9), constraint_9(q_rand_9))

# define callback function to display the robot at each iteration 
def display_callback_9(pose): 
    display_9(pose) 
    time.sleep(.1) 

# ---------------------------------- 
# Optimize the configuration with constraints using BFGS 
# ---------------------------------- 

# BFGS cannot be used directly to optimize over equality constraints 
# a trick is to add the constraints to the cost function with a penalty term 
def compute_penalty(pose): 
    ''' 
    cost(x) + 10 * || constraint(x) ||^2
    '''

    # cost of the robot 
    cost = compute_cost_9(pose) 
    
    # array of constraints 
    constraints = constraint_9(pose) 
    
    # penalty term of contraints 
    penalty = 10 * sum( constraints**2 ) 
    
    total_cost = cost + penalty 
    
    return total_cost 

print("Optimizing with BFGS") 
q_opt = fmin_bfgs( compute_penalty, q_rand_9, callback = display_callback_9 )

# ---------------------------------- 
# Optimize the configuration with constraints using SLSQP 
# ---------------------------------- 

print("Optimizing with SLSQP") 
q_opt = fmin_slsqp( 
    compute_cost_9, 
    q_rand_9, 
    callback = display_callback_9, 
    f_eqcons = constraint_9 
    )


# ---------------------------------- 
# KEEP SCRIPT RUNNING 
# ---------------------------------- 

print("Keep Meshcat server alive") 

while True: 
    time.sleep(1)