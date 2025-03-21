import os
import sys
import crocoddyl
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

cwd = os.getcwd()
sys.path.append(cwd)

import plot.meshcat_utils as vis_tools
import util.util
from crab_fns import * 

# ---------------------------------- 
# Create action model for torso orientation 
# ---------------------------------- 

def createNoSupportTorsoActionModel(base_target=None):
    
    # Define the cost sum (cost manager)
    costs = crocoddyl.CostModelSum(state, actuation.nu)

    # Adding the hand-placement cost
    if base_target is not None:
        w_base = np.array([1.] * 3 + [1.] * 3)        # (lin, ang)
        base_Mref = pin.SE3(base_target, np.zeros(3))
        activation_base = crocoddyl.ActivationModelWeightedQuad(w_base**2) 
        base_cost = crocoddyl.CostModelResidual(
            state,
            activation_base,
            crocoddyl.ResidualModelFramePlacement(state, base_id, base_Mref, actuation.nu),
        )
        costs.addCost("base_goal", base_cost, 1e2)

    # Adding state and control regularization terms
    w_x = np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [10] * state.nv)
    activation_xreg = crocoddyl.ActivationModelWeightedQuad(w_x**2)
    x_reg_cost = crocoddyl.CostModelResidual(
        state, activation_xreg, crocoddyl.ResidualModelState(state, x0, actuation.nu)
    )
    u_reg_cost = crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, actuation.nu)
    )
    costs.addCost("xReg", x_reg_cost, 1e-3)
    costs.addCost("uReg", u_reg_cost, 1e-4)

    # Adding the state limits penalization
    x_lb = np.concatenate([state.lb[1 : state.nv + 1], state.lb[-state.nv :]])
    x_ub = np.concatenate([state.ub[1 : state.nv + 1], state.ub[-state.nv :]])
    activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(x_lb, x_ub)
    )
    x_bounds = crocoddyl.CostModelResidual(
        state,
        activation_xbounds,
        crocoddyl.ResidualModelState(state, 0 * x0, actuation.nu),
    )
    costs.addCost("xBounds", x_bounds, 100.0)

    contacts = crocoddyl.ContactModelMultiple(state, actuation.nu)
    # contact_data = contacts.createData(rob_data)
    # Creating the action model
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contacts, costs
    )
    return dmodel

# ---------------------------------- 
# Create sequence of action models 
# ---------------------------------- 

def createSequence(dmodels, DT, N):
    return [
        [crocoddyl.IntegratedActionModelEuler(m, DT)] * N
        + [crocoddyl.IntegratedActionModelEuler(m, 0.0)]
        for m in dmodels
    ]


## ============================================ ##
## Main simulation 
## ============================================ ##

crab_urdf_file = cwd + "/robot_model/crab/crab.urdf"
package_dir    = cwd + "/robot_model/crab"
rob_model, col_model, vis_model = pin.buildModelsFromUrdf(
    crab_urdf_file,
    package_dir, 
    pin.JointModelFreeFlyer() )

# ---------------------------------- 
# Initial configuration 
# ---------------------------------- 

# initial configuration  
q0 = get_initial_pose()
v0 = np.zeros(rob_model.nv)
x0 = np.concatenate([q0, v0])

# Getting the frame ids
base_id = rob_model.getFrameId("base_link")

# remove gravity
rob_model.gravity = pin.Motion.Zero()

# create data  
rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model) 

# ---------------------------------- 
# create target orientation !!!!! 
# ---------------------------------- 

# base_targets = util.util.euler_to_rot([0., np.pi/2, 0.])
base_targets = util.util.euler_to_rot([np.pi/2, 0., 0.])

# Define the robot's state and actuation
state     = crocoddyl.StateMultibody(rob_model)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# Update Pinocchio model
pin.forwardKinematics(rob_model, rob_data, q0)
pin.updateFramePlacements(rob_model, rob_data)

# ---------------------------------- 
# Solve the problem 
# ---------------------------------- 

# time step 
DT = 2e-2 

# initialize solver 
fddp = [None]

# knots to re-orient torso
N_base_orientation = 300 

# create action model 
dmodel = createNoSupportTorsoActionModel(base_target = base_targets)

# create sequence of action models 
model_seqs = createSequence([dmodel], DT, N_base_orientation)

# create problem 
problem = crocoddyl.ShootingProblem(x0, sum(model_seqs, [])[:-1], model_seqs[-1][-1])

# set up solver 
fddp[0] = crocoddyl.SolverFDDP(problem)

# Adding callbacks to inspect the evolution of the solver (logs are printed in the terminal)
fddp[0].setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

# Solver settings
max_iter = 150
fddp[0].th_stop = 1e-3

# Set initial guess
xs = [x0] * (fddp[0].problem.T + 1)
us = fddp[0].problem.quasiStatic([x0] * fddp[0].problem.T)
print("Problem solved:", fddp[0].solve(xs, us, max_iter))
print("Number of iterations:", fddp[0].iter)

# ---------------------------------- 
# Display results in meschat 
# ---------------------------------- 

save_freq = 1
display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                  rob_data, vis_data, col_data, ctrl_freq=1/DT, save_freq=save_freq)
display.displayFromCrocoddylSolver(fddp)

print("Done") 

# ---------------------------------- 
# Plot joint angles 
# ---------------------------------- 

# Get the solution trajectory
xs = fddp[0].xs

# Get joint angles over time (excluding the floating base)
# The first 7 values in q are for the floating base (3 for position, 4 for quaternion)
n_base_dof = 7  # Floating base DOFs 
joint_angles = np.array([x[:rob_model.nq][n_base_dof:] for x in xs])

# Create time array
time_array = np.arange(0, len(xs)) * DT

# Get joint limits from the robot model
lower_limits = rob_model.lowerPositionLimit[n_base_dof:]
upper_limits = rob_model.upperPositionLimit[n_base_dof:]

# Create a dictionary mapping joint names to their trajectories
joint_names = []
for i in range(1, rob_model.njoints):  # Skip the first joint (universe/root)
    if i > 1:  # Skip the floating base joint
        joint_names.append(rob_model.names[i])

joint_trajectories = {}
joint_lower_limits = {}
joint_upper_limits = {}

for i, name in enumerate(joint_names):
    # Make sure we don't go out of bounds
    if i < joint_angles.shape[1]:
        joint_trajectories[name] = joint_angles[:, i]
        joint_lower_limits[name] = lower_limits[i]
        joint_upper_limits[name] = upper_limits[i]

# Create a 7x4 grid of subplots
fig, axes = plt.subplots(7, 4, figsize=(16, 20))

# Counter for subplot position
row, col = 0, 0
max_rows, max_cols = 7, 4

# Plot each trajectory in its own subplot
for name, trajectory in joint_trajectories.items():
    
    # Plot the trajectory in the current subplot
    axes[row, col].plot(time_array, trajectory)
    
    # Add joint limits as red dashed lines
    if name in joint_lower_limits and name in joint_upper_limits:
        lower_limit = joint_lower_limits[name]
        upper_limit = joint_upper_limits[name]
        
        # Only plot limits that are not at infinity
        # if lower_limit > -1e10:  # Avoid plotting very large negative values
        axes[row, col].axhline(lower_limit, color='r', linestyle='--', label='Lower limit')
        
        # if upper_limit < 1e10:  # Avoid plotting very large positive values
        axes[row, col].axhline(upper_limit, color='r', linestyle='--', label='Upper limit')
    
    axes[row, col].set_title(name, fontsize=10)
    axes[row, col].grid(True)
    
    # Only add x-label to bottom row
    if row == max_rows - 1:
        axes[row, col].set_xlabel('Time (s)')
    
    # Only add y-label to leftmost column
    if col == 0:
        axes[row, col].set_ylabel('Angle (rad)')
    
    # Move to the next subplot position
    row += 1
    if row >= max_rows:
        row = 0
        col += 1
        if col >= max_cols:
            # We've filled all subplots
            break

# Add a single legend for the entire figure
handles, labels = axes[0, 0].get_legend_handles_labels()
if handles:  # Only add legend if we have any labelss
    fig.legend(handles, labels, loc='upper right')

fig.suptitle('Joint Angle Trajectories', fontsize=16)

# Increase spacing between subplots to prevent overlap
plt.subplots_adjust(hspace=0.5, wspace=0.3, left=0.07, right=0.95, top=0.93, bottom=0.05)
plt.show()

# ---------------------------------- 
# Plot joint torques 
# ---------------------------------- 

# Get the solution control inputs (torques)
us = fddp[0].us

# Create a dictionary mapping joint names to their torque trajectories
joint_torques = {}
for i, name in enumerate(joint_names):
    if i < len(us[0]):  # Make sure we don't go out of bounds
        # Extract torque trajectory for this joint
        joint_torques[name] = np.array([u[i] for u in us])

# Create a 7x4 grid of subplots for torques
fig_torque, axes_torque = plt.subplots(7, 4, figsize=(16, 20))

# Counter for subplot position
row, col = 0, 0
max_rows, max_cols = 7, 4

# Plot each torque trajectory in its own subplot
for name, trajectory in joint_torques.items():
    
    # Plot the torque trajectory in the current subplot
    axes_torque[row, col].plot(time_array[:-1], trajectory)  # Note: us is one shorter than xs
    
    # We'll skip the effort limits for now to avoid errors
    
    axes_torque[row, col].set_title(name, fontsize=10)
    axes_torque[row, col].grid(True)
    
    # Only add x-label to bottom row
    if row == max_rows - 1:
        axes_torque[row, col].set_xlabel('Time (s)')
    
    # Only add y-label to leftmost column
    if col == 0:
        axes_torque[row, col].set_ylabel('Torque (N⋅m)')
    
    # Move to the next subplot position
    row += 1
    if row >= max_rows:
        row = 0
        col += 1
        if col >= max_cols:
            # We've filled all subplots
            break

fig_torque.suptitle('Joint Torques', fontsize=16)

# Increase spacing between subplots to prevent overlap
plt.subplots_adjust(hspace=0.5, wspace=0.3, left=0.07, right=0.95, top=0.93, bottom=0.05)
plt.show()

# ==================================================================== 
# KEEP SCRIPT RUNNING 
# ==================================================================== 

print("Keep Meshcat server alive") 

while True: 
    time.sleep(1)

