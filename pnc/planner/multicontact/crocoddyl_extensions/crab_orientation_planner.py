import os
import sys
import crocoddyl
import numpy as np
import pinocchio as pin

cwd = os.getcwd()
sys.path.append(cwd)

import plot.meshcat_utils as vis_tools
import util.util

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

def createSequence(dmodels, DT, N):
    return [
        [crocoddyl.IntegratedActionModelEuler(m, DT)] * N
        + [crocoddyl.IntegratedActionModelEuler(m, 0.0)]
        for m in dmodels
    ]


def get_default_initial_pose():
    q0 = np.zeros(35,)
    hip_yaw_angle = 5
    q0[0] = 0.                          # l_hip_ie
    q0[1] = np.radians(hip_yaw_angle)   # l_hip_aa
    q0[2] = -np.pi / 4                  # l_hip_fe
    q0[3] = np.pi / 4                   # l_knee_fe_jp
    q0[4] = np.pi / 4                   # l_knee_fe_jd
    q0[5] = -np.pi / 4                  # l_ankle_fe
    q0[6] = np.radians(-hip_yaw_angle)  # l_ankle_ie
    q0[7] = 0.                          # l_shoulder_fe
    q0[8] = np.pi / 6                   # l_shoulder_aa
    q0[9] = 0.                          # l_shoulder_ie
    q0[10] = -np.pi / 2                 # l_elbow_fe
    q0[11] = 0.                         # l_wrist_ps
    q0[12] = 0.                         # l_wrist_pitch
    q0[13] = 0.                         # left_ezgripper_knuckle_palm_L1_1
    q0[14] = 0.                         # left_ezgripper_knuckle_L1_L2_1
    q0[15] = 0.                         # left_ezgripper_knuckle_palm_L1_2
    q0[16] = 0.                         # left_ezgripper_knuckle_L1_L2_2
    q0[17] = 0.                         # neck pitch
    q0[18] = 0.                         # r_hip_ie
    q0[19] = np.radians(-hip_yaw_angle) # r_hip_aa
    q0[20] = -np.pi / 4                 # r_hip_fe
    q0[21] = np.pi / 4                  # r_knee_fe_jp
    q0[22] = np.pi / 4                  # r_knee_fe_jd
    q0[23] = -np.pi / 4                 # r_ankle_fe
    q0[24] = np.radians(hip_yaw_angle)  # r_ankle_ie
    q0[25] = 0.                         # r_shoulder_fe
    q0[26] = -0.9                       # r_shoulder_aa
    q0[27] = 0.                         # r_shoulder_ie
    q0[28] = -np.pi / 2                 # r_elbow_fe
    q0[29] = 0.                         # r_wrist_ps
    q0[30] = 0.                         # r_wrist_pitch
    q0[31] = 0.                         # right_ezgripper_knuckle_palm_L1_1
    q0[32] = 0.                         # right_ezgripper_knuckle_L1_L2_1
    q0[33] = 0.                         # right_ezgripper_knuckle_palm_L1_2
    q0[34] = 0.                         # right_ezgripper_knuckle_L1_L2_2

    # floating_base = np.array([0., 0., 0., 0., 0., 0., 1.])
    # return np.concatenate((floating_base, q0))
    return q0 


# Load robot
crab_urdf_file = cwd + "/robot_model/crab/crab.urdf"
package_dir = cwd + "/robot_model/crab"
rob_model, col_model, vis_model = pin.buildModelsFromUrdf(crab_urdf_file,
                                                          package_dir, pin.JointModelFreeFlyer())
# remove gravity
rob_model.gravity = pin.Motion.Zero()

rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)

q0 = get_default_initial_pose()
v0 = np.zeros(rob_model.nv)
x0 = np.concatenate([q0, v0])

# Getting the frame ids
base_id = rob_model.getFrameId("base_link")

# Define the robot's state and actuation
state = crocoddyl.StateMultibody(rob_model)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# Update Pinocchio model
pin.forwardKinematics(rob_model, rob_data, q0)
pin.updateFramePlacements(rob_model, rob_data)

# create target orientation
base_targets = util.util.euler_to_rot([0., np.pi/2, 0.])

#
# Solve the problem
#
DT = 2e-2
fddp = [None]

N_base_orientation = 100  # knots to re-orient torso
dmodel = createNoSupportTorsoActionModel(base_target=base_targets)
model_seqs = createSequence([dmodel], DT, N_base_orientation)
problem = crocoddyl.ShootingProblem(x0, sum(model_seqs, [])[:-1], model_seqs[-1][-1])
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

# Creating display
save_freq = 1
display = vis_tools.MeshcatPinocchioAnimation(rob_model, col_model, vis_model,
                  rob_data, vis_data, col_data, ctrl_freq=1/DT, save_freq=save_freq)
display.displayFromCrocoddylSolver(fddp)

print("Done") 

