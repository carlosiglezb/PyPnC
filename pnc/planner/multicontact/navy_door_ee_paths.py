r"""
Note: we assume the bounds are given from a vision processing algorithm (e.g., using SDP)
The navy door is located such that all points:
    .. math::
    
  x \in [0.10, 0.20] 
  y \in [-0.4, -0.2] AND \in [0.2, 0.4], if z \in [0.4, 1.4],
  y \in [-0.4, 0.4] if z \in [0.0, 0.4] OR z \in [1.4, 2.0]
are inside the door (i.e., not in the safe region).
"""

r"""
Note: we assume the kinematic constraint has been specified/learned as convex polytopes
"""
import numpy as np
import pnc.planner.multicontact.fastpathplanning.fastpathplanning as fpp
import matplotlib.pyplot as plt

##############################
# offline preprocessing
##############################

# lower bounds of the safe boxes (LF)
"""
Cube definitions:
C1: left-foot to kinematic limit in z
C2: left foot through door up to kinematic limit in x
C3: left foot step down after passing door 
"""
L_lf = np.array([
    [-0.1, -0.1, 0.0],      # prevent leg-crossing
    [-0.1, -0.1, 0.4],     # prevent leg-crossing
    [ 0.2, -0.1, 0.0]      # prevent leg-crossing
])
L_rf = np.array([
    [-0.1, -0.3, 0.0],      # prevent leg-crossing
    [-0.1, -0.3, 0.4],     # prevent leg-crossing
    [ 0.2, -0.3, 0.0]      # prevent leg-crossing
])
L_lh = np.array([
    [-0.1, -0.1, 0.7],      # prevent leg-crossing
    [-0.1, 0.0, 0.7],     # prevent leg-crossing
    [0.2,  -0.1, 0.7]      # prevent leg-crossing
])
L_rh = np.array([
    [-0.1, -0.3, 0.7],      # prevent leg-crossing
    [-0.1, -0.3, 0.7],     # prevent leg-crossing
    [ 0.2, -0.3, 0.7]      # prevent leg-crossing
])

# upper bounds of the safe boxes
U_lf = np.array([
    [0.1, 0.3, 0.6],        # z stops at kin. limit
    [0.4, 0.3, 0.6],        # x stops at kin. limit
    [0.5, 0.3, 0.6]         # x stops at kin. limit
])
U_rf = np.array([
    [0.1, 0.1, 0.6],      # prevent leg-crossing
    [0.4, 0.1, 0.6],     # prevent leg-crossing
    [0.5, 0.1, 0.6]      # prevent leg-crossing
])
U_lh = np.array([
    [0.1, 0.3, 1.3],      # prevent leg-crossing
    [0.4, 0.3, 1.3],     # prevent leg-crossing
    [0.5, 0.3, 1.3]      # prevent leg-crossing
])
U_rh = np.array([
    [0.1, 0.1, 1.3],      # prevent leg-crossing
    [0.4, 0.0, 1.3],     # prevent leg-crossing
    [0.5, 0.1, 1.3]      # prevent leg-crossing
])

S_lf = fpp.SafeSet(L_lf, U_lf, verbose=True)
S_rf = fpp.SafeSet(L_rf, U_rf, verbose=True)
S_lh = fpp.SafeSet(L_lh, U_lh, verbose=True)
S_rh = fpp.SafeSet(L_rh, U_rh, verbose=True)

##############################
# online path planning
##############################
T = 3                                   # traversal time
alpha = [0, 0, 1]                       # cost weights

# Left Foot
p_lf_init = np.array([0.0, 0.1, 0.0])     # initial point
p_lf_term = np.array([0.4, 0.1, 0.0])     # terminal point
p_lf = fpp.plan(S_lf, p_lf_init, p_lf_term, T, alpha)

# Right Foot
p_rf_init = np.array([0.0, -0.1, 0.0])     # initial point
p_rf_term = np.array([0.4, -0.1, 0.0])     # terminal point
p_rf = fpp.plan(S_rf, p_rf_init, p_rf_term, T, alpha)

# Left Hand
p_lh_init = np.array([0.0, 0.2, 0.9])     # initial point
p_lh_term = np.array([0.4, 0.2, 0.9])     # terminal point
p_lh = fpp.plan(S_lh, p_lh_init, p_lh_term, T, alpha)

# Right Hand
p_rh_init = np.array([0.0, -0.2, 0.9])     # initial point
p_rh_term = np.array([0.4, -0.2, 0.9])     # terminal point
p_rh = fpp.plan(S_rh, p_rh_init, p_rh_term, T, alpha)

#
# Plots
#

# Safe Regions (Boxes)
ax = plt.figure().add_subplot(projection='3d')
S_lf.plot3d(ax)                     # plot safe set
S_rf.plot3d(ax)                     # plot safe set
S_lh.plot3d(ax)                     # plot safe set
S_rh.plot3d(ax)                     # plot safe set

# Planned path
p_lf.plot3d(ax)                     # plot smooth path
p_rf.plot3d(ax)                     # plot smooth path
p_lh.plot3d(ax)                     # plot smooth path
p_rh.plot3d(ax)                     # plot smooth path
plt.xlim([np.min(L_lf[:, 0], axis=0), np.max(U_lf[:, 0], axis=0)])
plt.ylim([np.min(L_rf[:, 1], axis=0), np.max(U_lf[:, 1], axis=0)])
plt.show()
