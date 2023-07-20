import numpy as np
import fastpathplanning as fpp
import matplotlib.pyplot as plt

from itertools import product

#
# offline preprocessing
#

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

# lower bounds of the safe boxes (LF)
"""
Cube definitions:
C1: left-foot to kinematic limit in z
C2: left foot through door up to kinematic limit in x
C3: left foot step down after passing door 
"""
L_lf = np.array([
    [-0.1, -0.1, 0.0],      # prevent leg-crossing
    [-0.1,  -0.1, 0.4],     # prevent leg-crossing
    [0.2,  -0.1, 0.0]      # prevent leg-crossing
])

# upper bounds of the safe boxes
U_lf = np.array([
    [0.1, 0.3, 0.6],        # z stops at kin. limit
    [0.4, 0.3, 0.6],        # x stops at kin. limit
    [0.5, 0.3, 0.6]         # x stops at kin. limit
])

S_lf = fpp.SafeSet(L_lf, U_lf, verbose=True)

# online path planning
p_init = np.array([0.0, 0.1, 0.0])     # initial point
p_term = np.array([0.4, 0.1, 0.0])     # terminal point
T = 3                                   # traversal time
alpha = [0, 0, 1]                       # cost weights
p_lf = fpp.plan(S_lf, p_init, p_term, T, alpha)

#
# Plots
#

# Safe Regions (Boxes)
ax = plt.figure().add_subplot(projection='3d')
S_lf.plot3d(ax)                     # plot safe set
plt.xlim([np.min(L_lf[:, 0], axis=0), np.max(U_lf[:, 0], axis=0)])
plt.ylim([np.min(L_lf[:, 1], axis=0), np.max(U_lf[:, 1], axis=0)])

# Planned path
p_lf.plot3d(ax)                     # plot smooth path
plt.xlim([np.min(L_lf[:, 0], axis=0), np.max(U_lf[:, 0], axis=0)])
plt.ylim([np.min(L_lf[:, 1], axis=0), np.max(U_lf[:, 1], axis=0)])
plt.show()
