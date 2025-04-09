"""This is an extension of the original Fast Path Planning algorithm
described in:

"Marcucci, T., Nobel, P., Tedrake, R. and Boyd, S., 2024.
Fast path planning through large collections of safe boxes.
IEEE Transactions on Robotics."

where we consider multiple points (e.g., the origin of multiple robot frames)
constrained by each other's paths, polytopoic Safe regions created from
IRIS regions, and the inclusion of safe collision avoidance constraints
through rigid body primitives.

Details of this extension can be found in the paper:

"C. Gonzalez and L. Sentis, "Guiding Collision-Free Humanoid Multi-Contact
Locomotion using Convex Kinematic Relaxations and Dynamic Optimization,"
2024 IEEE-RAS 23rd International Conference on Humanoid Robots (Humanoids),
Nancy, France, 2024, pp. 592-599, doi: 10.1109/Humanoids58906.2024.10769791."
"""

from .mfpp_polygonal import *
from .mfpp_smooth import *
from .multiframe_fpp import *
