from meshcat.geometry import (MeshLambertMaterial, Sphere)

#
# Color definitions
#
RED = 0xff0000
GREEN = 0x00ff00
BLUE = 0x0000ff
GREY = 0x888888
WHITE = 0xffffff
BLACK = 0x000000
CYAN = 0x00ffff

#
# Commonly used object colors and opacities
#
def meshcat_domain_obj():
    return MeshLambertMaterial(color=BLUE, opacity=0.05)


def meshcat_obstacle_obj():
    return MeshLambertMaterial(color=GREY)


def meshcat_iris_obj():
    return MeshLambertMaterial(color=RED, opacity=0.3)


def meshcat_reach_obj():
    return MeshLambertMaterial(color=WHITE, opacity=0.2)


def meshcat_point_obj():
    return MeshLambertMaterial(color=GREEN)


def meshcat_collision_obj():
    return MeshLambertMaterial(color=BLACK)


def meshcat_safe_obj():
    return MeshLambertMaterial(color=CYAN)