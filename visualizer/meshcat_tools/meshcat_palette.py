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
PURPLE = 0x800080

#
# Commonly used object colors and opacities
#
def meshcat_domain_obj(opacity=0.05):
    return MeshLambertMaterial(color=BLUE, opacity=opacity)


def meshcat_obstacle_obj(color=GREY, opacity=1.0):
    return MeshLambertMaterial(color=color, opacity=opacity)


def meshcat_iris_obj(opacity=0.3):
    return MeshLambertMaterial(color=RED, opacity=opacity)


def meshcat_reach_obj(opacity=0.2):
    return MeshLambertMaterial(color=WHITE, opacity=opacity)


def meshcat_point_obj(opacity=1.0):
    return MeshLambertMaterial(color=GREEN, opacity=opacity)


def meshcat_collision_obj():
    return MeshLambertMaterial(color=BLACK)


def meshcat_safe_obj():
    return MeshLambertMaterial(color=CYAN)