import sys
import numpy as np

# package used to load half-space (plane) parameters
from ruamel.yaml import YAML

# package used for visualization
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf


class FrameTraversableRegion:
    def __init__(self, frame_name, reachable_stl_path,
                 convex_hull_halfspace_path,
                 visualizer=None, b_visualize=False):
        if visualizer is not None:
            b_visualize = True

        self._frame_name = frame_name
        self._reachable_stl_path = reachable_stl_path
        self._b_visualize = b_visualize

        # Visualize convex hull (reachable region)
        if b_visualize:
            try:
                if visualizer is not None:
                    # see if using Pinocchio's MeshcatVisualizer
                    if hasattr(visualizer, 'model'):
                        print("Using Pinocchio Meshcat Visualizer for Frame Traversable Region")
                        self._vis = visualizer.viewer
                else:
                    self._vis = meshcat.Visualizer() if visualizer is None else visualizer
                    self._vis.open()
                    self._vis.wait()

                obj = g.Mesh(g.StlMeshGeometry.from_file(reachable_stl_path))
                obj.material.transparent = True
                obj.material.opacity = 0.6
                self._vis["reachable_regions"][frame_name].set_object(obj)

            except ImportError as err:
                print(
                    "Error initializing Meshcat. Make sure you have installed meshcat-python"
                )
                print(err)
                sys.exit(0)

        # Retrieve halfspace coefficients defining the convex hull
        with open(convex_hull_halfspace_path, 'r') as f:
            yml = YAML().load(f)
            plane_coeffs = []
            for i_plane in range(len(yml)):
                plane_coeffs.append(yml[i_plane])

        self._origin_pos = np.zeros((3,))    # [x,y,z] of torso

    def update_origin_pose(self, origin_pos, origin_ori_angle=0.,
                           origin_ori_direction=None):
        # if no direction is given, assume it is about the z-axis
        if origin_ori_direction is None:
            origin_ori_direction = np.array([0., 0., 1.])

        if self._b_visualize:
            self._update_reachable_viewer(origin_pos,
                          origin_ori_angle, origin_ori_direction)

        # save new origin
        self._origin_pos = origin_pos

    def _update_reachable_viewer(self, origin_pos,
                                 origin_ori_angle,
                                 origin_ori_direction):
        tf_trans = tf.translation_matrix(origin_pos)
        tf_ori = tf.rotation_matrix(origin_ori_angle, origin_ori_direction)

        # first translate, then rotate
        tf_new = tf.concatenate_matrices(tf_trans, tf_ori)
        self._vis["reachable_regions"][self._frame_name].set_transform(tf_new)




