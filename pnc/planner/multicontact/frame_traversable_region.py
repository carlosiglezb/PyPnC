import sys
import numpy as np

# package used to load half-space (plane) parameters
from ruamel.yaml import YAML

# package for frame path planning
import pnc.planner.multicontact.fastpathplanning.fastpathplanning as fpp

# package used for visualization
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf


def convert_rgba_to_meshcat_obj(obj, color_rgb):
    obj.color = int(color_rgb[0] * 255) * 256 ** 2 + int(
            color_rgb[1] * 255) * 256 + int(color_rgb[2] * 255)
    obj.transparent = True
    obj.opacity = color_rgb[3]


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
                convert_rgba_to_meshcat_obj(obj.material, [0.9, 0.9, 0.9, 0.2])
                self._vis["traversable_regions"]["reachable"][frame_name].set_object(obj)

            except ImportError as err:
                print(
                    "Error initializing Meshcat. Make sure you have installed meshcat-python"
                )
                print(err)
                sys.exit(0)

        # Retrieve halfspace coefficients defining the convex hull
        with open(convex_hull_halfspace_path, 'r') as f:
            yml = YAML().load(f)
            self._plane_coeffs = []
            for i_plane in range(len(yml)):
                self._plane_coeffs.append(yml[i_plane])

        self._origin_pos = np.zeros((3,))    # [x,y,z] of torso

        # initialize list for collision-free safe set (boxes) for planning
        self._plan_safe_box_list = []       # safe boxes per frame
        self._plan_T = []
        self._plan_alpha = [0, 0, 1]    # cost weights [pos, vel, acc, ...]
        self._N_boxes = 0

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
        self._vis["traversable_regions"]["reachable"][self._frame_name].set_transform(tf_new)

    def load_collision_free_boxes(self, box_llim, box_ulim, b_visualize=True):
        self._plan_safe_box_list.append(fpp.SafeSet(
            box_llim, box_ulim, verbose=True))

        # set visualization
        if b_visualize:
            for box_ll, box_ul in zip(box_llim, box_ulim):
                box_dims = box_ul - box_ll
                box = g.Box(box_dims)
                box_color_rgba = [0.7, 0., 0., 0.1]

                # convert RGBA to meshcat convention and visualize
                obj = g.MeshPhongMaterial()
                convert_rgba_to_meshcat_obj(obj, box_color_rgba)
                self._vis["traversable_regions"]["safe"][self._frame_name][str(self._N_boxes)].set_object(box, obj)
                box_center = tf.translation_matrix((box_ul + box_ll) / 2.)
                self._vis["traversable_regions"]["safe"][self._frame_name][str(self._N_boxes)].set_transform(box_center)
                self._N_boxes += 1




