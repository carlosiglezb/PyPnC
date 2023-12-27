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
    def __init__(self, frame_name, reachable_stl_path=None,
                 convex_hull_halfspace_path=None,
                 visualizer=None,
                 b_visualize_reach=False,
                 b_visualize_safe=False):
        r"""Creates a convex polyhedron composed of :
        (1) the reachable space of a frame w.r.t. root, and
        (2) the safe, collision-free region specified for the frame

        Arguments
        ---------
        frame_name : String
            Name of the frame for which TraversableRegion is made for.
        reachable_stl_path : String
            Path to the stl (convex polygon) describing the reachable space of the frame
        convex_hull_halfspace_path : String
            Path to hyperplane parameters, e.g., of (a, b, c, d) in
                .. math::
                    ax + by + cz + d = 0
            of STL in reachable_stl_path. Parameters are
            specified w.r.t. root (e.g., torso frame)
        visualizer : Meshcat.Visualizer
            Viewer window for displaying reachable and safe regions
        b_visualize_reach : Bool
            Whether we want to visualize the reachable region or not
        b_visualize_safe : Bool
            Whether we want to visualize the safe, collision-free region or not
        """
        if visualizer is not None:
            b_visualize_reach = True

        self.frame_name = frame_name
        self._reachable_stl_path = reachable_stl_path
        self._b_visualize_reach = b_visualize_reach

        # Visualize convex hull (reachable region)
        if b_visualize_reach or b_visualize_safe:
            try:
                if visualizer is not None:
                    # see if using Pinocchio's MeshcatVisualizer
                    if hasattr(visualizer, 'model'):
                        print("Using existing visualizer for Frame Traversable Region")
                        self._vis = visualizer.viewer
                else:
                    self._vis = meshcat.Visualizer()
                    self._vis.open()
                    self._vis.wait()

                if reachable_stl_path is not None:
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
        if convex_hull_halfspace_path is not None:
            with open(convex_hull_halfspace_path, 'r') as f:
                yml = YAML().load(f)
                self._plane_coeffs = []
                for i_plane in range(len(yml)):
                    self._plane_coeffs.append(yml[i_plane])
        else:
            print(f"Convex hull not specified for {frame_name}")

        # set the origin of the reachable space (i.e., origin of torso w.r.t. world)
        self._origin_pos = np.zeros((3,))               # [x,y,z] of torso w.r.t. world
        self._origin_ori_direction = np.zeros((3,))     # w.r.t. world frame
        self._origin_ori_angle = 0.                     # about ori_direction

        # initialize list for collision-free safe set (boxes) for planning
        self._plan_safe_box_list = None       # safe boxes per frame
        self._plan_T = []
        self._plan_alpha = [0, 0, 1]    # cost weights [pos, vel, acc, ...]
        self._N_boxes = 0
        self._b_visualize_safe = b_visualize_safe

    def update_origin_pose(self, origin_pos, origin_ori_angle=0.,
                           origin_ori_direction=None):
        # if no direction is given, assume it is about the z-axis
        if origin_ori_direction is None:
            origin_ori_direction = np.array([0., 0., 1.])

        if self._b_visualize_reach and self._reachable_stl_path is not None:
            # Note: this does NOT update the actual plane equations, it simply
            # shifts the origin of the STL part to origin_pos. The equations
            # update is currently being done in the LocomanipulationFramePlanner
            self._update_reachable_viewer(origin_pos,
                          origin_ori_angle, origin_ori_direction)

        # save new origin
        self._origin_pos = origin_pos
        self._origin_ori_angle = origin_ori_angle
        self._origin_ori_direction = origin_ori_direction

    def _update_reachable_viewer(self, origin_pos,
                                 origin_ori_angle,
                                 origin_ori_direction):
        tf_trans = tf.translation_matrix(origin_pos)
        tf_ori = tf.rotation_matrix(origin_ori_angle, origin_ori_direction)

        # first translate, then rotate
        tf_new = tf.concatenate_matrices(tf_trans, tf_ori)
        self._vis["traversable_regions"]["reachable"][self.frame_name].set_transform(tf_new)

    def load_collision_free_boxes(self, box_llim, box_ulim):
        self._plan_safe_box_list = fpp.SafeSet(
            box_llim, box_ulim, verbose=False)

        # set visualization
        if self._b_visualize_safe:
            for box_ll, box_ul in zip(box_llim, box_ulim):
                box_dims = box_ul - box_ll
                box = g.Box(box_dims)
                box_color_rgba = [0.7, 0., 0., 0.1]

                # convert RGBA to meshcat convention and visualize
                obj = g.MeshPhongMaterial()
                convert_rgba_to_meshcat_obj(obj, box_color_rgba)
                self._vis["traversable_regions"]["safe"][self.frame_name][str(self._N_boxes)].set_object(box, obj)
                box_center = tf.translation_matrix((box_ul + box_ll) / 2.)
                self._vis["traversable_regions"]["safe"][self.frame_name][str(self._N_boxes)].set_transform(box_center)
                self._N_boxes += 1

    def get_visualizer(self):
        return self._vis


