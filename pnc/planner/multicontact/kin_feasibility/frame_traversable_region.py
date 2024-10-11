import sys
import numpy as np

# package used to load half-space (plane) parameters
from ruamel.yaml import YAML

# package for frame path planning
import pnc.planner.multicontact.kin_feasibility.fastpathplanning.fastpathplanning as fpp

# package used for visualization
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from util.pydrake_meshcat_interface import *
from util.polytope_math import extract_plane_eqn_from_coeffs
from visualizer.meshcat_tools.meshcat_palette import meshcat_reach_obj
# Iris Regions Manager
from vision.iris.iris_regions_manager import IrisRegionsManager


def convert_rgba_to_meshcat_obj(obj, color_rgb):
    obj.color = int(color_rgb[0] * 255) * 256 ** 2 + int(
            color_rgb[1] * 255) * 256 + int(color_rgb[2] * 255)
    obj.transparent = True
    obj.opacity = color_rgb[3]


class FrameTraversableRegion:
    def __init__(self, frame_name,
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
        convex_hull_halfspace_path : String
            Path to hyperplane parameters, e.g., of (a, b, c, d) in
                .. math::
                    ax + by + cz + d = 0
            Parameters are specified w.r.t. root (e.g., torso frame)
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
        self._b_visualize_reach = b_visualize_reach

        # Retrieve halfspace coefficients defining the convex hull
        self._plane_coeffs = []
        if convex_hull_halfspace_path is not None:
            with open(convex_hull_halfspace_path, 'r') as f:
                yml = YAML().load(f)
                for i_plane in range(len(yml)):
                    self._plane_coeffs.append(yml[i_plane])
        else:
            print(f"Convex hull not specified for {frame_name}")

        # Visualize convex hull (reachable region)
        if b_visualize_reach or b_visualize_safe:
            try:
                if visualizer is not None:
                    # see if using Pinocchio's MeshcatVisualizer
                    if hasattr(visualizer, 'model'):
                        print(f"Using Pinocchio Meshcat visualizer for {frame_name} Frame Traversable Region")
                        self._vis = visualizer.viewer
                    else:   # using bare Meshcat
                        print(f"Using Meshcat visualizer for {frame_name} Frame Traversable Region")
                        self._vis = visualizer
                else:
                    self._vis = meshcat.Visualizer()
                    self._vis.open()
                    self._vis.wait()

                if convex_hull_halfspace_path is not None:
                    # load the convex hull
                    A, b = extract_plane_eqn_from_coeffs(self._plane_coeffs)
                    polyhedron = HPolyhedron(A, -b)
                    obj = pydrake_geom_to_meshcat(polyhedron)
                    self._vis["traversable_regions"]["reachable"][frame_name].set_object(obj, meshcat_reach_obj())

            except ImportError as err:
                print(
                    "Error initializing Meshcat. Make sure you have installed meshcat-python"
                )
                print(err)
                sys.exit(0)

        # set the origin of the reachable space (i.e., origin of torso w.r.t. world)
        self._origin_pos = np.zeros((3,))               # [x,y,z] of torso w.r.t. world
        self._origin_ori_direction = np.zeros((3,))     # w.r.t. world frame
        self._origin_ori_angle = 0.                     # about ori_direction

        # initialize list for collision-free safe set (boxes) for planning
        self._plan_safe_box_list = None       # safe boxes per frame
        self._plan_ir_mgr = None       # iris regions per frame
        self._plan_T = []
        self._plan_alpha = [0, 0, 1]    # cost weights [pos, vel, acc, ...]
        self._N_boxes = 0
        self._b_visualize_safe = b_visualize_safe

    def update_origin_pose(self, origin_pos, origin_ori_angle=0.,
                           origin_ori_direction=None):
        # if no direction is given, assume it is about the z-axis
        if origin_ori_direction is None:
            origin_ori_direction = np.array([0., 0., 1.])

        tf_new = origin_pos

        if self._b_visualize_reach and len(self._plane_coeffs) != 0:
            # Note: this does NOT update the actual plane equations, it simply
            # shifts the origin of the STL part to origin_pos. The equations
            # update is currently being done in the LocomanipulationFramePlanner
            tf_new = self._update_reachable_viewer(origin_pos,
                          origin_ori_angle, origin_ori_direction)

        # save new origin
        self._origin_pos = origin_pos
        self._origin_ori_angle = origin_ori_angle
        self._origin_ori_direction = origin_ori_direction
        return tf_new

    def _update_reachable_viewer(self, origin_pos,
                                 origin_ori_angle,
                                 origin_ori_direction):
        tf_trans = tf.translation_matrix(origin_pos)
        tf_ori = tf.rotation_matrix(origin_ori_angle, origin_ori_direction)

        # first translate, then rotate
        tf_new = tf.concatenate_matrices(tf_trans, tf_ori)
        self._vis["traversable_regions"]["reachable"][self.frame_name].set_transform(tf_new)
        self._vis["traversable_regions"]["reachable"].set_property("visible", True)
        return tf_new

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

    def load_iris_regions(self, iris_region_mgr: IrisRegionsManager):
        self._plan_ir_mgr = iris_region_mgr

        # set visualization
        if self._b_visualize_safe:
            for ir in iris_region_mgr.iris_list:
                ir.visualize(self._vis["traversable_regions/safe"][self.frame_name], self._N_boxes)
                self._N_boxes += 1

    def get_visualizer(self):
        return self._vis

    def visualize_reachable_region_from_pos(self, torso_pos):
        torso_init = np.array([0., 0., 1.01])
        if len(self._plane_coeffs) == 0:
            raise ValueError(f"Convex hull path not specified for {self.frame_name}")
        else:
            # load the convex hull
            H, d = extract_plane_eqn_from_coeffs(self._plane_coeffs)

        # create random points
        p_tmp = np.random.uniform(-1.5, 2.5, (10000, 3))

        # remove points that are outside the reachable constraint
        for idx, p in enumerate(p_tmp):
            if (H @ (p - torso_pos + torso_init).T <= -d).all():
                continue
            else:
                p_tmp[idx] = np.zeros((3, ))

        # visualize the points
        obj = g.Points(g.PointsGeometry(p_tmp.T), g.PointsMaterial(size=0.01))
        self._vis["traversable_regions"]["reachable"][self.frame_name][str(1)].set_object(obj)

