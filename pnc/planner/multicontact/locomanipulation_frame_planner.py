import pnc.planner.multicontact.fastpathplanning.fastpathplanning as fpp
from collections import OrderedDict
from pnc.planner.multicontact.frame_traversable_region import convert_rgba_to_meshcat_obj

import meshcat.geometry as g
import meshcat.transformations as tf

class LocomanipulationFramePlanner:
    def __init__(self, traversable_regions_list):

        self.safe_boxes = OrderedDict()
        self.reachability_planes = OrderedDict()
        self.path = []
        self.frame_names = []
        for region in traversable_regions_list:
            self.frame_names.append(region._frame_name)
            self.safe_boxes[region._frame_name] = region._plan_safe_box_list
            if region._frame_name != 'torso':
                self.reachability_planes[region._frame_name] = region._plane_coeffs

    def plan(self, p_init, p_term, T, alpha, der_init={}, der_term={}, verbose=True):
        S = self.safe_boxes
        R = self.reachability_planes
        self.path = fpp.plan_multiple(S, R, p_init, p_term, T, alpha, der_init, der_term, verbose)

    def plot(self, visualizer):

        for frame in self.frame_names:
            bezier_curve = self.path[frame]
            self.visualize_bezier_points(visualizer, frame, bezier_curve)

    @staticmethod
    def visualize_bezier_points(visualizer, frame, bezier_curve):
        color_waypoint = [0., 1., 0., 0.6]      # blue
        color_transition = [1., 1., 0., 0.6]    # yellow
        r_bezier_pts = 0.01
        pt_number, seg_number = 0, 1
        for bez in bezier_curve.beziers:
            t, points = bez.get_sample_points()
            for p in points:
                obj = g.Sphere(r_bezier_pts)
                tf_pos = tf.translation_matrix(p)

                # check if "in-between" waypoint or "transition" waypoint
                if (pt_number == seg_number*(len(points))-1) or (pt_number == (seg_number-1) * (len(points))):
                    convert_rgba_to_meshcat_obj(obj, color_transition)
                else:
                    convert_rgba_to_meshcat_obj(obj, color_waypoint)
                visualizer.viewer["traversable_regions"]["path"][frame][str(pt_number)].set_object(
                    obj, g.MeshLambertMaterial(color=obj.color))
                visualizer.viewer["traversable_regions"]["path"][frame][str(pt_number)].set_transform(tf_pos)

                pt_number += 1
            seg_number += 1
