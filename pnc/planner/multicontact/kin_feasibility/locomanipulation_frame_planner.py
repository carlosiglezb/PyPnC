from typing import List

import pnc.planner.multicontact.fastpathplanning.fastpathplanning as fpp
from collections import OrderedDict
from pnc.planner.multicontact.kin_feasibility.frame_traversable_region import convert_rgba_to_meshcat_obj
from util.polytope_math import extract_plane_eqn_from_coeffs

import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from ruamel.yaml import YAML


class LocomanipulationFramePlanner:
    def __init__(self, traversable_regions_list,
                 starting_stance_foot='LF',
                 aux_frames_path=None,
                 fixed_frames=None,
                 motion_frames_seq=None):

        # fixed, motion, and free frames filled out in the creation of hyperplanes
        self.fixed_frames, self.motion_frames_seq, self.free_frames = [], [], []

        self.safe_boxes = OrderedDict()
        self.reachability_planes = OrderedDict()
        self.path = []
        self.points = None      # control points (from Bezier trajectory solution)
        self.box_seq = []
        self.frame_names = []
        self.starting_stance_foot = starting_stance_foot
        for region in traversable_regions_list:
            self.frame_names.append(region.frame_name)
            # self.safe_boxes[region.frame_name] = region._plan_safe_box_list
            self.safe_boxes[region.frame_name] = region._plan_ir_mgr

            # do the torso at the very end to get reachability from contact frames
            if region.frame_name != 'torso':
                H, d_prime = extract_plane_eqn_from_coeffs(region._plane_coeffs)
                # d_prime = self.update_plane_offset_from_root(region._origin_pos,
                #                                                 H, d_prime)
                self.reachability_planes[region.frame_name] = {'H': H, 'd': d_prime}

            if region.frame_name == starting_stance_foot:
                init_standing_pos = region._origin_pos

            # # save if fixed, motion, or free frame at initial contact sequence
            # if fixed_frames is not None:
            #     if region.frame_name in fixed_frames[0]:
            #         self.fixed_frames.append(region.frame_name)
            #     elif region.frame_name in motion_frames[0]:
            #         self.motion_frames.append(region.frame_name)
            #     else:
            #         self.free_frames.append(region.frame_name)

        # check fixed and motion frames do not conflict
        self.fixed_frames = fixed_frames
        self.motion_frames_seq = motion_frames_seq

        # the torso must be reachable based on the frame in contact
        # H, d_prime = self.add_offset_to_plane_eqn_from_file(starting_stance_foot, ee_offset_file_path,
        #                                                     init_standing_pos)
        self.reachability_planes['torso'] = {'H': H, 'd': d_prime}

        # auxiliary frames associated to one of the above initialized reachable regions
        if aux_frames_path is not None:
            self.aux_frames = self.add_fixed_distance_between_points(aux_frames_path)
        else:
            self.aux_frames = aux_frames_path


    def add_offset_to_plane_eqn_from_file(self, frame_name,
                                          ee_offset_file_path,
                                          init_standing_pos):
        H = self.reachability_planes[frame_name]['H']
        d_vec = self.reachability_planes[frame_name]['d']
        torso_p_contact_offset = np.zeros((3,))
        if ee_offset_file_path is not None:
            with open(ee_offset_file_path, 'r') as f:
                yml = YAML().load(f)
                torso_p_contact_offset[0] = float(yml[frame_name]['x'])
                torso_p_contact_offset[1] = float(yml[frame_name]['y'])
                torso_p_contact_offset[2] = float(yml[frame_name]['z'])
        d_prime = d_vec + H @ (init_standing_pos + torso_p_contact_offset)  # grab all the 'd' coefficients
        return H, d_prime

    # def add_reachable_frame_constraint(self, frame_name, associated_traversable_region):
    #     new_frame_constr = {'name': frame_name,
    #                         'constrained_to': associated_traversable_region}
    #     self.aux_frames.append(new_frame_constr)

    def plan(self, p_init: dict[str, np.array],
             p_term: np.array,
             T: np.float64,
             alpha: List[np.float64],
             verbose: bool=True):
        S = self.safe_boxes
        R = self.reachability_planes
        A = self.aux_frames
        fixed_frames = self.fixed_frames
        motion_frames_seq = self.motion_frames_seq
        self.path, self.box_seq = fpp.plan_multiple(S, R, p_init, T, alpha, verbose, A,
                                                    fixed_frames, motion_frames_seq)

    def plan_iris(self, p_init: dict[str, np.array],
             T: np.float64,
             alpha: List[np.float64],
             verbose: bool=False):
        S = self.safe_boxes     # dict of IrisRegionsManager
        R = self.reachability_planes
        A = self.aux_frames
        fixed_frames = self.fixed_frames
        motion_frames_seq = self.motion_frames_seq
        self.path, self.box_seq, self.points = fpp.plan_multiple_iris(S, R, p_init, T, alpha, verbose, A,
                                                    fixed_frames, motion_frames_seq)

    def plot(self, visualizer, static_html=False):
        i = 0
        for p in self.path:
            for seg in range(len(p.beziers)):
                bezier_curve = [p.beziers[seg]]
                fr_name = self.frame_names[i]
                self.visualize_bezier_points(visualizer.viewer, fr_name, bezier_curve, seg)
            i += 1

        visualizer.viewer["traversable_regions"].set_property("visible", False)
        if static_html:
            # create and save locally in static html form
            res = visualizer.viewer.static_html()
            save_file = './data/multi-contact-plan.html'
            with open(save_file, "w") as f:
                f.write(res)

        tot = [len(s['LF']) for s in self.box_seq]
        num_tot_seq = sum(tot)
        # for i, p in self.points.items():
        #     if i < num_tot_seq:
        #         self.visualize_simple_points(visualizer.viewer, f'torso/knots/{str(i)}', p[0].value, color=[0., 0., 0., 1.0])
        #     elif i < 2 * num_tot_seq:
        #         self.visualize_simple_points(visualizer.viewer, f'LF/knots/{str(i)}', p[0].value, color=[0., 0., 0., 1.0])
        #     elif i < 3 * num_tot_seq:
        #         self.visualize_simple_points(visualizer.viewer, f'RF/knots/{str(i)}', p[0].value, color=[0., 0., 0., 1.0])
        #     elif i < 4 * num_tot_seq:
        #         self.visualize_simple_points(visualizer.viewer, f'LH/knots/{str(i)}', p[0].value, color=[0., 0., 0., 1.0])
        #     else:
        #         self.visualize_simple_points(visualizer.viewer, f'RH/knots/{str(i)}', p[0].value, color=[0., 0., 0., 1.0])
        # visualizer.viewer["paths/torso/knots"].set_property("visible", False)
        # visualizer.viewer["paths/LF/knots"].set_property("visible", False)
        # visualizer.viewer["paths/RF/knots"].set_property("visible", False)
        # visualizer.viewer["paths/LH/knots"].set_property("visible", False)
        # visualizer.viewer["paths/RH/knots"].set_property("visible", False)

    @staticmethod
    def add_fixed_distance_between_points(path):
        aux_frames = []
        with open(path, 'r') as f:
            yml = YAML().load(f)
            for fr in range(len(yml)):
                aux_frames.append(yml[fr])

        return aux_frames

    @staticmethod
    def visualize_simple_points(viewer, frame_name, points, color=None):
        """ Visualizes points in Meshcat
        Arguments
        ---------
        viewer: Meshcat Visualizer
        frame_name: Name of viewer to display in Meshcat controls
        points: ndarray
            Points arranged in size [N, dim] where N is the number
            of points, and dim is the dimension space, e.g., 2 for
            2-D, 3 for 3D.
        """
        if color is None:
            color = [1., 1., 0., 0.6]       # yellow
        r_bezier_pts = 0.01

        # Create yellow sphere
        obj = g.Sphere(r_bezier_pts)
        convert_rgba_to_meshcat_obj(obj, color)

        pt_number = 0
        for p in points:
            tf_pos = tf.translation_matrix(p)
            viewer["paths"][frame_name][str(pt_number)][str(pt_number)].set_object(
                obj, g.MeshLambertMaterial(color=obj.color))
            viewer["paths"][frame_name][str(pt_number)][str(pt_number)].set_transform(tf_pos)
            pt_number += 1

    @staticmethod
    def visualize_bezier_points(vis_viewer, frame, bezier_curve, segment=0):
        color_waypoint = [0., 1., 0., 0.6]      # blue
        color_transition = [1., 1., 0., 0.6]    # yellow
        r_bezier_pts = 0.01
        pt_number, seg_number = 0, 1
        for bez in bezier_curve:
            t, points = bez.get_sample_points()
            if np.ndim(points) == 3:
                points = points[0, :]
            for p in points:
                obj = g.Sphere(r_bezier_pts)
                tf_pos = tf.translation_matrix(p)

                # check if "in-between" waypoint or "transition" waypoint
                if (pt_number == seg_number*(len(points))-1) or (pt_number == (seg_number-1) * (len(points))):
                    convert_rgba_to_meshcat_obj(obj, color_transition)
                else:
                    convert_rgba_to_meshcat_obj(obj, color_waypoint)
                vis_viewer["paths"][frame][str(segment)][str(pt_number)].set_object(
                    obj, g.MeshLambertMaterial(color=obj.color))
                vis_viewer["paths"][frame][str(segment)][str(pt_number)].set_transform(tf_pos)

                pt_number += 1
            seg_number += 1

    @staticmethod
    def update_plane_offset_from_root(_origin_pos, H, d):
        return d - H @ _origin_pos

    def debug_sample_points(self, visualizer, frame_name):
        # sample random points in 3D within [-1, 1] on all axes
        verts = 4.*np.random.random((10000, 3)).astype(np.float32) - 2.

        # keep points inside reachable region
        H = self.reachability_planes[frame_name]['H']
        d = self.reachability_planes[frame_name]['d']
        idx = 0
        for p in verts:
            if np.any(H @ p >= -d):
                verts[idx] = np.NAN
            else:
                obj = g.Sphere(0.01)
                tf_pos = tf.translation_matrix(p)
                visualizer.viewer["traversable_regions"]["point_cloud"][frame_name][str(idx)].set_object(
                    obj, g.MeshLambertMaterial())
                visualizer.viewer["traversable_regions"]["point_cloud"][frame_name][str(idx)].set_transform(tf_pos)

            idx += 1
