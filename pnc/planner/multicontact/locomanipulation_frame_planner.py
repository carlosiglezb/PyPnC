import pnc.planner.multicontact.fastpathplanning.fastpathplanning as fpp
from collections import OrderedDict
from pnc.planner.multicontact.frame_traversable_region import convert_rgba_to_meshcat_obj

import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from ruamel.yaml import YAML


class LocomanipulationFramePlanner:
    def __init__(self, traversable_regions_list, ee_offset_file_path,
                 starting_stance_foot='LF',
                 aux_frames_path=None):

        self.safe_boxes = OrderedDict()
        self.reachability_planes = OrderedDict()
        self.path = []
        self.frame_names = []
        self.starting_stance_foot = starting_stance_foot
        for region in traversable_regions_list:
            self.frame_names.append(region._frame_name)
            self.safe_boxes[region._frame_name] = region._plan_safe_box_list

            # do the torso at the very end to get reachability from contact frames
            if region._frame_name != 'torso':
                H, d_prime = self.extract_plane_eqn_from_coeffs(region._plane_coeffs)
                d_prime = self.update_plane_offset_from_root(region._origin_pos,
                                                                H, d_prime)
                self.reachability_planes[region._frame_name] = {'H': H, 'd': d_prime}

            if region._frame_name == starting_stance_foot:
                init_standing_pos = region._origin_pos

        # the torso must be reachable based on the frame in contact
        H, d_prime = self.add_offset_to_plane_eqn_from_file(starting_stance_foot, ee_offset_file_path,
                                                            init_standing_pos)
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

    def plan(self, p_init, p_term, T, alpha, der_init={}, der_term={}, verbose=True):
        S = self.safe_boxes
        R = self.reachability_planes
        A = self.aux_frames
        self.path = fpp.plan_multiple(S, R, p_init, p_term, T, alpha, der_init,
                                      der_term, verbose, A)

    def plot(self, visualizer):

        bezier_curve = self.path
        self.visualize_bezier_points(visualizer, 'LF', bezier_curve)
        # for frame in self.frame_names:
        #     bezier_curve = self.path[frame]
        #     self.visualize_bezier_points(visualizer, frame, bezier_curve)

    @staticmethod
    def add_fixed_distance_between_points(path):
        aux_frames = []
        with open(path, 'r') as f:
            yml = YAML().load(f)
            for fr in range(len(yml)):
                aux_frames.append(yml[fr])

        return aux_frames

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

    @staticmethod
    def extract_plane_eqn_from_coeffs(coeffs):
        H = np.zeros((len(coeffs), 3))
        d_vec = np.zeros((len(coeffs),))
        i = 0
        for h in coeffs:
            H[i] = np.array([h['a'], h['b'], h['c']])
            d_vec[i] = h['d']
            i += 1
        return H, d_vec

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
