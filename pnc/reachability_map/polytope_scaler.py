import sys
import os
from copy import deepcopy

import meshcat
import numpy as np

cwd = os.getcwd()
sys.path.append(cwd)

from ruamel.yaml import YAML
from util.polytope_math import extract_plane_eqn_from_coeffs
from util.pydrake_meshcat_interface import pydrake_geom_to_meshcat
from visualizer.meshcat_tools.meshcat_palette import *
import pydrake.geometry.optimization as mut


def main(args):
    scale = args.scale
    print(f"Scaling the polytopes by {scale}")

    save_loc = cwd + '/pnc/reachability_map/output/'

    frame_names = ['LF', 'RF', 'L_knee', 'R_knee', 'LH', 'RH']
    convex_hull_halfspace_path = {}
    for fr in frame_names:
        convex_hull_halfspace_path[fr] = cwd + '/pnc/reachability_map/output/draco3_' + fr + '.yaml'

    # Scale
    original_pydrake_geom = {}
    scaled_pydrake_geom = {}
    for fr in frame_names:
        with open(convex_hull_halfspace_path[fr], 'r') as f:
            yml = YAML().load(f)
            plane_coeffs = []
            for i_plane in range(len(yml)):
                plane_coeffs.append(yml[i_plane])
            A, b = extract_plane_eqn_from_coeffs(plane_coeffs)

            # scale the polytope
            original_pydrake_geom[fr] = mut.HPolyhedron(A, -b)
            scaled_pydrake_geom[fr] = original_pydrake_geom[fr].Scale(scale, center=np.array([0., 0., 0.]))

    # visualize original and scaled polytopes to check
    vis = meshcat.Visualizer()
    vis.open()
    vis.wait()
    vis['/Background'].set_property('visible', False)
    for fr in frame_names:
        original_mcat_geom = pydrake_geom_to_meshcat(original_pydrake_geom[fr])
        scaled_mcat_geom = pydrake_geom_to_meshcat(scaled_pydrake_geom[fr])
        vis['original'][fr].set_object(original_mcat_geom, meshcat_domain_obj())
        vis['scaled'][fr].set_object(scaled_mcat_geom, meshcat_iris_obj())

    # save scaled polytopes
    yaml = YAML()
    plane_dict = {'a': 0., 'b': 0., 'c': 0., 'd': 0.}
    for fr in frame_names:
        filename = 'draco3_' + fr + '_scaled'
        plane_coeffs_list = []
        for i in range(scaled_pydrake_geom[fr].A().shape[0]):
            plane_dict['a'] = float(scaled_pydrake_geom[fr].A()[i][0])
            plane_dict['b'] = float(scaled_pydrake_geom[fr].A()[i][1])
            plane_dict['c'] = float(scaled_pydrake_geom[fr].A()[i][2])
            plane_dict['d'] = -float(scaled_pydrake_geom[fr].b()[i])
            plane_coeffs_list.append(deepcopy(plane_dict))
        with open(save_loc + filename + '.yaml', 'w') as f:
            yaml.dump(plane_coeffs_list, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=0.85,
                        help="Scale factor for the polytopes")
    args = parser.parse_args()
    main(args)

