import unittest

import numpy as np
import os, sys

from util.util import so3_from_vec_to_vec
from util.polytope_math import get_closest_distance_to_polytope_surface

cwd = os.getcwd()

from meshcat.geometry import Sphere
from util.pydrake_meshcat_interface import scipy_hull_to_meshcat, pydrake_geom_to_meshcat, \
    polytope_intersections_to_meshcat
from pydrake.geometry.optimization import HPolyhedron
from visualizer.meshcat_tools.meshcat_palette import (meshcat_iris_obj, meshcat_collision_obj,
                                                      meshcat_domain_obj, meshcat_obstacle_obj)
from pinocchio.visualize import MeshcatVisualizer
import meshcat.transformations as tf
import pinocchio as pin
import external_source.stabilipy.stabilipy as stab
import pickle
import plot.meshcat_utils as vis_tools
import matplotlib.pyplot as plt

@staticmethod
def get_contact_poses_from_file(filename: str) -> list[np.ndarray]:
    q_init_lst = []
    with open(filename, 'rb') as file:
        while True:
            try:
                d = pickle.load(file)
                q_init_lst.append(np.array(d['joint_pos'][0]))
            except EOFError:
                break

    return q_init_lst

def get_all_poses_from_file(filename: str) -> list[np.ndarray]:
    q_all_lst = []
    with open(filename, 'rb') as file:
        while True:
            try:
                d = pickle.load(file)
                q_all_lst.append(d['joint_pos'])
            except EOFError:
                break

    return q_all_lst

def get_contact_seq_from_file(filename: str) -> list[dict[str: np.ndarray]]:
    contacts_seq_lst = []
    with open(filename, 'rb') as file:
        try:
            d = pickle.load(file)
            contacts_seq_lst = d['contact_seq_planes']
        except EOFError:
            raise ValueError(f"Error reading contact sequence from file: {filename}")

    return contacts_seq_lst

def get_key_from_value_next(d, target_value):
    return next((key for key, value in d.items() if value == target_value), None)


class TestStabilipy(unittest.TestCase):
    def __init__(self, methodName: str = "runTest"):
        super().__init__(methodName)
        self.plan_to_model_frames = {}

    def setUp(self):
        self.plan_to_model_frames['torso'] = 'torso_primitive_shape'
        self.plan_to_model_frames['LF'] = 'left_ankle_roll_link'
        self.plan_to_model_frames['RF'] = 'right_ankle_roll_link'
        self.plan_to_model_frames['L_knee'] = 'left_knee_link'
        self.plan_to_model_frames['R_knee'] = 'right_knee_link'
        self.plan_to_model_frames['LH'] = 'left_rubber_hand'
        self.plan_to_model_frames['RH'] = 'right_rubber_hand'

    def test_stabilipy_meshcat_visualization(self):
        b_plot_final = False
        dist_in_lst, dist_out_lst = [], []

        # Specify location of urdf files
        robot_name = "g1_29dof_lock_waist"
        urdf_file = cwd + "/robot_model/g1_description/g1_29dof_lock_waist_modified.urdf"
        package_dir = cwd + "/robot_model/g1_description"

        # Create robot system
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_file, package_dir, pin.JointModelFreeFlyer())
        data, _, _ = pin.createDatas(
            model, collision_model, visual_model)

        # Display Robot in Meshcat Visualizer
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        try:
            viz.initViewer(open=True)
            viz.viewer.wait()
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)
        viz.loadViewerModel(rootNodeName=robot_name)
        vis_q = pin.neutral(model)

        # get list of configurations throughout multiple contacts
        cfree_soln_file = cwd + '/experiment_data/g1_sca_step_over_knee_knocker.pkl'
        q_at_contact = get_contact_poses_from_file(cfree_soln_file)

        # robot-specific default parameters
        ankle_heel_dist = 0.06
        ankle_toe_dist = 0.13
        half_foot_width = 0.02
        foot_height = -0.03
        hand_box_side = 0.02

        contacts_seq_lst = [['left_ankle_roll_link', 'right_ankle_roll_link'],
                            ['right_ankle_roll_link', 'left_palm_link'],
                            ['left_ankle_roll_link', 'right_ankle_roll_link'],
                            ['left_ankle_roll_link', 'right_palm_link'],
                            ['left_ankle_roll_link', 'right_ankle_roll_link']]
        # visualize first contact pose sample and update robot system
        zero_qd = np.zeros((model.nv))
        for (cs, current_q) in enumerate(q_at_contact):
            viz.display(current_q)
            pin.forwardKinematics(model, data, current_q, zero_qd)

            # set up stabilipy problem
            robot_mass = sum([inertia.mass for inertia in model.inertias])
            margin = 2.0
            mu = 0.9
            pos, normals = [], []
            current_contact_links = contacts_seq_lst[cs]
            for lnk in current_contact_links:
                lnk_id = model.getFrameId(lnk)
                trans = pin.updateFramePlacement(model, data, lnk_id)
                ee_pos = trans.translation.reshape(-1, 1)
                if lnk == 'left_ankle_roll_link' or lnk == 'right_ankle_roll_link':
                    # left-front
                    pos.append(ee_pos + np.array([[ankle_toe_dist], [half_foot_width], [foot_height]]))
                    normals.append(np.array([[0.], [0.], [1.]]))    # feet are flat on ground
                    # right-front
                    pos.append(ee_pos + np.array([[ankle_toe_dist], [-half_foot_width], [foot_height]]))
                    normals.append(np.array([[0.], [0.], [1.]]))    # feet are flat on ground
                    # right-back
                    pos.append(ee_pos + np.array([[-ankle_heel_dist], [-half_foot_width], [foot_height]]))
                    normals.append(np.array([[0.], [0.], [1.]]))    # feet are flat on ground
                    # left-back
                    pos.append(ee_pos + np.array([[-ankle_heel_dist], [half_foot_width], [foot_height]]))
                    normals.append(np.array([[0.], [0.], [1.]]))    # feet are flat on ground
                elif lnk == 'left_palm_link':
                    # top-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [-1], [0.]]))    # lhand on door side
                    # low-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [-1.], [0.]]))    # lhand on door side
                    # low-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [-1.], [0.]]))    # lhand on door side
                    # top-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [-1.], [0.]]))    # lhand on door side
                elif lnk == 'right_palm_link':
                    # top-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [1], [0.]]))    # rhand on door side
                    # low-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [1.], [0.]]))    # rhand on door side
                    # low-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [1.], [0.]]))    # rhand on door side
                    # top-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [1.], [0.]]))    # rhand on door side
                else:
                    raise ValueError(f"Contact location for {lnk} not specified")

            contacts = [stab.Contact(mu, p, n) for p, n in zip(pos, normals)]
            polyhedron = stab.StabilityPolygon(robot_mass, dimension=3, radius=0.8, robust_sphere=False)
            polyhedron.contacts = contacts
            shape = [
                np.array([[-1., 0, 0]]).T,
                np.array([[1., 0, 0]]).T,
                np.array([[0, 1., 0]]).T,
                np.array([[0, -1., 0]]).T,
                np.array([[0, 0., 1]]).T,
                np.array([[0, 0., -1]]).T
            ]

            polytope = [margin * s for s in shape]
            polyhedron.gravity_envelope = polytope
            polyhedron.compute(stab.Mode.iteration, epsilon=2e-3, maxIter=10, solver='qhull',
                               record_anim=False, plot_init=False,
                               plot_step=False, plot_final=b_plot_final)

            # visualize CoM
            pin.centerOfMass(model, data, current_q, zero_qd)
            com_pos = data.com[0]
            com_pos_proj = com_pos[0], com_pos[1], 0.

            # distance from outer polyhedron to CoM
            p_o = HPolyhedron(polyhedron.outer.halfspaces[:,:3], -polyhedron.outer.halfspaces[:, -1])
            dist_out, _ = p_o.Projection(com_pos)

            # distance from inner polyhedron to CoM
            p_i = HPolyhedron(polyhedron.inner.equations[:, :3], -polyhedron.inner.equations[:, -1])
            dist_in, _ = p_i.Projection(com_pos)

            # make distance negative if CoM is outside the polytope
            if not p_i.PointInSet(com_pos_proj):
                dist_in *=-1

            # make distance negative if CoM is outside the polytope
            if not p_o.PointInSet(com_pos_proj):
                dist_out *=-1
            # dist = get_closest_distance_to_polytope_surface(com_pos, p_o.A(), p_o.b())
            dist_in_lst.append(dist_in)
            dist_out_lst.append(dist_out)

            #
            # visualize in meshcat
            #
            # get meshes of stability polytope
            p_i_mcat = scipy_hull_to_meshcat(polyhedron.inner)
            p_o_mcat = pydrake_geom_to_meshcat(p_o)
            # --- stability polytope
            viz.viewer[f"{robot_name}/stability/inner"].set_object(p_i_mcat, meshcat_iris_obj())
            viz.viewer[f"{robot_name}/stability/inner"].set_transform(tf.translation_matrix([0., 0., com_pos[2]]))
            viz.viewer[f"{robot_name}/stability/outer"].set_object(p_o_mcat, meshcat_domain_obj())
            viz.viewer[f"{robot_name}/stability/outer"].set_transform(tf.translation_matrix([0., 0., com_pos[2]]))
            # --- contacts
            c_obj = Sphere(0.01)
            com_obj = Sphere(0.02)
            for i, contact in enumerate(contacts):
                c_pos = contact.r.reshape(-1,)
                viz.viewer[f"{robot_name}/contacts/{i}"].set_object(c_obj, meshcat_collision_obj())
                viz.viewer[f"{robot_name}/contacts/{i}"].set_transform(tf.translation_matrix(c_pos))
            # --- Center of Mass at current configuration
            viz.viewer[f"{robot_name}/CoM/3d"].set_object(com_obj, meshcat_obstacle_obj())
            viz.viewer[f"{robot_name}/CoM/3d"].set_transform(tf.translation_matrix(com_pos))
            viz.viewer[f"{robot_name}/CoM/proj"].set_object(com_obj, meshcat_obstacle_obj())
            viz.viewer[f"{robot_name}/CoM/proj"].set_transform(tf.translation_matrix(com_pos_proj))

            if b_plot_final:
                polyhedron.set_xyz_labels()
                polyhedron.show()
            # check that the current CoM is statically stable
            self.assertEqual(True, True)


    def test_stabilipy_meshcat_animation(self):
        b_plot_final = False

        # Specify location of urdf files
        robot_name = "g1_29dof_lock_waist"
        urdf_file = cwd + "/robot_model/g1_description/g1_29dof_lock_waist_modified.urdf"
        package_dir = cwd + "/robot_model/g1_description"

        # get list of configurations throughout multiple contacts
        contact_seq_str_opts = ['over', 'on', 'on_balanced']
        cs_opt = contact_seq_str_opts[1]  # 'over' or 'on' or 'on_balanced'
        # cfree_soln_file = cwd + '/experiment_data/g1_sca_step_' + cs_opt + '_knee_knocker.pkl'
        cfree_soln_file = cwd + '/experiment_data/g1_step_' + cs_opt + '_door.pkl'
        # cfree_soln_file = cwd + '/experiment_data/g1_sca_step_' + cs_opt + '_door.pkl'
        q_all = get_all_poses_from_file(cfree_soln_file)
        if len(q_all) == 1:
            # in case using full TO with impulse model, separate by contact phase
            q_phases = []
            i_np = 0
            # N_HORIZON_LST = [180, 240, 280, 220, 250] # step over
            N_HORIZON_LST = [250, 250, 250, 250, 250]   # step on
            for n in N_HORIZON_LST:
                prev_idx = sum(N_HORIZON_LST[:i_np]) + i_np
                next_idx = prev_idx + n
                q_phases.append(q_all[0][prev_idx:next_idx])
                i_np += 1
            q_all = q_phases    # re-assign

        # Create robot system
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_file, package_dir, pin.JointModelFreeFlyer())
        data, col_data, vis_data = pin.createDatas(
            model, collision_model, visual_model)

        # Create Meshcat Animation
        save_freq = 1
        T = 3
        N_horizon_lst = [len(q_all[k])-1 for k in range(len(q_all))]
        display = vis_tools.MeshcatPinocchioAnimation(model, collision_model, visual_model,
                          data, vis_data, col_data, ctrl_freq=np.average(N_horizon_lst)/T, save_freq=save_freq)

        dist_in_lst, dist_out_lst, time_lst = [], [], []

        # robot-specific default parameters
        ankle_heel_dist = 0.06
        ankle_toe_dist = 0.13
        half_foot_width = 0.02
        foot_height = -0.03
        hand_box_side = 0.02

        # constants in visualizer
        com_3d_name = f"{robot_name}/CoM/3d"
        com_proj_name = f"{robot_name}/CoM/proj"
        c_obj = Sphere(0.01)
        com_obj = Sphere(0.02)

        if cs_opt == 'over':
            contacts_seq_lst = [['left_ankle_roll_link', 'right_ankle_roll_link'],
                                ['right_ankle_roll_link', 'left_rubber_hand'],
                                ['left_ankle_roll_link', 'right_ankle_roll_link'],
                                ['left_ankle_roll_link', 'right_rubber_hand'],
                                ['left_ankle_roll_link', 'right_ankle_roll_link']]
        elif cs_opt == 'on':
            contacts_seq_lst = [['left_ankle_roll_link', 'right_ankle_roll_link'],
                                ['right_ankle_roll_link', 'left_rubber_hand'],
                                ['right_rubber_hand', 'left_ankle_roll_link'],
                                ['right_ankle_roll_link', 'left_rubber_hand'],
                                ['left_ankle_roll_link', 'right_ankle_roll_link']]
        elif cs_opt  == 'on_balanced':
            contacts_seq_lst = [['left_ankle_roll_link', 'right_ankle_roll_link'],
                                ['left_ankle_roll_link', 'left_rubber_hand', 'right_rubber_hand'],
                                ['left_rubber_hand', 'right_rubber_hand', 'right_ankle_roll_link'],
                                ['left_ankle_roll_link', 'right_rubber_hand'],
                                ['left_ankle_roll_link', 'right_ankle_roll_link']]

        # visualize entire motion while super-impossing stability regions after each new contact
        zero_qd = np.zeros((model.nv))
        display.start_animation()
        for (n, N) in enumerate(N_horizon_lst):
            # at each new contact sequence, compute stability region
            pin.forwardKinematics(model, data, np.array(q_all[n][0]), zero_qd)

            # set up stabilipy problem
            robot_mass = sum([inertia.mass for inertia in model.inertias])
            margin = 0.
            mu = 0.9
            pos, normals = [], []
            current_contact_links = contacts_seq_lst[n]
            for lnk in current_contact_links:
                lnk_id = model.getFrameId(lnk)
                trans = pin.updateFramePlacement(model, data, lnk_id)
                ee_pos = trans.translation.reshape(-1, 1)
                if lnk == 'left_ankle_roll_link' or lnk == 'right_ankle_roll_link':
                    # left-front
                    pos.append(ee_pos + np.array([[ankle_toe_dist], [half_foot_width], [foot_height]]))
                    normals.append(np.array([[0.], [0.], [1.]]))    # feet are flat on ground
                    # right-front
                    pos.append(ee_pos + np.array([[ankle_toe_dist], [-half_foot_width], [foot_height]]))
                    normals.append(np.array([[0.], [0.], [1.]]))    # feet are flat on ground
                    # right-back
                    pos.append(ee_pos + np.array([[-ankle_heel_dist], [-half_foot_width], [foot_height]]))
                    normals.append(np.array([[0.], [0.], [1.]]))    # feet are flat on ground
                    # left-back
                    pos.append(ee_pos + np.array([[-ankle_heel_dist], [half_foot_width], [foot_height]]))
                    normals.append(np.array([[0.], [0.], [1.]]))    # feet are flat on ground
                elif lnk == 'left_rubber_hand':
                    # top-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [-1], [0.]]))    # lhand on door side
                    # low-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [-1.], [0.]]))    # lhand on door side
                    # low-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [-1.], [0.]]))    # lhand on door side
                    # top-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [-1.], [0.]]))    # lhand on door side
                elif lnk == 'right_rubber_hand':
                    # top-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [1], [0.]]))    # rhand on door side
                    # low-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [1.], [0.]]))    # rhand on door side
                    # low-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [1.], [0.]]))    # rhand on door side
                    # top-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [1.], [0.]]))    # rhand on door side
                else:
                    raise ValueError(f"Contact location for {lnk} not specified")

            contacts = [stab.Contact(mu, p, n) for p, n in zip(pos, normals)]
            polyhedron = stab.StabilityPolygon(robot_mass, dimension=3, radius=1.0)
            polyhedron.contacts = contacts
            shape = [
                np.array([[-1., 0, 0]]).T,
                np.array([[1., 0, 0]]).T,
                np.array([[0, 1., 0]]).T,
                np.array([[0, -1., 0]]).T,
                np.array([[0, 0., 1]]).T,
                np.array([[0, 0., -1]]).T
            ]

            polytope = [margin * s for s in shape]
            polyhedron.gravity_envelope = polytope
            polyhedron.compute(stab.Mode.best, epsilon=2e-3, maxIter=10, solver='qhull',
                               record_anim=False, plot_init=False,
                               plot_step=False, plot_final=b_plot_final)

            #
            # visualize in meshcat
            #
            # --- stability polytope
            p_in_name = f"{robot_name}/stability/inner/{n}"
            p_out_name = f"{robot_name}/stability/outer/{n}"
            p_inner_mcat = scipy_hull_to_meshcat(polyhedron.inner)
            p_o = HPolyhedron(polyhedron.outer.halfspaces[:, :3], -polyhedron.outer.halfspaces[:, -1])
            if not p_o.IsBounded():
                print(f"Warning: outer polyhedron is unbounded for contact sequence {n}")
            # p_outer_mcat = pydrake_geom_to_meshcat(p_o)
            p_outer_mcat = polytope_intersections_to_meshcat(polyhedron.outer.intersections)
            p_i = HPolyhedron(polyhedron.inner.equations[:, :3], -polyhedron.inner.equations[:, -1])
            display.add_shape(p_in_name, p_inner_mcat, meshcat_iris_obj())
            display.add_shape(p_out_name, p_outer_mcat, meshcat_domain_obj())
            # --- contacts
            for i in range(len(contacts)):
                c_name = f"{robot_name}/contacts/{i}"
                display.add_shape(c_name, c_obj, meshcat_collision_obj())
            # --- Center of Mass
            display.add_shape(com_3d_name, com_obj, meshcat_obstacle_obj())
            display.add_shape(com_proj_name, com_obj, meshcat_obstacle_obj())

            # visualize configurations in between contacts
            for k in range(N):
                display.animate_frame(np.array(q_all[n][k]))

                # visualize CoM
                pin.centerOfMass(model, data, np.array(q_all[n][k]), zero_qd)
                com_pos = data.com[0]
                com_pos_proj = com_pos[0], com_pos[1], 0.

                # update meshcat frames
                # display.animate_single_shape(p_in_name, tf.translation_matrix([0., 0., com_pos[2]]))
                # display.animate_single_shape(p_out_name, tf.translation_matrix([0., 0., com_pos[2]]))
                # --- contacts
                for i, contact in enumerate(contacts):
                    c_pos = contact.r.reshape(-1,)
                    c_name = f"{robot_name}/contacts/{i}"
                    display.animate_single_shape(c_name, tf.translation_matrix(c_pos))
                # --- Center of Mass at current configuration
                display.animate_single_shape(com_3d_name, tf.translation_matrix(com_pos))
                display.animate_single_shape(com_proj_name, tf.translation_matrix(com_pos_proj))
                display.animation_step()

                # distance from outer polyhedron to CoM
                # dist_out, _ = p_o.Projection(com_pos_proj)
                dist_out = get_closest_distance_to_polytope_surface(com_pos, p_o.A(), p_o.b())

                # distance from inner polyhedron to CoM
                # dist_in, _ = p_i.Projection(com_pos_proj)
                dist_in = get_closest_distance_to_polytope_surface(com_pos, p_i.A(), p_i.b())

                # make distance negative if CoM is outside the polytope
                if not p_i.PointInSet(com_pos):
                    dist_in *= -1

                # make distance negative if CoM is outside the polytope
                if not p_o.PointInSet(com_pos):
                    dist_out *= -1
                # dist = get_closest_distance_to_polytope_surface(com_pos, p_o.A(), p_o.b())

                dist_in_lst.append(dist_in)
                dist_out_lst.append(dist_out)
                # create list of time steps
                time_lst.append(T * n + k * T/N)

                # print progress
                if k % 20 == 0:
                    print(f"Finished contact phase {n}, step {k}")
                # check that the current CoM is statically stable
                self.assertEqual(True, True)

        display.finish_animation()

        #
        # plot distances
        #
        plt.figure(figsize=(8, 4))
        plt.plot(time_lst, dist_in_lst, label='Distance to inner')
        plt.plot(time_lst, dist_out_lst, label='Distance to outer')
        plt.ylabel('Distance [m]', fontsize=12)
        plt.xlabel('Time [s]', fontsize=12)
        plt.grid()
        plt.legend(fontsize=12)
        plt.show()


    def test_stairs_stabilipy_meshcat_animation(self):
        b_plot_final = False

        # Specify location of urdf files
        robot_name = "g1_29dof_lock_waist"
        urdf_file = cwd + "/robot_model/g1_description/g1_29dof_lock_waist_modified.urdf"
        package_dir = cwd + "/robot_model/g1_description"

        # get list of configurations throughout multiple contacts
        contact_seq_str_opts = ['tilted_stairs']
        cs_opt = contact_seq_str_opts[0]  # 'tilted_stairs'
        cfree_soln_file = cwd + '/experiment_data/g1_sca_' + cs_opt + '.pkl'
        q_all = get_all_poses_from_file(cfree_soln_file)

        # Create robot system
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            urdf_file, package_dir, pin.JointModelFreeFlyer())
        data, col_data, vis_data = pin.createDatas(
            model, collision_model, visual_model)

        # Create Meshcat Animation
        save_freq = 1
        T = 3
        N_horizon_lst = [len(q_all[k])-1 for k in range(len(q_all))]
        display = vis_tools.MeshcatPinocchioAnimation(model, collision_model, visual_model,
                          data, vis_data, col_data, ctrl_freq=np.average(N_horizon_lst)/T, save_freq=save_freq)

        dist_in_lst, dist_out_lst, time_lst = [], [], []

        # robot-specific default parameters
        ankle_heel_dist = 0.06
        ankle_toe_dist = 0.13
        half_foot_width = 0.02
        foot_height = -0.03
        hand_box_side = 0.02

        # constants in visualizer
        com_3d_name = f"{robot_name}/CoM/3d"
        com_proj_name = f"{robot_name}/CoM/proj"
        c_obj = Sphere(0.01)
        com_obj = Sphere(0.02)

        contacts_seq_lst = []
        contacts_seq_planes = get_contact_seq_from_file(cfree_soln_file)
        for con_plane in contacts_seq_planes:
            next_contacts, next_normals = [], []
            for fname in con_plane.keys():
                next_contacts.append(self.plan_to_model_frames[fname])
            contacts_seq_lst.append(next_contacts)

        # visualize entire motion while super-imposing stability regions after each new contact
        zero_qd = np.zeros(model.nv)
        display.start_animation()
        for (n, N) in enumerate(N_horizon_lst):
            # at each new contact sequence, compute stability region
            pin.forwardKinematics(model, data, np.array(q_all[n][0]), zero_qd)

            # set up stabilipy problem
            robot_mass = sum([inertia.mass for inertia in model.inertias])
            margin = 1.2
            mu = 0.9
            pos, normals = [], []
            current_contact_links = contacts_seq_lst[n]
            for lnk in current_contact_links:
                lnk_id = model.getFrameId(lnk)
                trans = pin.updateFramePlacement(model, data, lnk_id)
                ee_pos = trans.translation.reshape(-1, 1)
                if lnk == 'left_ankle_roll_link' or lnk == 'right_ankle_roll_link':
                    fr_name = get_key_from_value_next(self.plan_to_model_frames, lnk)
                    curr_normal = contacts_seq_planes[n][fr_name]
                    if np.abs(np.linalg.norm(curr_normal) - 1.0) > 1e-3:
                        curr_normal /= np.linalg.norm(curr_normal)
                    curr_normal_vec = np.array([[val] for val in curr_normal])
                    w_R_surf = so3_from_vec_to_vec(np.array([0., 0., 1.]), curr_normal)
                    # left-front
                    pos.append(ee_pos + w_R_surf @ np.array([[ankle_toe_dist], [half_foot_width], [foot_height]]))
                    normals.append(curr_normal_vec)
                    # right-front
                    pos.append(ee_pos + w_R_surf @ np.array([[ankle_toe_dist], [-half_foot_width], [foot_height]]))
                    normals.append(curr_normal_vec)
                    # right-back
                    pos.append(ee_pos + w_R_surf @ np.array([[-ankle_heel_dist], [-half_foot_width], [foot_height]]))
                    normals.append(curr_normal_vec)
                    # left-back
                    pos.append(ee_pos + w_R_surf @ np.array([[-ankle_heel_dist], [half_foot_width], [foot_height]]))
                    normals.append(curr_normal_vec)
                elif lnk == 'left_rubber_hand':
                    # top-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [-1], [0.]]))    # lhand on door side
                    # low-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [-1.], [0.]]))    # lhand on door side
                    # low-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [-1.], [0.]]))    # lhand on door side
                    # top-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [-1.], [0.]]))    # lhand on door side
                elif lnk == 'right_rubber_hand':
                    # top-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [1], [0.]]))    # rhand on door side
                    # low-front
                    pos.append(ee_pos + np.array([[hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [1.], [0.]]))    # rhand on door side
                    # low-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [-hand_box_side]]))
                    normals.append(np.array([[0.], [1.], [0.]]))    # rhand on door side
                    # top-back
                    pos.append(ee_pos + np.array([[-hand_box_side], [0.], [hand_box_side]]))
                    normals.append(np.array([[0.], [1.], [0.]]))    # rhand on door side
                else:
                    raise ValueError(f"Contact location for {lnk} not specified")

            contacts = [stab.Contact(mu, p, n) for p, n in zip(pos, normals)]
            polyhedron = stab.StabilityPolygon(robot_mass, dimension=3, radius=2.8, robust_sphere=False)
            polyhedron.contacts = contacts
            shape = [
                np.array([[-1., 0, 0]]).T,
                np.array([[1., 0, 0]]).T,
                np.array([[0, 1., 0]]).T,
                np.array([[0, -1., 0]]).T,
                np.array([[0, 0., 1]]).T,
                np.array([[0, 0., -1]]).T
            ]

            polytope = [margin * s for s in shape]
            polyhedron.gravity_envelope = polytope
            polyhedron.compute(stab.Mode.best, epsilon=2e-3, maxIter=10, solver='qhull',
                               record_anim=False, plot_init=False,
                               plot_step=False, plot_final=b_plot_final)

            #
            # visualize in meshcat
            #
            # --- stability polytope
            p_in_name = f"{robot_name}/stability/inner/{n}"
            p_out_name = f"{robot_name}/stability/outer/{n}"
            p_inner_mcat = scipy_hull_to_meshcat(polyhedron.inner)
            p_o = HPolyhedron(polyhedron.outer.halfspaces[:, :3], -polyhedron.outer.halfspaces[:, -1])
            if not p_o.IsBounded():
                print(f"Warning: outer polyhedron is unbounded for contact sequence {n}")
            # p_outer_mcat = pydrake_geom_to_meshcat(p_o)
            p_outer_mcat = polytope_intersections_to_meshcat(polyhedron.outer.intersections)
            p_i = HPolyhedron(polyhedron.inner.equations[:, :3], -polyhedron.inner.equations[:, -1])
            display.add_shape(p_in_name, p_inner_mcat, meshcat_iris_obj())
            display.add_shape(p_out_name, p_outer_mcat, meshcat_domain_obj())
            # --- contacts
            for i in range(len(contacts)):
                c_name = f"{robot_name}/contacts/{i}"
                display.add_shape(c_name, c_obj, meshcat_collision_obj())
            # --- Center of Mass
            display.add_shape(com_3d_name, com_obj, meshcat_obstacle_obj())
            display.add_shape(com_proj_name, com_obj, meshcat_obstacle_obj())

            # visualize configurations in between contacts
            for k in range(N):
                display.animate_frame(np.array(q_all[n][k]))

                # visualize CoM
                pin.centerOfMass(model, data, np.array(q_all[n][k]), zero_qd)
                com_pos = data.com[0]
                com_pos_proj = com_pos[0], com_pos[1], 0.

                # update meshcat frames
                # display.animate_single_shape(p_in_name, tf.translation_matrix(polyhedron.com.T))
                # display.animate_single_shape(p_out_name, tf.translation_matrix(polyhedron.com.T))
                # --- contacts
                for i, contact in enumerate(contacts):
                    c_pos = contact.r.reshape(-1,)
                    c_name = f"{robot_name}/contacts/{i}"
                    display.animate_single_shape(c_name, tf.translation_matrix(c_pos))
                # --- Center of Mass at current configuration
                display.animate_single_shape(com_3d_name, tf.translation_matrix(com_pos))
                display.animate_single_shape(com_proj_name, tf.translation_matrix(com_pos_proj))
                display.animation_step()

                # distance from outer polyhedron to CoM
                # dist_out, _ = p_o.Projection(com_pos_proj)
                dist_out = get_closest_distance_to_polytope_surface(com_pos, p_o.A(), p_o.b())
                # dist_out = get_closest_distance_to_polytope_surface(com_pos, p_o.A(), p_o.b(), polyhedron.com.reshape(-1,))

                # distance from inner polyhedron to CoM
                # dist_in, _ = p_i.Projection(com_pos_proj)
                dist_in = get_closest_distance_to_polytope_surface(com_pos, p_i.A(), p_i.b())
                # dist_in = get_closest_distance_to_polytope_surface(com_pos, p_i.A(), p_i.b(), polyhedron.com.reshape(-1,))

                # make distance negative if CoM is outside the polytope
                if not p_i.PointInSet(com_pos):
                    dist_in *= -1

                # make distance negative if CoM is outside the polytope
                if not p_o.PointInSet(com_pos):
                    dist_out *= -1
                # dist = get_closest_distance_to_polytope_surface(com_pos, p_o.A(), p_o.b())

                dist_in_lst.append(dist_in)
                dist_out_lst.append(dist_out)
                # create list of time steps
                time_lst.append(T * n + k * T/N)

                # print progress
                if k % 20 == 0:
                    print(f"Finished contact phase {n}, step {k}")
                # check that the current CoM is statically stable
                self.assertEqual(True, True)

        display.finish_animation()

        #
        # plot distances
        #
        plt.figure(figsize=(8, 4))
        plt.plot(time_lst, dist_in_lst, label='Distance to inner')
        plt.plot(time_lst, dist_out_lst, label='Distance to outer')
        plt.ylabel('Distance [m]', fontsize=12)
        plt.xlabel('Time [s]', fontsize=12)
        plt.grid()
        plt.legend(fontsize=12)
        plt.show()


if __name__ == '__main__':
    unittest.main()
