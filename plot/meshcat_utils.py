import os
import sys
from typing import List

import numpy as np
import coal
import pinocchio as pin
from meshcat.geometry import TriangularMeshGeometry

# Pinocchio Meshcat
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf

# Python-Meshcat
from meshcat.animation import Animation
from pinocchio.visualize.meshcat_visualizer import hasMeshFileInfo

# Crocoddyl tools
from crocoddyl.libcrocoddyl_pywrap import *  # noqa
from pydrake.geometry.optimization import HPolyhedron

from util.pydrake_meshcat_interface import pydrake_geom_to_meshcat
from util.util import vec_to_roll_pitch
from visualizer.meshcat_tools.meshcat_palette import meshcat_iris_obj, meshcat_obstacle_obj, meshcat_domain_obj, \
    meshcat_point_obj, PURPLE, GREEN, GREY, BLUE, YELLOW

cwd = os.getcwd()
sys.path.append(cwd)


def get_force_trajectory_from_solver(solver):
    """
    Snippet copied from Crocoddyl's DisplayAbstract class
    """
    fs = []
    models = [*solver.problem.runningModels.tolist()]
    datas = [*solver.problem.runningDatas.tolist()]
    for i, data in enumerate(datas):
        model = models[i]
        if hasattr(data, "differential"):
            if isinstance(
                data.differential,
                DifferentialActionDataContactFwdDynamics,
            ) or isinstance(
                data.differential,
                DifferentialActionDataContactInvDynamics,
            ):
                fc = []
                for (
                    key,
                    contact,
                ) in data.differential.multibody.contacts.contacts.todict().items():
                    if model.differential.contacts.contacts[key].active:
                        joint = model.differential.state.pinocchio.frames[
                            contact.frame
                        ].parentJoint
                        oMi = contact.pinocchio.oMi[joint]
                        oMf = oMi * contact.jMf
                        fiMo = pin.SE3(
                            oMi.rotation.T,
                            contact.jMf.translation,
                        )
                        force = fiMo.actInv(contact.f)
                        w_force = fiMo.act(force)
                        # w_force = oMi.act(force)
                        R = np.eye(3)
                        mu = 0.7
                        for k, c in model.differential.costs.costs.todict().items():
                            if isinstance(
                                c.cost.residual,
                                ResidualModelContactFrictionCone,
                            ):
                                if contact.frame == c.cost.residual.id:
                                    R = c.cost.residual.reference.R
                                    mu = c.cost.residual.reference.mu
                                    continue
                        fc.append(
                            {
                                "key": str(joint),
                                "oMf": oMf,
                                "f": force,
                                "w_f": w_force,
                                "R": R,
                                "mu": mu,
                            }
                        )
                fs.append(fc)
            elif isinstance(data.differential, StdVec_DiffActionData):
                fc = []
                for key, contact in (
                    data.differential[0]
                    .multibody.contacts.contacts.todict()
                    .items()
                ):
                    if model.differential.contacts.contacts[key].active:
                        joint = model.differential.state.pinocchio.frames[
                            contact.frame
                        ].parentJoint
                        oMf = contact.pinocchio.oMi[joint] * contact.jMf
                        fiMo = pin.SE3(
                            contact.pinocchio.oMi[joint].rotation.T,
                            contact.jMf.translation,
                        )
                        force = fiMo.actInv(contact.fext)
                        w_force = fiMo.act(force)
                        R = np.eye(3)
                        mu = 0.7
                        for k, c in model.differential.costs.costs.todict().items():
                            if isinstance(
                                c.cost.residual,
                                ResidualModelContactFrictionCone,
                            ):
                                if contact.frame == c.cost.residual.id:
                                    R = c.cost.residual.reference.R
                                    mu = c.cost.residual.reference.mu
                                    continue
                        fc.append(
                            {
                                "key": str(joint),
                                "oMf": oMf,
                                "f": contact.fext,
                                "w_f": w_force,
                                "R": R,
                                "mu": mu,
                            }
                        )
                fs.append(fc)
        elif isinstance(data, ActionDataImpulseFwdDynamics):
            fc = []
            for key, impulse in data.multibody.impulses.impulses.todict().items():
                if model.impulses.impulses[key].active:
                    joint = model.state.pinocchio.frames[impulse.frame].parentJoint
                    oMf = impulse.pinocchio.oMi[joint] * impulse.jMf
                    fiMo = pin.SE3(
                        impulse.pinocchio.oMi[joint].rotation.T,
                        impulse.jMf.translation,
                    )
                    force = fiMo.actInv(impulse.f)
                    w_force = fiMo.act(force)
                    R = np.eye(3)
                    mu = 0.7
                    for k, c in model.costs.costs.todict().items():
                        if isinstance(
                            c.cost.residual,
                            ResidualModelContactFrictionCone,
                        ):
                            if impulse.frame == c.cost.residual.id:
                                R = c.cost.residual.reference.R
                                mu = c.cost.residual.reference.mu
                                continue
                    fc.append(
                        {
                            "key": str(joint),
                            "oMf": oMf,
                            "f": force,
                            "w_f": w_force,
                            "R": R,
                            "mu": mu,
                        }
                    )
            fs.append(fc)
    return fs


def get_scaled_and_oriented_grf_tf(scale,
                                   pos, ori,
                                   force_ori,
                                   arrow_ini_height=0.1):
    # scale and place GRF on ground plane (level)
    scale_in_z = tf.scale_matrix(scale, None, np.array([0., 0., 1.]))
    shift_in_z = tf.translation_matrix(np.array([0., 0., scale * arrow_ini_height / 2]))
    scaled_arrow_tf = tf.concatenate_matrices(shift_in_z, scale_in_z)

    # rotate according to force direction
    force_rpy = np.zeros(3)
    force_rpy[:2] = vec_to_roll_pitch(force_ori)
    force_ori_mat = tf.euler_matrix(force_rpy[0], force_rpy[1], force_rpy[2], 'sxyz')
    scaled_arrow_tf = tf.concatenate_matrices(force_ori_mat, scaled_arrow_tf)

    # translate arrow to frame position and orientation
    tf_pos = tf.translation_matrix(pos)
    # tf_pos[:3, :3] = ori
    scaled_arrow_tf = tf.concatenate_matrices(tf_pos, scaled_arrow_tf)

    return scaled_arrow_tf


class MeshcatPinocchioAnimation:
    def __init__(self, pin_robot_model, collision_model, visual_model,
                 robot_data, visual_data, collision_data,
                 ctrl_freq=1000, save_freq=50):
        # self.robot = pin_robot_model
        self.robot_data = robot_data
        self.model = pin_robot_model
        self.robot_nq = pin_robot_model.nq
        self.viz = MeshcatVisualizer(self.model, collision_model, visual_model)
        try:
            self.viz.initViewer(open=True)
            self.viz.viewer.wait()
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)
        self.viz.loadViewerModel(rootNodeName=self.model.name)

        # animation settings
        self.anim = Animation(default_framerate=ctrl_freq / save_freq)
        self.frame_idx = 0              # index of frame being saved in animation
        self.save_freq = save_freq      # display every save_freq simulation steps

        self.visual_model = visual_model
        self.visual_data = visual_data
        self.collision_model = collision_model
        self.collision_data = collision_data

    def add_robot(self, robot_name, pin_rob_model, collision_model, visual_model,
                  rob_position=None, rob_quaternion=None):
        if rob_position is None:
            rob_position = np.array([0., 0., 0.])
        if rob_quaternion is None:
            rob_quaternion = np.array([0, 0., 0., 1.])   # xyzw
        viz = MeshcatVisualizer(pin_rob_model, collision_model, visual_model)
        viz.initViewer(self.viz.viewer)
        viz.loadViewerModel(rootNodeName=robot_name)

        rob_quaternion = rob_quaternion[[3, 0, 1, 2]]    # shift to wxyz used by tf
        tf_transl = tf.translation_matrix(rob_position)
        tf_rot = tf.quaternion_matrix(rob_quaternion)
        tf_pose = tf.concatenate_matrices(tf_transl, tf_rot)
        viz.viewer[robot_name].set_transform(tf_pose)

    def add_arrow(self, obj_name, color=[1, 0, 0], height=0.1):
        arrow_shaft = g.Cylinder(height, 0.01)
        arrow_head = g.Cylinder(0.04, 0.04, radiusTop=0.001, radiusBottom=0.04)
        material = g.MeshPhongMaterial()
        material.color = int(color[0] * 255) * 256 ** 2 + int(
            color[1] * 255) * 256 + int(color[2] * 255)

        arrow_offset = tf.translation_matrix([0., height/2., 0.])
        shaft_rotation = tf.rotation_matrix(np.pi/2., [1., 0., 0.])
        # arrow_vertical = tf.concatenate_matrices(arrow_offset, shaft_rotation)
        self.viz.viewer[obj_name]["arrow"].set_object(arrow_shaft, material)
        self.viz.viewer[obj_name]["arrow"].set_transform(shaft_rotation)
        self.viz.viewer[obj_name]["arrow/head"].set_object(arrow_head, material)
        self.viz.viewer[obj_name]["arrow/head"].set_transform(arrow_offset)

    def add_shapes_from(self, shapes_lst: List[HPolyhedron]):
        for i, shape in enumerate(shapes_lst):
            name = 'env/' + str(i)
            meshcat_geom = pydrake_geom_to_meshcat(shape)
            if i == 0:
                self.add_shape(name, meshcat_geom, meshcat_obstacle_obj())
            if i == 1 or i == 2:
                self.add_shape(name, meshcat_geom, meshcat_obstacle_obj(GREY, 0.5))
            elif i == 3:
                self.add_shape(name, meshcat_geom, meshcat_obstacle_obj(BLUE, 0.7))
            elif i == 4:
                self.add_shape(name, meshcat_geom, meshcat_obstacle_obj(GREEN, 0.7))
            elif i == 5:
                self.add_shape(name, meshcat_geom, meshcat_obstacle_obj(PURPLE, 0.7))

    def add_shape(self, viewer_name, meshcat_shape, obj_material=None):
        if obj_material is None:
            obj_material = meshcat_iris_obj()
        self.viz.viewer[viewer_name].set_object(meshcat_shape, obj_material)

    def displayForcesFromCrocoddylSolver(self, fs_ti, frame):
        for contact in range(len(fs_ti)):
            pos = fs_ti[contact]['oMf'].translation
            ori = fs_ti[contact]['oMf'].rotation
            force_dir = fs_ti[contact]['w_f'].linear

            scale = np.linalg.norm(force_dir) / 100.
            grf_tf = get_scaled_and_oriented_grf_tf(scale, pos, ori, force_dir)

            link_name = self.model.names[int(fs_ti[contact]['key'])]
            frame['forces'][link_name].set_transform(grf_tf)
            self.viz.viewer['forces'][link_name].set_transform(grf_tf)

    def displayFromCrocoddylSolver(self, solver):
        for it in solver:
            models = it.problem.runningModels.tolist() + [it.problem.terminalModel]
            dts = [m.dt if hasattr(m, "differential") else 0. for m in models]

            fs = get_force_trajectory_from_solver(it)

            for sim_time_idx in np.arange(0, len(fs), self.save_freq):
                q = np.array(it.xs[int(sim_time_idx)][:self.robot_nq])
                self.viz.display(q)

                fs_ti = fs[sim_time_idx]

                with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
                    self.display_visualizer_frames(frame, q)
                    self.display_collisions(frame, q)
                    self.displayForcesFromCrocoddylSolver(fs_ti, frame)

                self.frame_idx += 1     # increase frame index counter

        # save animation
        self.viz.viewer.set_animation(self.anim, play=False)

    def start_animation(self):
        self.frame_idx = 0

    def finish_animation(self):
        # save animation
        self.viz.viewer.set_animation(self.anim, play=False)

    def animate_single_collision(self, collision_name: str, collision_target: np.array):
        with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
            self.display_single_collision(frame, collision_target, collision_name)

    def animate_single_shape(self, viewer_name: str,
                             collision_target: np.ndarray):
        with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
            self.display_single_shape(frame, viewer_name, collision_target)

    def animate_target(self, end_effector_name, targets, color=None):
        with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
            self.display_targets(end_effector_name, targets, color, animation=True)

    def animate_frame(self, q):
        with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
            self.display_visualizer_frames(frame, q)

    def animate_frame_with_collisions(self, q):
        with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
            self.display_visualizer_frames(frame, q)
            self.display_collisions(frame, q)

    def animation_step(self):
        self.frame_idx += 1

    def display_visualizer_frames(self, frame, q):
        meshcat_visualizer = self.viz

        geom_model = self.visual_model
        geom_data = self.visual_data

        pin.forwardKinematics(self.model, self.robot_data, q)
        pin.updateGeometryPlacements(self.model, self.robot_data,
                                     geom_model, geom_data)
        for visual in geom_model.geometryObjects:
            viewer_name = meshcat_visualizer.getViewerNodeName(visual, pin.GeometryType.VISUAL)
            # Get mesh pose.
            M = geom_data.oMg[geom_model.getGeometryId(visual.name)]
            # Manage scaling
            if hasMeshFileInfo(visual):
                scale = np.asarray(visual.meshScale).flatten()
                S = np.diag(np.concatenate((scale, [1.0])))
                # S = visual.placement.homogeneous
                T = np.array(M.homogeneous).dot(S)
            else:
                T = M.homogeneous
            # Update viewer configuration.
            frame[viewer_name].set_transform(T)

    def display_collisions(self, frame, q):
        meshcat_visualizer = self.viz

        geom_model = self.collision_model
        geom_data = self.collision_data

        pin.forwardKinematics(self.model, self.robot_data, q)
        pin.updateGeometryPlacements(self.model, self.robot_data,
                                     geom_model, geom_data)
        for visual in geom_model.geometryObjects:
            viewer_name = meshcat_visualizer.getViewerNodeName(visual, pin.GeometryType.COLLISION)
            # Get mesh pose.
            M = geom_data.oMg[geom_model.getGeometryId(visual.name)]
            # Manage scaling
            if hasMeshFileInfo(visual):
                scale = np.asarray(visual.meshScale).flatten()
                S = np.diag(np.concatenate((scale, [1.0])))
                # S = visual.placement.homogeneous
                T = np.array(M.homogeneous).dot(S)
            else:
                T = M.homogeneous
            # Update viewer configuration.
            frame[viewer_name].set_transform(T)

    def display_single_collision(self, frame, target, base_name):
        meshcat_visualizer = self.viz

        geom_model = self.collision_model
        for visual in geom_model.geometryObjects:
            if visual.name == base_name:
                viewer_name = meshcat_visualizer.getViewerNodeName(visual, pin.GeometryType.COLLISION)
                T = np.array([
                    [1.0, 0.0, 0.0, target[0]],
                    [0.0, 1.0, 0.0, target[1]],
                    [0.0, 0.0, 1.0, target[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ])
                frame[viewer_name].set_transform(T)
                return

    def display_single_shape(self,
                             frame,
                             viewer_name,
                             target_pos):
        frame[viewer_name].set_transform(target_pos)

    def display_targets(self, end_effector_name, targets, color=None, animation=False):
        if color is None:
            color = [1, 0, 0]
        material = g.MeshPhongMaterial()
        material.color = int(color[0] * 255) * 256 ** 2 + int(
            color[1] * 255) * 256 + int(color[2] * 255)
        material.opacity = 0.4
        for i, target in enumerate(targets):
            self.viz.viewer[end_effector_name + "/" + str(i)].set_object(g.Sphere(0.01),  material)
            Href = np.array(
                [
                    [1.0, 0.0, 0.0, target[0]],
                    [0.0, 1.0, 0.0, target[1]],
                    [0.0, 0.0, 1.0, target[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            if animation:
                self.anim.at_frame(self.viz.viewer, self.frame_idx)[end_effector_name + "/" + str(i)].set_transform(Href)
            else:
                self.viz.viewer[end_effector_name+"/" + str(i)].set_transform(Href)

    def hide_visuals(self, viz_list, b_visualize=False):
        for viz in viz_list:
            self.viz.viewer[viz].set_property("visible", b_visualize)

    def save_html(self, path, filename):
        viewer_html = self.viz.viewer.static_html()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + filename, "w") as f:
            f.write(viewer_html)


def coal_geom_to_meshcat(geom: pin.GeometryObject):
    """Convert a pinocchio collision geometry to a meshcat geometry."""
    if isinstance(geom, coal.Sphere):
        radius = geom.radius
        sphere = g.Sphere(radius=radius)
        return sphere
    elif isinstance(geom, coal.Cylinder):
        # Note: Meshcat Cylinders are aligned with the y-axis while
        # Pinocchio Cylinders are aligned with the z-axis.
        return g.Cylinder(
            radiusTop=geom.radius,
            radiusBottom=geom.radius,
            height=2*geom.halfLength,
        )
    elif isinstance(geom, coal.Box):
        return g.Box(2*geom.halfSide)
    else:
        raise NotImplementedError(
            f"Geometry type {geom.__class__} conversion to Meshcat not defined.")