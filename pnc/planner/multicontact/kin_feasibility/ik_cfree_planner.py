import time
from typing import List, OrderedDict

import meshcat
from pinocchio.visualize import MeshcatVisualizer
# kinematics tools
import qpsolvers
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, JointCouplingTask, PostureTask

import numpy as np

from pnc.planner.multicontact.fastpathplanning.smooth import CompositeBezierCurve
from util import util
# Planner
from pnc.planner.multicontact.kin_feasibility.locomanipulation_frame_planner import LocomanipulationFramePlanner


def display_visualizer_frames(meshcat_visualizer, frame):
    for visual in meshcat_visualizer.visual_model.geometryObjects:
        # Get mesh pose.
        M = meshcat_visualizer.visual_data.oMg[
            meshcat_visualizer.visual_model.getGeometryId(visual.name)]
        # Manage scaling
        scale = np.asarray(visual.meshScale).flatten()
        S = np.diag(np.concatenate((scale, [1.0])))
        T = np.array(M.homogeneous).dot(S)
        # Update viewer configuration.
        frame[meshcat_visualizer.getViewerNodeName(
            visual, pin.GeometryType.VISUAL)].set_transform(T)

def set_desired_frame_task(task: pink.FrameTask,
                           quat: np.array(4),
                           pos: np.array(3)):
    task.set_target(pin.SE3(util.quat_to_rot(quat), pos))


def set_desired_posture_task(task: pink.PostureTask,
                             q_nominal: np.array):
    task.set_target(q_nominal)


class IKCFreePlanner:
    def __init__(self, pin_robot_model: pin.Model,
                 pin_robot_data: pin.Data,
                 plan_frames_to_model_map: dict[str: str],
                 q0: np.array = None,
                 dt: float = 0.02):
        self.dt = dt
        self.task_dict = {}             # filled out in PInk tasks (setup_tasks)
        self.planner = None
        self._b_record_anim = False

        if q0 is None:
            q0 = np.zeros(pin_robot_model.nq)

        # PInK robot data configuration
        self.pink_config = pink.Configuration(pin_robot_model, pin_robot_data, q0)

        # Compute initial end-effector positions and orientations
        self.frames_pos, self.frames_quat = {}, {}
        for p_frame, m_frame in plan_frames_to_model_map.items():
            self.frames_pos[p_frame] = self.pink_config.get_transform_frame_to_world(m_frame).translation
            self.frames_quat[p_frame] = util.rot_to_quat(self.pink_config.get_transform_frame_to_world(m_frame).rotation)

        # PInK tasks
        self.tasks = self._initialize_tasks()

        # Select quadprog solver, if available
        self.solver = qpsolvers.available_solvers[0]
        if "quadprog" in qpsolvers.available_solvers:
            self.solver = "quadprog"

    def _initialize_tasks(self) -> List[pink.Task]:
        torso_task = FrameTask(
            "torso_link",
            position_cost=0.001,
            orientation_cost=0.005,
        )
        left_foot_task = FrameTask(
            "left_ankle_roll_link",     #"l_foot_contact",
            position_cost=1.0,
            orientation_cost=0.05,
        )
        right_foot_task = FrameTask(
            "right_ankle_roll_link",     #"r_foot_contact",
            position_cost=5.0,
            orientation_cost=0.05,
        )
        left_knee_task = FrameTask(
            "left_knee_link",     #"l_knee_fe_ld
            position_cost=0.05,
            orientation_cost=0.001,
        )
        right_knee_task = FrameTask(
            "right_knee_link",          #r_knee_fe_ld",
            position_cost=0.2,
            orientation_cost=0.001,
        )
        left_hand_task = FrameTask(
            "left_palm_link",       #"l_hand_contact",
            position_cost=5.0,
            orientation_cost=0.001,
            gain=0.1,
        )
        right_hand_task = FrameTask(
            "right_palm_link",       #"r_hand_contact",
            position_cost=0.01,
            orientation_cost=0.0001,
            gain=0.1,
        )
        posture_task = PostureTask(
            cost=1e-3,  # [cost] / [rad]
        )
        # ----- Joint coupling task
        # r_knee_holonomic_task = JointCouplingTask(
        #     ["r_knee_fe_jp", "r_knee_fe_jd"],
        #     [1.0, -1.0],
        #     10.0,
        #     self.pink_config,
        #     lm_damping=5e-7,
        # )
        # r_knee_holonomic_task.gain = 0.05
        # l_knee_holonomic_task = JointCouplingTask(
        #     ["l_knee_fe_jp", "l_knee_fe_jd"],
        #     [1.0, -1.0],
        #     10.0,
        #     self.pink_config,
        #     lm_damping=5e-7,
        # )
        # l_knee_holonomic_task.gain = 0.05
        self.task_dict = {'torso_task': torso_task,
                          'lfoot_task': left_foot_task,
                          'rfoot_task': right_foot_task,
                          'lknee_task': left_knee_task,
                          'rknee_task': right_knee_task,
                          'lhand_task': left_hand_task,
                          'rhand_task': right_hand_task,}
                          # 'posture_task': posture_task,
                          # 'lknee_constr_task': l_knee_holonomic_task,
                          # 'rknee_constr_task': r_knee_holonomic_task}

        return [torso_task, left_foot_task, right_foot_task, left_knee_task, right_knee_task,
                left_hand_task, right_hand_task] #, posture_task,
                # l_knee_holonomic_task, r_knee_holonomic_task]

    def set_planner(self, planner: LocomanipulationFramePlanner):
        self.planner = planner

    def plan(self, p_init: np.array,
             T: float,
             alpha: np.array,
             w_rigid: np.array,
             visualizer: MeshcatVisualizer = None,
             verbose: bool = False):
        if self.planner is None:
            raise ValueError("Planner not set")

        # compute plan
        ik_all_start_time = time.time()
        self.planner.plan_iris(p_init, T, alpha, w_rigid, verbose)
        print("[Compute Time] Total IK solve time: ", time.time() - ik_all_start_time)
        if visualizer is not None:
            self.planner.plot(visualizer)

        # get some information from planner
        frame_names = self.planner.frame_names      # in the order matching path solution
        bez_paths = self.planner.path

        # record video
        if self._b_record_anim:
            anim = meshcat.animation.Animation()
            anim.default_framerate = int(1 / self.dt)

        # evaluate end-effector path at each time step
        # seg_memory = np.zeros(len(bez_paths), dtype=int)
        # t, frame_idx = 0., int(0)
        # while t < bez_paths[0].b:           # length should be: len(bez_paths)
        #     curr_ee_point = []
        #     # determine current ee position for all frames
        #     for i_f, fr_path in enumerate(bez_paths):
        #         # get current segment from overall plan
        #         if t > fr_path.beziers[seg_memory[i_f]].b:
        #             seg_memory[i_f] += 1
        #         bezier_curve = fr_path.beziers[seg_memory[i_f]]
        #         curr_ee_point.append(bezier_curve(t))
        #
        #     # set desired tasks at current time
        #     set_desired_frame_task(self.task_dict['torso_task'], self.frames_quat['torso'], curr_ee_point[0])
        #     set_desired_frame_task(self.task_dict['lfoot_task'], self.frames_quat['LF'], curr_ee_point[1])
        #     set_desired_frame_task(self.task_dict['rfoot_task'], self.frames_quat['RF'], curr_ee_point[2])
        #     set_desired_frame_task(self.task_dict['lknee_task'], self.frames_quat['L_knee'], curr_ee_point[3])
        #     set_desired_frame_task(self.task_dict['rknee_task'], self.frames_quat['R_knee'], curr_ee_point[4])
        #     set_desired_frame_task(self.task_dict['lhand_task'], self.frames_quat['LH'], curr_ee_point[5])
        #     set_desired_frame_task(self.task_dict['rhand_task'], self.frames_quat['RH'], curr_ee_point[6])
        #
        #     # solve IK
        #     self.solve_ik()
        #
        #     # visualize / record
        #     if visualizer is not None:
        #         visualizer.display(self.pink_config.q)
        #
        #         torso_frame = meshcat.geometry.triad(0.2)
        #         des_torso_frame = meshcat.geometry.triad(0.2)
        #         visualizer.viewer["frames/torso"].set_object(torso_frame)
        #         visualizer.viewer["frames/torso"].set_transform(self.pink_config.get_transform_frame_to_world("torso_link").homogeneous)    #torso_link
        #         visualizer.viewer["frames/torso_d"].set_object(des_torso_frame)
        #         visualizer.viewer["frames/torso_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[0]))
        #
        #         lf_frame = meshcat.geometry.triad(0.2)
        #         des_lf_frame = meshcat.geometry.triad(0.2)
        #         visualizer.viewer["frames/LF"].set_object(lf_frame)
        #         visualizer.viewer["frames/LF"].set_transform(self.pink_config.get_transform_frame_to_world("left_ankle_roll_link").homogeneous) #l_foot_contact
        #         visualizer.viewer["frames/LF_d"].set_object(des_lf_frame)
        #         visualizer.viewer["frames/LF_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[1]))
        #
        #         rf_frame = meshcat.geometry.triad(0.2)
        #         des_rf_frame = meshcat.geometry.triad(0.2)
        #         visualizer.viewer["frames/RF"].set_object(rf_frame)
        #         visualizer.viewer["frames/RF"].set_transform(self.pink_config.get_transform_frame_to_world("right_ankle_roll_link").homogeneous)    #r_foot_contact
        #         visualizer.viewer["frames/RF_d"].set_object(des_rf_frame)
        #         visualizer.viewer["frames/RF_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[2]))
        #
        #         lk_frame = meshcat.geometry.triad(0.2)
        #         des_lk_frame = meshcat.geometry.triad(0.2)
        #         visualizer.viewer["frames/LK"].set_object(lk_frame)
        #         visualizer.viewer["frames/LK"].set_transform(self.pink_config.get_transform_frame_to_world("left_knee_link").homogeneous)
        #         visualizer.viewer["frames/LK_d"].set_object(des_lk_frame)
        #         visualizer.viewer["frames/LK_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[3]))
        #
        #         rk_frame = meshcat.geometry.triad(0.2)
        #         des_rk_frame = meshcat.geometry.triad(0.2)
        #         visualizer.viewer["frames/RK"].set_object(rk_frame)
        #         visualizer.viewer["frames/RK"].set_transform(self.pink_config.get_transform_frame_to_world("right_knee_link").homogeneous)
        #         visualizer.viewer["frames/RK_d"].set_object(des_rk_frame)
        #         visualizer.viewer["frames/RK_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[4]))
        #
        #         lh_frame = meshcat.geometry.triad(0.2)
        #         des_lh_frame = meshcat.geometry.triad(0.2)
        #         visualizer.viewer["frames/LH"].set_object(lh_frame)
        #         visualizer.viewer["frames/LH"].set_transform(self.pink_config.get_transform_frame_to_world("left_palm_link").homogeneous)
        #         visualizer.viewer["frames/LH_d"].set_object(des_lh_frame)
        #         T_lh = meshcat.transformations.translation_matrix(curr_ee_point[5])
        #         T_lh[:3, :3] = util.quat_to_rot(self.frames_quat['LH'])
        #         visualizer.viewer["frames/LH_d"].set_transform(T_lh)
        #
        #         rh_frame = meshcat.geometry.triad(0.2)
        #         des_rh_frame = meshcat.geometry.triad(0.2)
        #         visualizer.viewer["frames/RH"].set_object(rh_frame)
        #         visualizer.viewer["frames/RH"].set_transform(self.pink_config.get_transform_frame_to_world("right_palm_link").homogeneous)
        #         visualizer.viewer["frames/RH_d"].set_object(des_rh_frame)
        #         T_rh = meshcat.transformations.translation_matrix(curr_ee_point[6])
        #         T_rh[:3, :3] = util.quat_to_rot(self.frames_quat['RH'])
        #         visualizer.viewer["frames/RH_d"].set_transform(T_rh)
        #
        #         # record video
        #         with anim.at_frame(visualizer.viewer, frame_idx) as frame:
        #             display_visualizer_frames(visualizer, frame)
        #             frame["frames/torso"].set_transform(self.pink_config.get_transform_frame_to_world("torso_link").homogeneous)
        #             frame["frames/LH"].set_transform(self.pink_config.get_transform_frame_to_world("left_palm_link").homogeneous)
        #             frame["frames/RH"].set_transform(self.pink_config.get_transform_frame_to_world("right_palm_link").homogeneous)
        #             frame["frames/LF"].set_transform(self.pink_config.get_transform_frame_to_world("left_ankle_roll_link").homogeneous)
        #             frame["frames/RF"].set_transform(self.pink_config.get_transform_frame_to_world("right_ankle_roll_link").homogeneous)
        #             frame["frames/LK"].set_transform(self.pink_config.get_transform_frame_to_world("left_knee_link").homogeneous)
        #             frame["frames/RK"].set_transform(self.pink_config.get_transform_frame_to_world("right_knee_link").homogeneous)
        #             frame["frames/torso_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[0]))
        #             frame["frames/LH_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[5]))
        #             frame["frames/RH_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[6]))
        #             frame["frames/LF_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[1]))
        #             frame["frames/RF_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[2]))
        #             frame["frames/LK_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[3]))
        #             frame["frames/RK_d"].set_transform(meshcat.transformations.translation_matrix(curr_ee_point[4]))
        #         frame_idx += 1
        #
        #     t += self.dt

        # save video
        if self._b_record_anim:
            visualizer.viewer.set_animation(anim, play=False)

    def solve_ik(self):
        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(self.pink_config, self.tasks, self.dt, solver=self.solver)
        self.pink_config.integrate_inplace(velocity, self.dt)

    @staticmethod
    def _get_bez_segment(frame_bez_paths: CompositeBezierCurve,
                        t: float) -> int:
        seg = 0
        for s in frame_bez_paths.beziers:
            if t > s.b:
                seg += 1
            else:
                break
        return seg

    def get_ee_des_pos(self, frame_name_idx: int, t: float):
        frame_bez_path = self.planner.path[frame_name_idx]
        seg = self._get_bez_segment(frame_bez_path, t)
        if seg >= len(frame_bez_path.beziers):
            print(f'Segment {seg} was out of bounds for frame {frame_name_idx} at time {t}.')
            seg = len(frame_bez_path.beziers) - 1
        bezier_curve = frame_bez_path.beziers[seg]
        return bezier_curve(t)