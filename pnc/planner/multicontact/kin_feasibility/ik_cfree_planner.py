import time
from typing import List

import meshcat
from pinocchio.visualize import MeshcatVisualizer
# kinematics tools
import pinocchio as pin
import numpy as np

from config.multicontact.planner_config import PlannerConfig
from util.path_parameterization import CompositeBezierCurve, get_bez_segment, get_frame_des_pos
from util import util
from .baseline_frame_planner import BaselineFramePlanner
# Planner
from .locomanipulation_frame_planner import LocomanipulationFramePlanner

b_use_ik_solver = False

if b_use_ik_solver:
    import qpsolvers
    import pink
    from pink import solve_ik
    from pink.tasks import FrameTask, PostureTask

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

if b_use_ik_solver:
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
                 gains: PlannerConfig = None,
                 dt: float = 0.02):
        self.dt = dt
        self.task_dict = {}             # filled out in PInk tasks (setup_tasks)
        self.solver_stats = {}
        self.planner = None
        self.plan_to_model_frames = None
        self._b_record_anim = False
        self.w_rigid_poly = np.array(gains.W_RIGID_LINK)

        if q0 is None and pin_robot_model is not None:
            q0 = np.zeros(pin_robot_model.nq)

        if b_use_ik_solver:
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

    if b_use_ik_solver:
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

    def set_planner(self, planner: LocomanipulationFramePlanner | BaselineFramePlanner):
        self.planner = planner

    def plan(self, p_init: np.array,
             T: float,
             planner_params: PlannerConfig,
             visualizer: MeshcatVisualizer = None,
             verbose: bool = False,
             save_html:bool = False):
        if self.planner is None:
            raise ValueError("Planner not set")

        # compute plan
        alpha = planner_params.ALPHA
        w_rigid = np.array(planner_params.W_RIGID_LINK)
        ik_all_start_time = time.time()
        self.planner.plan_iris(p_init, T, alpha, w_rigid, self.w_rigid_poly, verbose)
        self.solver_stats = self.planner.solver_stats
        self.solver_stats['ik_plan_total_time'] = time.time() - ik_all_start_time
        if verbose:
            print("[Compute Time] Total IK solve time: ", self.solver_stats['plan_iris_time'])
        if visualizer is not None:
            self.planner.plot(visualizer, save_html)

        # get some information from planner
        frame_names = self.planner.frame_names      # in the order matching path solution
        bez_paths = self.planner.path

        # record video
        if self._b_record_anim:
            anim = meshcat.animation.Animation()
            anim.default_framerate = int(1 / self.dt)

        # save video
        if self._b_record_anim:
            visualizer.viewer.set_animation(anim, play=False)

    if b_use_ik_solver:
        def solve_ik(self):
            # Compute velocity and integrate it into next configuration
            velocity = solve_ik(self.pink_config, self.tasks, self.dt, solver=self.solver)
            self.pink_config.integrate_inplace(velocity, self.dt)

    def set_plan_to_model_frames(self, plan_to_model_frames: dict[str: str]):
        self.plan_to_model_frames = plan_to_model_frames

    def pack_current_targets(self, t):
        planner_path = self.planner.path
        idx_LF = list(self.plan_to_model_frames.keys()).index('LF')
        idx_L_knee = list(self.plan_to_model_frames.keys()).index('L_knee')
        idx_RF = list(self.plan_to_model_frames.keys()).index('RF')
        idx_R_knee = list(self.plan_to_model_frames.keys()).index('R_knee')
        idx_LH = list(self.plan_to_model_frames.keys()).index('LH')
        idx_RH = list(self.plan_to_model_frames.keys()).index('RH')
        idx_torso = list(self.plan_to_model_frames.keys()).index('torso')
        lfoot_t = get_frame_des_pos(planner_path[idx_LF], t)
        lknee_t = get_frame_des_pos(planner_path[idx_L_knee], t)
        rfoot_t = get_frame_des_pos(planner_path[idx_RF], t)
        rknee_t = get_frame_des_pos(planner_path[idx_R_knee], t)
        lhand_t = get_frame_des_pos(planner_path[idx_LH], t)
        rhand_t = get_frame_des_pos(planner_path[idx_RH], t)
        base_t = get_frame_des_pos(planner_path[idx_torso], t)
        frame_targets_dict = {
            'torso': base_t,
            'LF': lfoot_t,
            'RF': rfoot_t,
            'L_knee': lknee_t,
            'R_knee': rknee_t,
            'LH': lhand_t,
            'RH': rhand_t
        }
        return frame_targets_dict
