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
    def __init__(self, pin_robot: pin.RobotWrapper,
                 plan_frames_to_model_map: dict[str: str],
                 q0: np.array = None,
                 dt: float = 0.005):
        self.pin_robot = pin_robot
        self.dt = dt
        self.task_dict = {}             # filled out in PInk tasks (setup_tasks)
        self.planner = None

        if q0 is None:
            q0 = pin_robot.q0.copy()

        # PInK robot data configuration
        self.pink_config = pink.Configuration(pin_robot.model, pin_robot.data, q0)

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
            position_cost=0.01,
            orientation_cost=0.02,
        )
        left_foot_task = FrameTask(
            "l_foot_contact",
            position_cost=5.0,
            orientation_cost=0.5,
        )
        right_foot_task = FrameTask(
            "r_foot_contact",
            position_cost=5.0,
            orientation_cost=0.5,
        )
        left_knee_task = FrameTask(
            "l_knee_fe_ld",
            position_cost=2.0,
            orientation_cost=0.01,
        )
        right_knee_task = FrameTask(
            "r_knee_fe_ld",
            position_cost=2.0,
            orientation_cost=0.01,
        )
        left_hand_task = FrameTask(
            "l_hand_contact",
            position_cost=5.0,
            orientation_cost=0.1,
            gain=0.1,
        )
        right_hand_task = FrameTask(
            "r_hand_contact",
            position_cost=5.0,
            orientation_cost=0.1,
            gain=0.1,
        )
        posture_task = PostureTask(
            cost=1e-3,  # [cost] / [rad]
        )
        # ----- Joint coupling task
        r_knee_holonomic_task = JointCouplingTask(
            ["r_knee_fe_jp", "r_knee_fe_jd"],
            [1.0, -1.0],
            100.0,
            self.pink_config,
            lm_damping=5e-7,
        )
        r_knee_holonomic_task.gain = 0.05
        l_knee_holonomic_task = JointCouplingTask(
            ["l_knee_fe_jp", "l_knee_fe_jd"],
            [1.0, -1.0],
            100.0,
            self.pink_config,
            lm_damping=5e-7,
        )
        l_knee_holonomic_task.gain = 0.05
        self.task_dict = {'torso_task': torso_task,
                          'lfoot_task': left_foot_task,
                          'rfoot_task': right_foot_task,
                          'lknee_task': left_knee_task,
                          'rknee_task': right_knee_task,
                          'lhand_task': left_hand_task,
                          'rhand_task': right_hand_task,
                          # 'posture_task': posture_task,
                          'lknee_constr_task': l_knee_holonomic_task,
                          'rknee_constr_task': r_knee_holonomic_task}

        return [torso_task, left_foot_task, right_foot_task, left_knee_task, right_knee_task,
                left_hand_task, right_hand_task, #posture_task,
                l_knee_holonomic_task, r_knee_holonomic_task]

    def set_planner(self, planner: LocomanipulationFramePlanner):
        self.planner = planner

    def plan(self, p_init: np.array,
             T: float,
             alpha: np.array,
             visualizer: MeshcatVisualizer = None):
        if self.planner is None:
            raise ValueError("Planner not set")

        # compute plan
        self.planner.plan_iris(p_init, T, alpha)
        self.planner.plot(visualizer)

        # get some information from planner
        frame_names = self.planner.frame_names      # in the order matching path solution
        bez_paths = self.planner.path

        # record video
        anim = meshcat.animation.Animation()
        anim.default_framerate = int(1 / self.dt)

        # evaluate end-effector path at each time step
        t, frame_idx = 0., int(0)
        while t < (5.25):           # length should be: len(bez_paths)
            curr_ee_point = []
            # determine current ee position for all frames
            for fr_path in bez_paths:
                # get current segment from overall plan
                seg = int(t // T)
                bezier_curve = fr_path.beziers[seg]
                curr_ee_point.append(bezier_curve(t))

            # set desired tasks at current time
            set_desired_frame_task(self.task_dict['torso_task'], self.frames_quat['torso'], curr_ee_point[0])
            set_desired_frame_task(self.task_dict['lfoot_task'], self.frames_quat['LF'], curr_ee_point[1])
            set_desired_frame_task(self.task_dict['rfoot_task'], self.frames_quat['RF'], curr_ee_point[2])
            set_desired_frame_task(self.task_dict['lknee_task'], self.frames_quat['L_knee'], curr_ee_point[3])
            set_desired_frame_task(self.task_dict['rknee_task'], self.frames_quat['R_knee'], curr_ee_point[4])
            set_desired_frame_task(self.task_dict['lhand_task'], self.frames_quat['LH'], curr_ee_point[5])
            set_desired_frame_task(self.task_dict['rhand_task'], self.frames_quat['RH'], curr_ee_point[6])

            # solve IK
            self.solve_ik()

            # visualize / record
            if visualizer is not None:
                visualizer.display(self.pink_config.q)

                # record video
                with anim.at_frame(visualizer.viewer, frame_idx) as frame:
                    display_visualizer_frames(visualizer, frame)
                frame_idx += 1

            t += self.dt

        # save video
        if visualizer is not None:
            visualizer.viewer.set_animation(anim, play=False)

    def solve_ik(self):
        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(self.pink_config, self.tasks, self.dt, solver=self.solver)
        self.pink_config.integrate_inplace(velocity, self.dt)
