import numpy as np
import pinocchio
from pink import solve_ik, Configuration, PostureTask, FrameTask
import qpsolvers

class G1IKSolver:
    def __init__(self, robot_model: pinocchio.Model,
                 robot_data: pinocchio.Data,
                 q0: np.array,
                 b_use_knees: bool = True):
        # PInK robot data configuration
        self.pink_config = Configuration(robot_model, robot_data, q0)

        # Compute initial end-effector positions and orientations
        # self.frames_pos, self.frames_quat = {}, {}
        # for p_frame, m_frame in plan_frames_to_model_map.items():
        #     self.frames_pos[p_frame] = self.pink_config.get_transform_frame_to_world(m_frame).translation
        #     self.frames_quat[p_frame] = util.rot_to_quat(
        #         self.pink_config.get_transform_frame_to_world(m_frame).rotation)

        # PInK tasks
        self.tasks_dict = self._initialize_tasks(b_use_knees)

        # Select quadprog solver, if available
        self.solver = qpsolvers.available_solvers[0]
        if "quadprog" in qpsolvers.available_solvers:
            self.solver = "quadprog"

    def _initialize_tasks(self, b_use_knees:bool = True) -> dict[str, PostureTask | FrameTask]:
            torso_task = FrameTask(
                "torso_primitive_shape",    # torso_link
                position_cost=1.0,
                orientation_cost=0.5,
            )
            left_foot_task = FrameTask(
                "left_ankle_roll_link",  # "l_foot_contact",
                position_cost=1.0,
                orientation_cost=0.2,
            )
            right_foot_task = FrameTask(
                "right_ankle_roll_link",  # "r_foot_contact",
                position_cost=1.0,
                orientation_cost=0.2,
            )
            left_hand_task = FrameTask(
                "left_rubber_hand",  # "l_hand_contact",
                position_cost=1.0,
                orientation_cost=0.00001,
                gain=0.1,
            )
            right_hand_task = FrameTask(
                "right_rubber_hand",  # "r_hand_contact",
                position_cost=1.0,
                orientation_cost=0.00001,
                gain=0.1,
            )
            posture_task = PostureTask(
                cost=1e-3,  # [cost] / [rad]
            )
            task_dict = {
                'posture_task': posture_task,
                'torso_task': torso_task,
                'LF_task': left_foot_task,
                'RF_task': right_foot_task,
                'LH_task': left_hand_task,
                'RH_task': right_hand_task,
            }
            if b_use_knees:
                left_knee_task = FrameTask(
                    "left_knee_link",  # "l_knee_fe_ld
                    position_cost=0.2,
                    orientation_cost=0.00001,
                )
                right_knee_task = FrameTask(
                    "right_knee_link",  # r_knee_fe_ld",
                    position_cost=0.2,
                    orientation_cost=0.00001,
                )
                task_dict['L_knee_task'] = left_knee_task
                task_dict['R_knee_task'] = right_knee_task
            # return [torso_task, left_foot_task, right_foot_task, left_knee_task, right_knee_task,
            #         left_hand_task, right_hand_task]
            return task_dict


    def solve(self, targets_dict: dict[str: np.array], q_target) -> np.array:
        # set desired targets
        for task_name, task in self.tasks_dict.items():
            if type(task) is FrameTask:
                target_pose = pinocchio.SE3().Identity()
                frame_name = task_name.split('_task')[0]
                target_pose.translation = targets_dict[frame_name]
                task.set_target(target_pose)
            elif type(task) is PostureTask:
                task.set_target(q_target)
            else:
                raise NotImplementedError

        # integrate until we reach a steady pose
        dt = 5e-3  # [s]
        for t in np.arange(0.0, 3.0, dt):
            velocity = solve_ik(self.pink_config, self.tasks_dict.values(), dt, solver=self.solver)
            if np.linalg.norm(velocity) < 1e-1:
                break
            self.pink_config.integrate_inplace(velocity, dt)
        return self.pink_config.q
