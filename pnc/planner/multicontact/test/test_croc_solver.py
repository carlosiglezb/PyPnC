import unittest
from collections import OrderedDict
import os, sys

# Visualization tools
import matplotlib.pyplot as plt
from plot.helper import plot_vector_traj, Fxyz_labels
from plot.multiontact_plotter import MulticontactPlotter
from pnc.planner.multicontact.dyn_feasibility.G1MulticontactPlanner import G1MulticontactPlanner
from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import ContactSequence
import plot.meshcat_utils as vis_tools
from pnc.planner.multicontact.kin_feasibility import IKCFreePlanner, BaselineFramePlanner, MotionFrameSequencer

cwd = os.getcwd()
sys.path.append(cwd)

import pinocchio as pin
import numpy as np
import config.multicontact.g1_planner_config as g1_params
import config.multicontact.g1_baseline_planner_config as g1_baseline_params

B_VISUALIZE = False
B_SHOW_GRF_PLOTS = False
B_SHOW_COST_PLOTS = True


class DummyIKPlanner:
    def __init__(self, starting_frame_pos):
        self.starting_frame_pos = starting_frame_pos
        self.planner = None

    def pack_current_targets(self, time: float) -> dict[str: np.ndarray]:
        # use a constant target for all frames
        frame_targets_dict = {}
        for fr_name in self.starting_frame_pos.keys():
            frame_targets_dict[fr_name] = self.starting_frame_pos[fr_name]

        return frame_targets_dict

def load_robot_model(package_dir, urdf_file):
    rob_model, col_model, vis_model = pin.buildModelsFromUrdf(urdf_file,
                                                              package_dir,
                                                              pin.JointModelFreeFlyer())
    rob_data, col_data, vis_data = pin.createDatas(rob_model, col_model, vis_model)

    return rob_model, col_model, vis_model, rob_data, col_data, vis_data


def get_g1_default_initial_pose(n_joints):
    q0 = np.zeros(n_joints, )
    q0[0] = -np.pi / 4  # left_hip_pitch_joint
    q0[3] = np.pi / 2   # left_knee_joint
    q0[4] = -np.pi / 6  # left_ankle_pitch_joint
    q0[6] = -np.pi / 6  # right_hip_pitch_joints
    q0[9] = np.pi / 3   # right_knee_joint
    q0[10] = -np.pi / 6 # right_ankle_pitch_joint
    q0[14] = np.pi / 2  # left_shoulder_roll_joint
    # q0[15] = np.pi / 2  # left_shoulder_yaw_joint
    # q0[16] = np.pi / 4   # left_elbow_joint
    # q0[17] = np.pi / 2   # left_wrist_roll_joint
    q0[21] = -np.pi / 6  # r_shoulder_aa
    floating_base = np.array([0., 0., 0.7, 0., 0., 0., 1.])
    return np.concatenate((floating_base, q0))


class TestG1Planner(unittest.TestCase):
    def setUp(self):
        plan_to_model_frames = OrderedDict()
        force_joint_frames = OrderedDict()

        plan_to_model_frames['torso'] = 'torso_primitive_shape'
        plan_to_model_frames['LF'] = 'left_ankle_roll_link'
        plan_to_model_frames['RF'] = 'right_ankle_roll_link'
        plan_to_model_frames['L_knee'] = 'left_knee_link'
        plan_to_model_frames['R_knee'] = 'right_knee_link'
        plan_to_model_frames['LH'] = 'left_rubber_hand'
        plan_to_model_frames['RH'] = 'right_rubber_hand'
        force_joint_frames['LF'] = "left_ankle_roll_joint"
        force_joint_frames['RF'] = "right_ankle_roll_joint"
        force_joint_frames['LH'] = "left_wrist_yaw_joint"
        force_joint_frames['RH'] = "right_wrist_yaw_joint"
        package_dir = cwd + "/robot_model/g1_description"
        # robot_urdf_file = package_dir + "/g1_29dof_lock_waist_modified.urdf"
        robot_urdf_file = package_dir + "/g1_29dof_simple_collisions.urdf"
        self.plan_to_model_frames = plan_to_model_frames
        self.force_joint_frames = force_joint_frames

        # load robot model and corresponding robot data
        rob_model, col_model, vis_model, rob_data, col_data, vis_data = load_robot_model(package_dir, robot_urdf_file)
        self.refined_geom_model = pin.buildGeomFromUrdf(rob_model,
                                           robot_urdf_file,
                                           pin.GeometryType.COLLISION)
        self.refined_geom_model.addAllCollisionPairs()

        self.rob_model = rob_model
        self.col_model = col_model
        self.vis_model = vis_model

        # Getting the frame ids
        plan_to_model_ids = {}
        plan_to_model_ids['RF'] = rob_model.getFrameId(plan_to_model_frames['RF'])
        plan_to_model_ids['LF'] = rob_model.getFrameId(plan_to_model_frames['LF'])
        plan_to_model_ids['R_knee'] = rob_model.getFrameId(plan_to_model_frames['R_knee'])
        plan_to_model_ids['L_knee'] = rob_model.getFrameId(plan_to_model_frames['L_knee'])
        plan_to_model_ids['LH'] = rob_model.getFrameId(plan_to_model_frames['LH'])
        plan_to_model_ids['RH'] = rob_model.getFrameId(plan_to_model_frames['RH'])
        plan_to_model_ids['torso'] = rob_model.getFrameId(plan_to_model_frames['torso'])
        self.plan_to_model_ids = plan_to_model_ids

        q0 = get_g1_default_initial_pose(rob_model.nq - 7)
        v0 = np.zeros(rob_model.nv)
        x0 = np.concatenate([q0, v0])
        self.x0 = x0

        # update model with initial configuration
        pin.forwardKinematics(rob_model, rob_data, q0)
        pin.updateFramePlacements(rob_model, rob_data)
        self.rob_data = rob_data
        self.col_data = col_data
        self.vis_data = vis_data

        # save the initial positions of the frames
        starting_frame_pos = {}
        for fr_name in ['torso', 'LF', 'RF', 'LH', 'RH', 'L_knee', 'R_knee']:
            starting_frame_pos[fr_name] = rob_data.oMf[plan_to_model_ids[fr_name]].translation
        self.starting_frame_pos = starting_frame_pos
        self.planner_params = g1_params.MultiContactDoorConfig()
        self.baseline_planner_params = g1_baseline_params.MultiContactDoorConfig()

    def test_lean_on_left_wall(self):
        planner_params = self.planner_params
        rob_model = self.rob_model
        force_joint_frames = self.force_joint_frames
        contact_seq_planes = [{'RF': np.array([0, 0, 1]),
                               'LH': np.array([0, -1 , 0])}
                              ]

        # create a very simple IK planner
        ik_cfree_planner = DummyIKPlanner(self.starting_frame_pos)

        N_horizon_lst = [100]
        T = 2
        contact_sequence = ContactSequence(contact_seq_planes, N_horizon_lst, T)
        robot_dyn_plan = G1MulticontactPlanner(rob_model, contact_sequence, ik_cfree_planner, planner_params, None)
        robot_dyn_plan.set_plan_to_model_params(self.plan_to_model_ids)
        robot_dyn_plan.set_initial_configuration(self.x0)
        robot_dyn_plan.set_zero_configuration(self.x0[:rob_model.nq])
        robot_dyn_plan.plan(False)
        self.assertEqual(True, True)

        if B_VISUALIZE:
            save_freq = 10
            display_idx = np.arange(0, len(robot_dyn_plan.lf_targets), save_freq)
            display = vis_tools.MeshcatPinocchioAnimation(rob_model, self.col_model, self.vis_model,
                                                          self.rob_data, self.vis_data, self.col_data,
                                                          ctrl_freq=np.average(N_horizon_lst) / T, save_freq=save_freq)
            display.display_targets("lfoot_target", robot_dyn_plan.lf_targets[display_idx], [1, 1, 0])
            display.display_targets("lknee_target", robot_dyn_plan.lkn_targets[display_idx], [0, 0, 1])
            display.display_targets("rfoot_target", robot_dyn_plan.rf_targets[display_idx], [1, 1, 0])
            display.display_targets("rknee_target", robot_dyn_plan.rkn_targets[display_idx], [0, 0, 1])
            display.display_targets("lhand_target", robot_dyn_plan.lh_targets[display_idx], [0.5, 0, 0])
            display.display_targets("rhand_target", robot_dyn_plan.rh_targets[display_idx], [0.5, 0, 0])
            display.display_targets("base_target", robot_dyn_plan.base_targets[display_idx], [0, 0.5, 0])
            display.add_arrow("forces/" + force_joint_frames['RF'], color=[0, 0, 1])
            display.add_arrow("forces/" + force_joint_frames['LH'], color=[0, 1, 0])
            display.displayFromCrocoddylSolver(robot_dyn_plan.fddp)


        if B_SHOW_GRF_PLOTS:
            # Note: contact_links are l_ankle_ie, r_ankle_ie, l_wrist_pitch, r_wrist_pitch
            sim_steps_list = [len(robot_dyn_plan.fddp[i].us) for i in range(len(robot_dyn_plan.fddp))]
            sim_steps = np.sum(sim_steps_list)
            sim_time = np.zeros((sim_steps,))
            rf_lfoot, rf_rfoot, rf_lwrist, rf_rwrist = np.zeros((3, sim_steps)), \
                np.zeros((3, sim_steps)), np.zeros((3, sim_steps)), np.zeros((3, sim_steps))
            w_rf_lfoot, w_rf_rfoot, w_rf_lwrist, w_rf_rwrist = np.zeros((3, sim_steps)), \
                np.zeros((3, sim_steps)), np.zeros((3, sim_steps)), np.zeros((3, sim_steps))
            time_idx = 0
            for it in robot_dyn_plan.fddp:
                rf_list = vis_tools.get_force_trajectory_from_solver(it)
                for rf_t in rf_list:
                    for contact in rf_t:
                        # determine contact link
                        cur_link = int(contact['key'])
                        if rob_model.names[cur_link] == force_joint_frames['LF']:
                            rf_lfoot[:, time_idx] = contact['f'].linear
                            w_rf_lfoot[:, time_idx] = contact['w_f'].linear
                        elif rob_model.names[cur_link] == force_joint_frames['RF']:
                            rf_rfoot[:, time_idx] = contact['f'].linear
                            w_rf_rfoot[:, time_idx] = contact['w_f'].linear
                        elif rob_model.names[cur_link] == force_joint_frames['LH']:
                            rf_lwrist[:, time_idx] = contact['f'].linear
                            w_rf_lwrist[:, time_idx] = contact['w_f'].linear
                        elif rob_model.names[cur_link] == force_joint_frames['RH']:
                            rf_rwrist[:, time_idx] = contact['f'].linear
                            w_rf_rwrist[:, time_idx] = contact['w_f'].linear
                        else:
                            print(f"ERROR: Non-specified contact {rob_model.names[cur_link]}")
                    dt = it.problem.runningModels[0].dt     # assumes constant dt over fddp sequence
                    if time_idx < len(sim_time) - 1:
                        sim_time[time_idx+1] = sim_time[time_idx] + dt
                        time_idx += 1
                    else:
                        continue

            if B_SHOW_GRF_PLOTS:
                plot_vector_traj(sim_time, w_rf_lfoot.T, 'RF LFoot (World)', Fxyz_labels)
                plot_vector_traj(sim_time, w_rf_rfoot.T, 'RF RFoot (World)', Fxyz_labels)
                plot_vector_traj(sim_time, w_rf_lwrist.T, 'RF LWrist (World)', Fxyz_labels)
                plot_vector_traj(sim_time, w_rf_rwrist.T, 'RF RWrist (World)', Fxyz_labels)
                plt.show()

        if B_SHOW_COST_PLOTS:
            plan_plotter = MulticontactPlotter(robot_dyn_plan)
            plan_plotter.plot_costs()
            plt.show()


    def test_lean_on_left_wall_high_knees(self):
        planner_params = self.baseline_planner_params
        rob_model = self.rob_model
        force_joint_frames = self.force_joint_frames
        contact_seq_planes = [{'RF': np.array([0, 0, 1]),
                               'LH': np.array([0, -1 , 0])}
                              ]

        # set motion
        fixed_frames_seq, motion_frames_seq = [], MotionFrameSequencer()
        fixed_frames_seq.append(['torso', 'RF', 'R_knee', 'LH', 'RH'])  # phase 0
        motion_frames_seq.add_motion_frame({'LF': self.starting_frame_pos['LF'] + np.array([0.0, 0.0, 0.38]),
                                            'L_knee': self.starting_frame_pos['L_knee'] + np.array([0.0, 0.0, 0.38])})

        # create a very baseline IK planner w/ constant targets
        ik_cfree_planner = IKCFreePlanner(rob_model, self.rob_data, self.plan_to_model_frames,
                                          self.x0[:self.rob_model.nq], self.planner_params)
        frame_planner = BaselineFramePlanner(self.rob_data, self.plan_to_model_ids,
                                             motion_frames_seq, fixed_frames_seq, "linear")
        ik_cfree_planner.set_planner(frame_planner)

        N_horizon_lst = [150]
        T = 3
        contact_sequence = ContactSequence(contact_seq_planes, N_horizon_lst, T)
        robot_dyn_plan = G1MulticontactPlanner(rob_model, contact_sequence, ik_cfree_planner, planner_params, self.refined_geom_model)
        robot_dyn_plan.set_plan_to_model_params(self.plan_to_model_ids)
        robot_dyn_plan.set_initial_configuration(self.x0)
        robot_dyn_plan.set_zero_configuration(self.x0[:rob_model.nq])
        robot_dyn_plan.plan(False, sca_refinement=True)
        self.assertEqual(True, True)

        if B_VISUALIZE:
            save_freq = 10
            display_idx = np.arange(0, len(robot_dyn_plan.lf_targets), save_freq)
            display = vis_tools.MeshcatPinocchioAnimation(rob_model, self.col_model, self.vis_model,
                                                          self.rob_data, self.vis_data, self.col_data,
                                                          ctrl_freq=np.average(N_horizon_lst) / T, save_freq=save_freq)
            display.display_targets("lfoot_target", robot_dyn_plan.lf_targets[display_idx], [1, 1, 0])
            display.display_targets("lknee_target", robot_dyn_plan.lkn_targets[display_idx], [0, 0, 1])
            display.display_targets("rfoot_target", robot_dyn_plan.rf_targets[display_idx], [1, 1, 0])
            display.display_targets("rknee_target", robot_dyn_plan.rkn_targets[display_idx], [0, 0, 1])
            display.display_targets("lhand_target", robot_dyn_plan.lh_targets[display_idx], [0.5, 0, 0])
            display.display_targets("rhand_target", robot_dyn_plan.rh_targets[display_idx], [0.5, 0, 0])
            display.display_targets("base_target", robot_dyn_plan.base_targets[display_idx], [0, 0.5, 0])
            display.add_arrow("forces/" + force_joint_frames['RF'], color=[0, 0, 1])
            display.add_arrow("forces/" + force_joint_frames['LH'], color=[0, 1, 0])
            fddp = robot_dyn_plan.get_latest_fddp()
            display.displayFromCrocoddylSolver([fddp])

        if B_SHOW_COST_PLOTS:
            plan_plotter = MulticontactPlotter(robot_dyn_plan)
            plan_plotter.plot_costs()
            plan_plotter.plot_constraint_violations()
            plan_plotter.plot_joint_limit_margins()
            plt.show()


if __name__ == '__main__':
    unittest.main()
