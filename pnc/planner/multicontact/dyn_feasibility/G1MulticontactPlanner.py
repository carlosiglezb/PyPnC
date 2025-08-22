import time
from copy import copy

import numpy as np
import crocoddyl
from crocoddyl.libcrocoddyl_pywrap import StdVec_VectorX

from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import HumanoidMulticontactPlanner
from pnc.planner.multicontact.dyn_feasibility.humanoid_action_models import (createMultiFrameActionModel,
                                                                             createMultiFrameFinalActionModel,
                                                                             createMultiFrameFinalImpulseModel,
                                                                             createSequence,
                                                                             createFinalSequence, quasi_static)


def get_terminal_feet_gains():
    return np.array([12.] * 3 + [4.5] * 3)


class G1MulticontactPlanner(HumanoidMulticontactPlanner):
    def __init__(self, robot_model,
                 contact_seqs,
                 ik_cfree_planner,
                 planner_params,
                 geom_model=None):
        super().__init__(robot_model, contact_seqs, ik_cfree_planner, planner_params, geom_model)

        # names of joints used in reduced states (for plotting only)
        self.lleg_jnames = ['left_hip_roll_joint', 'left_hip_pitch_joint', 'left_hip_yaw_joint',
                            'left_knee_joint', 'left_ankle_roll_joint', 'left_ankle_pitch_joint']

        self.rleg_jnames = ['right_hip_roll_joint', 'right_hip_pitch_joint', 'right_hip_yaw_joint',
                            'right_knee_joint', 'right_ankle_roll_joint', 'right_ankle_pitch_joint']

        self.larm_jnames = ['left_shoulder_roll_joint', 'left_shoulder_pitch_joint', 'left_shoulder_yaw_joint',
                            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint']

        self.rarm_jnames = ['right_shoulder_roll_joint', 'right_shoulder_pitch_joint', 'right_shoulder_yaw_joint',
                            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']


    def plan(self):
        dyn_seg_solve_time = []

        state = self.state
        actuation = self.actuation
        x0 = self.x0
        T = self.T
        plan_to_model_ids = self.plan_to_model_ids
        planner_params = self.planner_params
        zero_config = self._zero_config

        fddp = self.fddp
        model_seq_all = []
        for i in range(self.contact_phases):
            model_seqs = []
            frames_in_contact = self.contact_planes_seq[i]
            N_current = self.horizon_lst[i]
            DT = T / (N_current - 1)
            for t in np.linspace(i * T, (i + 1) * T, N_current):
                if hasattr(self.ik_cfree_planner, "planner"):
                    frame_targets_dict = self.ik_cfree_planner.pack_current_targets(t)
                else:
                    frame_targets_dict = self.pack_current_targets(t)   # used for data reload
                if t < (i + 1) * T:
                    if i != (self.contact_phases - 1):
                        dmodel = createMultiFrameActionModel(state,
                                                             actuation,
                                                             x0,
                                                             plan_to_model_ids,
                                                             frames_in_contact,
                                                             self.contact_planes_seq[i + 1],
                                                             frame_targets_dict,
                                                             planner_weights=planner_params,
                                                             geom_model=self.geom_model,
                                                             robot_model=self.robot_model)
                    else:
                        dmodel = createMultiFrameActionModel(state,
                                                             actuation,
                                                             x0,
                                                             plan_to_model_ids,
                                                             frames_in_contact,
                                                             frames_in_contact,
                                                             frame_targets_dict,
                                                             planner_weights=planner_params)
                        # print(f"Applying Final Sequence model at {i}")
                    model_seqs += createSequence([dmodel], DT, 1)
                else:   # last time knot in current contact phase
                    if i != (self.contact_phases - 1):
                        next_frames_in_contact = self.contact_planes_seq[i + 1]
                        terminal_step = False
                    else:
                        next_frames_in_contact = frames_in_contact
                        terminal_step = True
                    # in the last time step, we use higher weights on frame orientations
                    dmodel = createMultiFrameFinalActionModel(state,
                                                              actuation,
                                                              x0,
                                                              plan_to_model_ids,
                                                              frames_in_contact,
                                                              next_frames_in_contact,
                                                              frame_targets_dict,
                                                              planner_weights=planner_params,
                                                              zero_config=zero_config,
                                                              terminal_step=terminal_step)
                    model_seqs += createFinalSequence([dmodel])
                    print(f"Last time in mode {i}. Applying Final Sequence")

                    # if in final contact phase, add extra knot to match dimensions of other phases
                    # else:
                    #     dmodel = createMultiFrameActionModel(state,
                    #                                          actuation,
                    #                                          x0,
                    #                                          plan_to_model_ids,
                    #                                          frames_in_contact,
                    #                                          ee_rpy,
                    #                                          frame_targets_dict,
                    #                                          None,
                    #                                          gains=gains,
                    #                                          terminal_step=b_terminal_step)
                    #     model_seqs += createSequence([dmodel], DT, 1)

                self.base_targets[self.knot_idx] = frame_targets_dict['torso']
                self.lf_targets[self.knot_idx] = frame_targets_dict['LF']
                self.rf_targets[self.knot_idx] = frame_targets_dict['RF']
                self.lh_targets[self.knot_idx] = frame_targets_dict['LH']
                self.rh_targets[self.knot_idx] = frame_targets_dict['RH']
                self.rkn_targets[self.knot_idx] = frame_targets_dict['R_knee']
                self.lkn_targets[self.knot_idx] = frame_targets_dict['L_knee']
                self.knot_idx += 1

            # add impulse model on frames in contact at the end of every contact phase
            # if i != (self.contact_phases - 1):
            #     imp_model = createMultiFrameFinalImpulseModel(state,
            #                                                   x0,
            #                                                   plan_to_model_ids,
            #                                                   frames_in_contact,
            #                                                   self.contact_planes_seq[i + 1],
            #                                                   frame_targets_dict,
            #                                                   planner_weights=planner_params)
            #     model_seqs = [*model_seqs, [imp_model]]
            #     new_contact_fr = [fr for fr in self.contact_planes_seq[i + 1].keys() if fr not in frames_in_contact.keys()]
            #     print(f"Applied impulse model at {i} on frame {new_contact_fr}")
            # else:
            # dmodel = createMultiFrameFinalActionModel(state,
            #                                           actuation,
            #                                           x0,
            #                                           plan_to_model_ids,
            #                                           frames_in_contact,
            #                                           frames_in_contact,
            #                                           frame_targets_dict,
            #                                           planner_weights=planner_params,
            #                                           zero_config=zero_config,
            #                                           terminal_step=True)
            # model_seqs += createFinalSequence([dmodel])
            # print(f"Applying Final Sequence model at {i}")

            problem = crocoddyl.ShootingProblem(x0, sum(model_seqs, [])[:-1], model_seqs[-1][-1])
            fddp[i] = crocoddyl.SolverFDDP(problem)

            # Adding callbacks to inspect the evolution of the solver (logs are printed in the terminal)
            # fddp[i].setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
            fddp[i].setCallbacks([crocoddyl.CallbackLogger()])

            # Solver settings
            max_iter = 200
            fddp[i].th_stop = 1e-3
            fddp[i].th_gapTol = 1e-2
            # fddp[i].th_grad = 1e-2
            # fddp[i].th_feas = 1e-3

            # Set initial guess
            xs = [x0] * (fddp[i].problem.T + 1)
            us = fddp[i].problem.quasiStatic([x0] * fddp[i].problem.T)
            # if i == 0:
            #     us = fddp[i].problem.quasiStatic([x0] * fddp[i].problem.T)
            # else:
            #     us = [quasi_static(frames_in_contact, state.pinocchio, x0)] * fddp[i].problem.T
            start_ddp_solve_time = time.time()
            print("Problem solved to convergence:", fddp[i].solve(xs, us, max_iter))
            dyn_seg_solve_time.append(time.time() - start_ddp_solve_time)
            print(f"Is feasible: {fddp[i].isFeasible}")
            print("Number of iterations:", fddp[i].iter)
            print("Total cost:", fddp[i].cost)
            print("Gradient norm:", fddp[i].stoppingCriteria())
            print("Time to solve:", dyn_seg_solve_time[-1])
            print("===============")

            # Set final state as initial state of next phase
            x0 = fddp[i].xs[-1]
            model_seq_all.append(np.copy([*model_seqs]))

        super().update_costs_from_solver(solver_type='seq')
        self.solver_stats['contacts_phases_solve_times'] = dyn_seg_solve_time
        print("[Compute Time] Dynamic feasibility check: ", sum(dyn_seg_solve_time))

        #
        # Full trajectory with Impulse models
        #
        # Add Impulse model into previous phases
        x_guess = []
        u_guess = StdVec_VectorX.copy(fddp[0].us)
        u_guess.append(fddp[0].us[-1])  # add for last step
        u_guess.append(np.array([]))    # add for impulse
        for i in range(self.contact_phases - 1):
            frames_in_contact = self.contact_planes_seq[i]
            next_frames_in_contact = self.contact_planes_seq[i + 1]
            if hasattr(self.ik_cfree_planner, "planner"):
                frame_targets_dict = self.ik_cfree_planner.pack_current_targets((i + 1) * T)
            else:
                frame_targets_dict = self.pack_current_targets((i + 1) * T)  # used for data reload
            x_last = fddp[i].xs[-1]
            # add impulse model on frames in contact at the end of every contact phase
            imp_model = createMultiFrameFinalImpulseModel(state,
                                                          x_last,
                                                          plan_to_model_ids,
                                                          frames_in_contact,
                                                          next_frames_in_contact,
                                                          frame_targets_dict,
                                                          planner_weights=planner_params)
            model_seq_all.insert(2 * i + 1, imp_model)

            # re-construct initial guess trajectory
            x_guess += fddp[i].xs.tolist()  # for full trajectory
            x_guess += [fddp[i].xs[-1]]  # add for impulse
            if i > 0:
                for j in range(len(fddp[i].us)):
                    u_guess.append(fddp[i].us[j])
                u_guess.append(fddp[i].us[-1])
                u_guess.append(np.array([]))

        x_guess += fddp[i+1].xs.tolist()  # for full trajectory
        for j in range(len(fddp[i+1].us)):
            u_guess.append(fddp[i+1].us[j])

        # Re-compute as full hybrid trajectory with Impulse model
        problem_full = crocoddyl.ShootingProblem(fddp[0].xs[0], sum(np.vstack(model_seq_all).tolist(),[])[:-1], model_seq_all[-1][-1].tolist()[0])
        self.fddp_full = crocoddyl.SolverFDDP(problem_full)
        self.fddp_full.setCallbacks([crocoddyl.CallbackLogger()])

        start_full_fddp_solve_time = time.time()
        print("Problem solved to convergence:", self.fddp_full.solve(x_guess, u_guess, max_iter))
        full_dyn_solve_time = time.time() - start_full_fddp_solve_time
        print("Is feasible:", self.fddp_full.isFeasible)
        print(f"Full hybrid TO solve time: {full_dyn_solve_time}")
        super().update_costs_from_solver(solver_type='full')

    def reset_default_gains(self, frame_name: str, updated_gains: np.array):
        self.planner_params.WBC_FRAME_TRACKING_GAINS[frame_name] = updated_gains
        # self._default_gains[frame_name] = updated_gains

    def set_zero_configuration(self, joint_configuration):
        self._zero_config = joint_configuration