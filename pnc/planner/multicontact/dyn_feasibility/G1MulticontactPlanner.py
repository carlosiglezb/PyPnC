import time
from copy import copy

import mim_solvers
import numpy as np
import pinocchio as pin
import crocoddyl
from crocoddyl.libcrocoddyl_pywrap import StdVec_VectorX

from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import HumanoidMulticontactPlanner
from pnc.planner.multicontact.dyn_feasibility.humanoid_action_models import (createMultiFrameActionModel,
                                                                             createMultiFrameFinalActionModel,
                                                                             createMultiFrameFinalImpulseModel,
                                                                             createSequence,
                                                                             createFinalSequence,
                                                                             quasi_static_ocp)
from pnc.planner.multicontact.kin_feasibility.g1_ik_solver import G1IKSolver


class FeasibilityExitCallback(mim_solvers.CallbackAbstract):
    def __init__(self, gap_threshold=0.5, constraint_threshold=1.0):
        super().__init__()
        self.gap_threshold = gap_threshold
        self.constraint_threshold = constraint_threshold

    def __call__(self, solver, args={}):
        # Extract current norms from the solver
        # In mim-solvers CSQP, these are solver.gaps_norm and solver.constraint_norm
        current_gaps = solver.gap_norm
        current_cons = solver.constraint_norm

        # Check if feasibility criteria are met
        if current_gaps < self.gap_threshold and current_cons < self.constraint_threshold:
            # Setting the stop threshold to a large value to force termination
            solver.termination_tolerance = solver.KKT + 1.0
            solver.stop_reached = True
            print(f"--> Feasibility reached: Gaps={current_gaps:.4f}, Cons={current_cons:.4f}. Terminating.")


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
        self.joint_names_dict = {
            'left_leg': self.lleg_jnames,
            'right_leg': self.rleg_jnames,
            'left_arm': self.larm_jnames,
            'right_arm': self.rarm_jnames
        }


    def plan(self, b_solve_hybrid: bool=True,
             integration_type: str='Euler',
             sca_refinement: bool=False,
             b_solve_by_sections: str='None',
             solver_type: str='SQP',
             b_use_knees: bool=True):
        dyn_seg_solve_time = []
        if solver_type == 'SQP':
            b_sqp = True
        else:
            b_sqp = False

        state = self.state
        actuation = self.actuation
        x0 = self.x0
        T = self.T
        plan_to_model_ids = self.plan_to_model_ids
        planner_params = self.planner_params
        zero_config = self._zero_config

        model_seq_all = []
        #
        # First, solve without impacts.
        # * Option 1 (seq): solve by sections, one contact phase at a time
        # * Option 2 (single): solve as single TO
        #
        if b_solve_by_sections == 'seq':
            self.solver_type = 'seq'
            fddp = self.fddp
            for i in range(self.contact_phases):
                model_seqs = []
                frames_in_contact = self.contact_planes_seq[i]
                N_current = self.horizon_lst[i]
                DT = T / N_current
                for t in np.linspace(i * T, (i + 1) * T, N_current):
                    frame_targets_dict = self.get_targets_from_planner(i, t)
                    if t < (i + 1) * T:
                        # get upcoming frames in contact (unless in last contact phase)
                        if i != (self.contact_phases - 1):
                            next_frames_in_contact = self.contact_planes_seq[i + 1]
                        else:
                            # last contact phase
                            next_frames_in_contact = frames_in_contact
                        dmodel = createMultiFrameActionModel(state,
                                                             actuation,
                                                             x0,
                                                             plan_to_model_ids,
                                                             frames_in_contact,
                                                             next_frames_in_contact,
                                                             frame_targets_dict,
                                                             joint_names_dict=self.joint_names_dict,
                                                             planner_weights=planner_params,
                                                             geom_model=self.geom_model,
                                                             robot_model=self.robot_model,
                                                             b_sqp=b_sqp)
                        model_seqs += createSequence([dmodel], DT, 1, integration_type)
                    else:   # last time knot in current contact phase
                        if i != (self.contact_phases - 1):
                            next_frames_in_contact = self.contact_planes_seq[i + 1]
                            terminal_step = False   # only set desired joint velocities to zero
                        else:
                            next_frames_in_contact = frames_in_contact
                            terminal_step = True    # sets desired pose to zero config and zero joint velocities
                        # in the last time step, we use higher weights on frame orientations
                        dmodel = createMultiFrameFinalActionModel(state,
                                                                  actuation,
                                                                  x0,
                                                                  plan_to_model_ids,
                                                                  frames_in_contact,
                                                                  next_frames_in_contact,
                                                                  frame_targets_dict,
                                                                  joint_names_dict=self.joint_names_dict,
                                                                  planner_weights=planner_params,
                                                                  zero_config=zero_config,
                                                                  terminal_step=terminal_step,
                                                                  robot_model=self.robot_model)
                        model_seqs += createFinalSequence([dmodel], integration_type)
                        print(f"Last time in mode {i}. Applying Final Sequence with terminal_step={terminal_step}")

                    # save targets
                    self.save_targets(frame_targets_dict)

                problem = crocoddyl.ShootingProblem(x0, sum(model_seqs, [])[:-1], model_seqs[-1][-1])
                if solver_type == 'SQP':
                    print("[SCA-Crocoddyl] Using CSQP solver for sequential TO")
                    fddp[i] = mim_solvers.SolverCSQP(problem)
                    fddp[i].setCallbacks([mim_solvers.CallbackLogger(), mim_solvers.CallbackVerbose()])
                    fddp[i].termination_tolerance = 1e-1
                    fddp[i].eps_abs = 1e-1
                    fddp[i].eps_rel = 1e-1
                    fddp[i].filter_size = 10     # documentation says not to change this!
                    fddp[i].rho_update_interval = 50
                    fddp[i].update_rho_with_heuristic = True
                    fddp[i].max_qp_iters = 500
                    # fddp[i].use_filter_line_search = False   # (default: True)
                    # fddp[i].mu_dynamic = -1  # Nocedal's L1 merit function
                    # fddp[i].lag_mul_inf_norm_coef = 10
                    # fddp[i].max_qp_iters = 50
                else:
                    print("[SCA-Crocoddyl] Using BoxFDDP solver for sequential TO")
                    fddp[i] = crocoddyl.SolverBoxFDDP(problem)

                    # Adding callbacks to inspect the evolution of the solver (logs are printed in the terminal)
                    # fddp[i].setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
                    fddp[i].setCallbacks([crocoddyl.CallbackLogger()])

                # Solver settings
                max_iter = 100
                fddp[i].th_stop = 1e-3
                fddp[i].th_gapTol = 1e-2
                fddp[i].reg_max = 1e4
                fddp[i].reg_incFactor = 3
                fddp[i].reg_decFactor = 3
                if i == 1 or i == 2 or i == 3:   # harder to solve, needs more iterations
                    fddp[i].reg_incFactor = 2         # default is 10 (smaller works for tight guess)
                    fddp[i].reg_decFactor = 2         # default is 10 (smaller works for tight guess)
                #     fddp[i].th_acceptStep = 0.01        # default is 0.1
                # fddp[i].th_acceptStep = 0.01     # default is 0.1
                # fddp[i].reg_min = 1e-3             # default is 1e-9
                # fddp[i].reg_max = 1e3
                # fddp[i].th_gaptol(1e-12); // default is 1e-16
                # fddp[i].th_grad = 1e-2
                # fddp[i].th_feas = 1e-3
                # fddp[i].th_stop(1e-2)

                # Set initial guess
                xs = [x0] * (fddp[i].problem.T + 1)
                us_static = quasi_static_ocp(frames_in_contact, plan_to_model_ids, state.pinocchio, x0)
                if integration_type == 'RK2':
                    us_static = np.concatenate((us_static, us_static))
                us = [us_static] * fddp[i].problem.T
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
                x0 = copy(fddp[i].xs[-1])
                model_seq_all.append(np.copy([*model_seqs]))

            super().update_costs_from_solver(solver_type='seq', integration_type=integration_type)
            self.solver_stats['contacts_phases_solve_times'] = dyn_seg_solve_time
            print("[Compute Time] Dynamic feasibility check: ", sum(dyn_seg_solve_time))
        elif b_solve_by_sections == 'single':
            self.solver_type = 'single'
            x0_stance = np.copy(x0)

            # here, we use IK to construct the initial guess
            robot_data = self.robot_model.createData()
            pin.forwardKinematics(self.robot_model, robot_data, x0_stance[:self.robot_model.nq])
            g1_ik = G1IKSolver(self.robot_model, robot_data, x0_stance[:self.robot_model.nq], b_use_knees)

            xs, us = [], []
            for i in range(self.contact_phases):
                frames_in_contact = self.contact_planes_seq[i]
                N_current = self.horizon_lst[i]

                # update initial guess joints based on IK of current targets
                q_ik = g1_ik.solve(self.get_targets_from_planner(i, i * T), x0_stance[:self.robot_model.nq])
                x0_stance[:self.robot_model.nq] = q_ik
                xs += [x0_stance] * N_current

                us_static = quasi_static_ocp(frames_in_contact, plan_to_model_ids, state.pinocchio, x0_stance)
                if integration_type == 'RK2':
                    us_static = np.concatenate((us_static, us_static))
                us += [us_static] * N_current

                # create next stance initial guess
                x0_next_stance = np.copy(x0_stance)
                q_ik = g1_ik.solve(self.get_targets_from_planner(i, (i + 1) * T), x0_stance[:self.robot_model.nq])
                x0_stance[:self.robot_model.nq] = q_ik

                # construct TO models for this contact phase
                model_seqs = []
                DT = T / N_current
                # for t in np.arange(i * T, (i + 1) * T, DT):
                for t in np.linspace(i * T, (i + 1) * T, N_current):
                    frame_targets_dict = self.get_targets_from_planner(i, t)

                    # get upcoming frames in contact (unless in last contact phase)
                    if i != (self.contact_phases - 1):
                        next_frames_in_contact = self.contact_planes_seq[i + 1]
                    else:
                        # last contact phase
                        next_frames_in_contact = self.contact_planes_seq[i]
                    dmodel = createMultiFrameActionModel(state,
                                                         actuation,
                                                         x0_next_stance,
                                                         plan_to_model_ids,
                                                         frames_in_contact,
                                                         next_frames_in_contact,
                                                         frame_targets_dict,
                                                         joint_names_dict=self.joint_names_dict,
                                                         planner_weights=planner_params,
                                                         geom_model=self.geom_model,
                                                         robot_model=self.robot_model,
                                                         b_sqp=b_sqp)
                    model_seqs += createSequence([dmodel], DT, 1, integration_type)

                    # save targets
                    self.save_targets(frame_targets_dict)

                model_seq_all.append(np.copy([*model_seqs]))

            frame_targets_dict = self.get_targets_from_planner(i, self.contact_phases*T)
            terminal_step = True    # sets desired pose to zero config and zero joint velocities
            # in the last time step, we use higher weights on frame orientations
            dmodel = createMultiFrameFinalActionModel(state,
                                                      actuation,
                                                      x0,
                                                      plan_to_model_ids,
                                                      frames_in_contact,
                                                      frames_in_contact,
                                                      frame_targets_dict,
                                                      joint_names_dict=self.joint_names_dict,
                                                      planner_weights=planner_params,
                                                      zero_config=zero_config,
                                                      terminal_step=terminal_step,
                                                      robot_model=self.robot_model)
            model_seq_all += createFinalSequence([dmodel], integration_type)
            print(f"Last time in mode {i}. Applying Final Sequence with terminal_step={terminal_step}")
            xs += [x0_stance]

            problem = crocoddyl.ShootingProblem(x0, sum(np.vstack(model_seq_all).tolist(),[])[:-1], model_seq_all[-1][-1])
            if solver_type == 'SQP':
                print("[SCA-Crocoddyl] Using CSQP solver for single TO")
                fddp = mim_solvers.SolverCSQP(problem)
                customFeas = FeasibilityExitCallback(gap_threshold=1e0, constraint_threshold=1e0)
                fddp.setCallbacks([mim_solvers.CallbackLogger(), mim_solvers.CallbackVerbose(), customFeas])
                fddp.termination_tolerance = 0.1  # relaxed for single TO used to warm-start
                fddp.eps_abs = 1e-2
                fddp.eps_rel = 1e-2
                fddp.filter_size = 10    # documentation says not to change this!
                fddp.update_rho_with_heuristic = True
                fddp.rho_update_interval = 50
                fddp.max_qp_iters = 100
                # fddp.use_filter_line_search = False   # (default: True)
                # fddp.mu_dynamic = -1  # Nocedal's L1 merit function
                # fddp.lag_mul_inf_norm_coef = 10
                # fddp.max_qp_iters = 50
            else:
                print("[SCA-Crocoddyl] Using BoxFDDP solver for single TO")
                fddp = crocoddyl.SolverBoxFDDP(problem)

                # Adding callbacks to inspect the evolution of the solver (logs are printed in the terminal)
                fddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
                # fddp.setCallbacks([crocoddyl.CallbackLogger()])

            # Solver settings
            max_iter = 100
            fddp.th_stop = 1e-2
            fddp.th_gapTol = 1e-2
            # fddp.reg_max = 1e4
            fddp.reg_incFactor = 3
            fddp.reg_decFactor = 3

            # Set initial guess
            # xs = [x0] * (fddp.problem.T + 1)
            # us_static = quasi_static_ocp(frames_in_contact, plan_to_model_ids, state.pinocchio, x0)
            # if integration_type == 'RK2':
            #     us_static = np.concatenate((us_static, us_static))
            # us = [us_static] * fddp.problem.T
            # us = problem.quasiStatic(xs[:-1])
            start_ddp_solve_time = time.time()
            print("Problem solved to convergence:", fddp.solve(xs, us, max_iter))
            dyn_seg_solve_time.append(time.time() - start_ddp_solve_time)
            print(f"Is feasible: {fddp.isFeasible}")
            print("Number of iterations:", fddp.iter)
            print("Total cost:", fddp.cost)
            print("Gradient norm:", fddp.stoppingCriteria())
            print("Time to solve:", dyn_seg_solve_time[-1])
            print("===============")

            # Set final state as initial state of next phase
            self.fddp_single = fddp
            self.solver_type = 'single'
            super().update_costs_from_solver(solver_type='single', integration_type=integration_type)
            self.solver_stats['contacts_phases_solve_times'] = dyn_seg_solve_time

        else:
            self.solver_type = None
            print(f"b_solve_by_sections set to {b_solve_by_sections}. Skipping 'seq' and 'single' step.")

        if b_solve_hybrid:
            # TODO refactor into method to apply at different stages
            #
            # Full trajectory with Impulse models
            #
            # Add Impulse model into previous solution (either single or by seq)
            x_guess = []
            latest_fddp = self.get_latest_fddp()
            u_guess = StdVec_VectorX()
            # u_guess = StdVec_VectorX.copy(latest_fddp.us)
            # u_guess.append(u_guess[-1])  # add for last step
            # u_guess.append(np.array([]))  # add for impulse
            startIdx =  0
            for i in range(self.contact_phases - 1):
                frames_in_contact = self.contact_planes_seq[i]
                next_frames_in_contact = self.contact_planes_seq[i + 1]
                frame_targets_dict = self.get_targets_from_planner(i, (i + 1) * T)
                last_idx_before_impact = startIdx + self.horizon_lst[i] - 1
                x_last = copy(latest_fddp.xs[last_idx_before_impact])
                # add impulse model on frames in contact at the end of every contact phase
                imp_model = createMultiFrameFinalImpulseModel(state,
                                                              x_last,
                                                              plan_to_model_ids,
                                                              frames_in_contact,
                                                              next_frames_in_contact,
                                                              frame_targets_dict,
                                                              planner_weights=planner_params)
                model_seq_all.insert(2 * i + 1, imp_model)

                # remove previous terminal models and replace with runningModel
                model_seq_all[2 * i] = np.reshape(np.delete(model_seq_all[2 * i], -1), (-1, 1))
                last_entry = copy(model_seq_all[2 * i][-1, 0])
                model_seq_all[2 * i] = np.concatenate(
                    (model_seq_all[2 * i], np.reshape(np.array([last_entry]), (-1, 1))))

                # re-construct initial guess trajectory
                # x_guess += fddp[i].xs.tolist()  # for full trajectory
                endIdx = startIdx + self.horizon_lst[i]
                x_guess += latest_fddp.xs[startIdx:endIdx].tolist()  # for full trajectory
                x_guess += [x_guess[-1]]  # add for impulse
                StdVec_VectorX.extend(u_guess, StdVec_VectorX.copy(latest_fddp.us[startIdx:endIdx]))   # values from current phase
                StdVec_VectorX.append(u_guess, np.array([]))            # impact has no control
                # if i > 0:
                #     for j in range(len(latest_fddp.us[startIdx:endIdx])):
                #         u_guess.append(latest_fddp.us[j])
                #     u_guess.append(latest_fddp.us[-1])
                #     u_guess.append(np.array([]))
                startIdx += self.horizon_lst[i]

            startIdx = sum(self.horizon_lst[:-1])
            endIdx = startIdx + self.horizon_lst[-1]
            x_guess += latest_fddp.xs[startIdx:].tolist()  # include last contact phase for full trajectory
            StdVec_VectorX.extend(u_guess,
                                  StdVec_VectorX.copy(latest_fddp.us[startIdx:endIdx]))  # values from current phase
            # for j in range(len(latest_fddp.us[startIdx:endIdx])):
            #     u_guess.append(latest_fddp.us[j])

            # Re-compute as full hybrid trajectory with Impulse model
            problem_full = crocoddyl.ShootingProblem(latest_fddp.xs[0], sum(np.vstack(model_seq_all).tolist(),[])[:-1], model_seq_all[-1][0])
            # self.fddp_full = crocoddyl.SolverFDDP(problem_full)
            self.fddp_full = crocoddyl.SolverBoxFDDP(problem_full)
            self.fddp_full.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
            self.fddp_full.th_stop = 1e-3
            self.fddp_full.th_gapTol = 1e-2
            self.fddp_full.reg_incFactor = 5  # default is 10 (this works for tight guess)
            self.fddp_full.reg_decFactor = 5  # default is 10 (this works for tight guess)
            # self.fddp_full.th_acceptStep = 0.01  # default is 0.1

            start_full_fddp_solve_time = time.time()
            print("Problem solved to convergence:", self.fddp_full.solve(x_guess, u_guess, max_iter))
            full_dyn_solve_time = time.time() - start_full_fddp_solve_time
            print("Is feasible:", self.fddp_full.isFeasible)
            print(f"Full hybrid TO solve time: {full_dyn_solve_time}")
            self.solver_type = 'full'
            super().update_costs_from_solver(solver_type=self.solver_type, integration_type=integration_type)

        if sca_refinement:
            self.plan_sca(solver_type='SQP', b_use_knees=b_use_knees)

    def save_targets(self, frame_targets_dict):
        if self.knot_idx >= len(self.base_targets):
            return
        if 'torso' in frame_targets_dict:
            self.base_targets[self.knot_idx] = frame_targets_dict['torso']
        if 'LF' in frame_targets_dict:
            self.lf_targets[self.knot_idx] = frame_targets_dict['LF']
        if 'RF' in frame_targets_dict:
            self.rf_targets[self.knot_idx] = frame_targets_dict['RF']
        if 'LH' in frame_targets_dict:
            self.lh_targets[self.knot_idx] = frame_targets_dict['LH']
        if 'RH' in frame_targets_dict:
            self.rh_targets[self.knot_idx] = frame_targets_dict['RH']
        if 'R_knee' in frame_targets_dict:
            self.rkn_targets[self.knot_idx] = frame_targets_dict['R_knee']
        if 'L_knee' in frame_targets_dict:
            self.lkn_targets[self.knot_idx] = frame_targets_dict['L_knee']
        self.knot_idx += 1

    def get_targets_from_planner(self, phase: int, t:float) -> dict[str, np.array]:
        if hasattr(self.ik_cfree_planner, "planner"):
            frame_targets_dict = self.ik_cfree_planner.get_frame_targets_from_kin_planner(phase, t)
        else:
            frame_targets_dict = self.pack_current_targets(t)  # used for data reload
        return frame_targets_dict

    def plan_sca(self, integration_type: str='Euler',
                 solver_type: str ='FDDP',
                 b_impulse: bool=False,
                 b_use_knees: bool=True):
        T = self.T
        state = self.state
        actuation = self.actuation
        x0 = self.x0
        plan_to_model_ids = self.plan_to_model_ids
        planner_params = self.planner_params
        zero_config = self._zero_config
        if self.solver_type is not None:
            latest_fddp_xs, latest_fddp_us = self.get_latest_fddp_xs_us()

        knot_idx = 0
        model_seqs = []

        x0_stance = np.copy(x0)

        # here, we use IK to construct the initial guess
        robot_data = self.robot_model.createData()
        pin.forwardKinematics(self.robot_model, robot_data, x0_stance[:self.robot_model.nq])
        g1_ik = G1IKSolver(self.robot_model, robot_data, x0_stance[:self.robot_model.nq], b_use_knees)

        xs, us = [], []
        for i in range(self.contact_phases):
            frames_in_contact = self.contact_planes_seq[i]
            N_current = self.horizon_lst[i]

            # update initial guess joints based on IK of current targets
            q_ik = g1_ik.solve(self.get_targets_from_planner(i, i * T), x0_stance[:self.robot_model.nq])
            x0_stance[:self.robot_model.nq] = q_ik
            xs += [x0_stance] * N_current

            us_static = quasi_static_ocp(frames_in_contact, plan_to_model_ids, state.pinocchio, x0_stance)
            if integration_type == 'RK2':
                us_static = np.concatenate((us_static, us_static))
            us += [us_static] * N_current

            DT = T / N_current
            # for t in np.arange(i * T, (i + 1) * T, DT):
            for t in np.linspace(i * T, (i + 1) * T, N_current):
                frame_targets_dict = self.get_targets_from_planner(i, t)
                # get upcoming frames in contact (unless in last contact phase)
                if i != (self.contact_phases - 1):
                    next_frames_in_contact = self.contact_planes_seq[i + 1]
                else:
                    # last contact phase
                    next_frames_in_contact = self.contact_planes_seq[i]
                dmodel = createMultiFrameActionModel(state,
                                                     actuation,
                                                     x0_stance,
                                                     plan_to_model_ids,
                                                     frames_in_contact,
                                                     next_frames_in_contact,
                                                     frame_targets_dict,
                                                     joint_names_dict=self.joint_names_dict,
                                                     planner_weights=planner_params,
                                                     geom_model=self.geom_model,
                                                     robot_model=self.robot_model,
                                                     b_sca=True)
                model_seqs += createSequence([dmodel], DT, 1, integration_type)

                # save targets if not done, yet
                if self.solver_type is None:
                    self.save_targets(frame_targets_dict)

            # Apply impulse model, except at end of last contact phase
            # Note: currently, this implementation assumes a TO has already been solved
            if b_impulse:
                if i != (self.contact_phases - 1):
                    x_last = copy(latest_fddp_xs[knot_idx])
                    imp_model = createMultiFrameFinalImpulseModel(state,
                                                                  x_last,
                                                                  plan_to_model_ids,
                                                                  frames_in_contact,
                                                                  next_frames_in_contact,
                                                                  frame_targets_dict,
                                                                  planner_weights=planner_params)
                    model_seqs += [imp_model]

        # last time knot in current contact phase
        frame_targets_dict = self.get_targets_from_planner(i, self.contact_phases * T)
        terminal_step = True    # sets desired pose to zero config and zero joint velocities
        # in the last time step, we use higher weights on frame orientations
        dmodel = createMultiFrameFinalActionModel(state,
                                                  actuation,
                                                  x0,
                                                  plan_to_model_ids,
                                                  frames_in_contact,
                                                  frames_in_contact,
                                                  frame_targets_dict,
                                                  joint_names_dict=self.joint_names_dict,
                                                  planner_weights=planner_params,
                                                  zero_config=zero_config,
                                                  terminal_step=terminal_step,
                                                  robot_model=self.robot_model)
        model_seqs += createFinalSequence([dmodel], integration_type)
        print(f"Last time in mode {i}. Applying Final Sequence with terminal_step={terminal_step}")
        xs += [x0_stance]

        problem = crocoddyl.ShootingProblem(x0, sum(np.vstack(model_seqs).tolist(), [])[:-1], model_seqs[-1][-1])
        if solver_type == 'SQP':
            print("[SCA-Crocoddyl] Using CSQP solver for SCA refinement")
            self.fddp_full_sca = mim_solvers.SolverCSQP(problem)
            customFeas = FeasibilityExitCallback(gap_threshold=1e0, constraint_threshold=1e0)
            self.fddp_full_sca.setCallbacks([mim_solvers.CallbackLogger(), mim_solvers.CallbackVerbose(), customFeas])
            self.fddp_full_sca.termination_tolerance = 1e0
            self.fddp_full_sca.eps_abs = 5e-1
            self.fddp_full_sca.eps_rel = 5e-1
            self.fddp_full_sca.filter_size = 10  # documentation says not to change this!
            self.fddp_full_sca.update_rho_with_heuristic = True
            self.fddp_full_sca.max_qp_iters = 100
            self.fddp_full_sca.rho_update_interval = 100
            # self.fddp_full_sca.use_filter_line_search = False   # (default: True)
            # self.fddp_full_sca.mu_dynamic = -1  # Nocedal's L1 merit function
            # self.fddp_full_sca.lag_mul_inf_norm_coef = 10
            # self.fddp_full_sca.max_qp_iters = 50
        else:
            print("[SCA-Crocoddyl] Using BoxFDDP solver for SCA refinement")
            self.fddp_full_sca = crocoddyl.SolverBoxFDDP(problem)
            self.fddp_full_sca.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
            # self.fddp_full_sca.setCallbacks([crocoddyl.CallbackLogger()])

        # Solver settings
        self.fddp_full_sca.th_stop = 1e1
        self.fddp_full_sca.th_gapTol = 1e-2
        self.fddp_full_sca.reg_max = 1e4
        # self.fddp_full_sca.preg = 1e-3
        # self.fddp_full_sca.dres = 1e-3
        # self.fddp_full_sca.reg_incFactor = 3
        # self.fddp_full_sca.reg_decFactor = 3

        max_iter = 100
        # Set initial guess from latest solve
        # TODO check dimensions and/or adjust
        if self.solver_type is not None:
            xs = copy(latest_fddp_xs)
            us = StdVec_VectorX.copy(latest_fddp_us)
        else:
            print("[SCA-Crocoddyl] No previous solution available for SCA refinement. Using quasi-static guess.")
            # xs = [x0] * (self.fddp_full_sca.problem.T + 1)
            # us_static = quasi_static_ocp(frames_in_contact, plan_to_model_ids, state.pinocchio, x0)
            # us = [us_static] * self.fddp_full_sca.problem.T
            ### below is Crocoddyl's quasiStatic method
            # us = self.fddp_full_sca.problem.quasiStatic([x0] * self.fddp_full_sca.problem.T)
            # us = self.fddp_full_sca.problem.quasiStatic(xs[:-1])

        # uncomment below when removing impulse models from the guess
        # idx_removed = 0
        # for idx in range(len(self.horizon_lst) - 1):
        #     idx_sum = sum(self.horizon_lst[:idx+1])
        #     del xs[idx_sum]
        #     del us[idx_sum]
        #     idx_removed += 1

        start_ddp_solve_time = time.time()
        print("[SCA-Crocoddyl] Problem solved to convergence:", self.fddp_full_sca.solve(xs, us, max_iter))
        dyn_seg_solve_time = time.time() - start_ddp_solve_time
        print(f"[SCA-Crocoddyl] Is feasible: {self.fddp_full_sca.isFeasible}")
        print("[SCA-Crocoddyl] Number of iterations:", self.fddp_full_sca.iter)
        print("[SCA-Crocoddyl] Total cost:", self.fddp_full_sca.cost)
        print("[SCA-Crocoddyl] Time to solve:", dyn_seg_solve_time)
        print("===============")
        self.solver_type = 'sca'
        super().update_costs_from_solver(solver_type='sca', integration_type=integration_type)
        super().update_constraint_residuals_from_solver()
        self.solver_stats['sca_solve_time'] = dyn_seg_solve_time

    # def reset_default_gains(self, frame_name: str, updated_gains: np.array):
    #     self.planner_params.WBC_FRAME_TRACKING_GAINS[frame_name] = updated_gains
    #     # self._default_gains[frame_name] = updated_gains

    def get_latest_fddp(self) -> crocoddyl.SolverFDDP:
        if self.solver_type == 'full':
            latest_fddp = self.fddp_full
        elif self.solver_type == 'single':
            latest_fddp = self.fddp_single
        elif self.solver_type == 'sca':
            latest_fddp = self.fddp_full_sca
        elif self.solver_type == 'seq':
            # TODO construct full trajectory from segments
            raise NotImplementedError
        else:
            latest_fddp = None
        return latest_fddp

    def get_latest_fddp_xs_us(self) -> (StdVec_VectorX, StdVec_VectorX):
        if self.solver_type == 'full':
            xs_all = self.fddp_full.xs
            us_all = self.fddp_full.us
        elif self.solver_type == 'single':
            xs_all = self.fddp_single.xs
            us_all = self.fddp_single.us
        elif self.solver_type == 'sca':
            xs_all = self.fddp_full_sca.xs
            us_all = self.fddp_full_sca.us
        elif self.solver_type == 'seq':
            # construct full trajectory from segments
            xs_all = StdVec_VectorX()
            us_all = StdVec_VectorX()
            StdVec_VectorX.extend(xs_all, self.fddp[0].xs)
            StdVec_VectorX.extend(us_all, StdVec_VectorX.copy(self.fddp[0].us))

            # copy all segments into the first one
            for i in range(1, len(self.fddp)):
                StdVec_VectorX.extend(xs_all, self.fddp[i].xs)
                StdVec_VectorX.extend(us_all, StdVec_VectorX.copy(self.fddp[i].us))
                StdVec_VectorX.extend(us_all, StdVec_VectorX(1, self.fddp[i].us[-1]))
            # add for terminal state
            StdVec_VectorX.extend(xs_all, StdVec_VectorX(1, self.fddp[-1].xs[-1]))
            StdVec_VectorX.extend(us_all, StdVec_VectorX(1, self.fddp[-1].us[-1]))
        else:
            raise NotImplementedError("No latest fddp available.")
        return xs_all, us_all


    def set_zero_configuration(self, joint_configuration):
        self._zero_config = joint_configuration