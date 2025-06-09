import time
from copy import copy

import numpy as np
import crocoddyl
from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import HumanoidMulticontactPlanner
from pnc.planner.multicontact.dyn_feasibility.humanoid_action_models import (createMultiFrameActionModel,
                                                                             createMultiFrameFinalActionModel,
                                                                             createSequence,
                                                                             createFinalSequence,
                                                                             createMultiFrameFinalImpulseModel)


def get_terminal_feet_gains():
    return np.array([12.] * 3 + [4.0, 2.5, 1.0])


class ErgoCubMulticontactPlanner(HumanoidMulticontactPlanner):
    def __init__(self, robot_model, contact_seqs, time_per_phase, ik_cfree_planner):
        super().__init__(robot_model, contact_seqs, time_per_phase, ik_cfree_planner)

        self.gains = {
            'torso': np.array([1.5, 3.5, 1.0] + [0.5, 1.0, 0.01]),  # (lin, ang)
            'feet': np.array([12.] * 3 + [0.01] * 3),  # (lin, ang)
            'L_knee': np.array([4.] * 3 + [0.00001] * 3),
            'R_knee': np.array([4.] * 3 + [0.00001] * 3),
            'LH': np.array([2.] * 3 + [0.00001] * 3),
            'RH': np.array([0.5] * 3 + [0.00001] * 3)
        }
        self._default_gains = copy(self.gains)
        self._zero_config = None

        # names of joints used in reduced states (for plotting only)
        self.lleg_jnames = ['l_hip_roll', 'l_hip_pitch', 'l_hip_yaw',
                            'l_knee', 'l_ankle_roll', 'l_ankle_pitch']

        self.rleg_jnames = ['r_hip_roll', 'r_hip_pitch', 'r_hip_yaw',
                            'r_knee', 'r_ankle_roll', 'r_ankle_pitch']


    def plan(self):
        print("===============")

        dyn_solve_time = 0.
        b_terminal_step = False

        state = self.state
        actuation = self.actuation
        x0 = self.x0
        T = self.T
        plan_to_model_ids = self.plan_to_model_ids
        ik_cfree_planner = self.ik_cfree_planner
        gains = self.gains
        zero_config = self._zero_config
        speed_up = 2.5

        fddp = self.fddp
        for i in range(self.contact_phases):
            model_seqs = []
            frames_in_contact = self.contact_planes_seq[i]
            N_current = self.horizon_lst[i]
            DT = T / (N_current - 1)
            t = i * T
            while t < (i + 1) * T:
            # for t in np.linspace(i * T, (i + 1) * T, N_current):
            #     frame_targets_dict = self.pack_current_targets(t)   # used for data reload
                frame_targets_dict = ik_cfree_planner.pack_current_targets(t)

                # if we are not in the last contact phase, use regular action model
                if t < (i + 1) * T - DT:
                    if i != (self.contact_phases - 1):
                        dmodel = createMultiFrameActionModel(state,
                                                             actuation,
                                                             x0,
                                                             plan_to_model_ids,
                                                             frames_in_contact,
                                                             self.contact_planes_seq[i + 1],
                                                             frame_targets_dict,
                                                             gains=gains)
                    else:
                        dmodel = createMultiFrameActionModel(state,
                                                             actuation,
                                                             x0,
                                                             plan_to_model_ids,
                                                             frames_in_contact,
                                                             frames_in_contact,
                                                             frame_targets_dict,
                                                             gains=gains)
                    model_seqs += createSequence([dmodel], DT, 1)
                else:   # this is the last step of the contact phase
                    # if we are not in the last contact phase, use Final action model
                    if i != (self.contact_phases - 1):      # TODO remove this condition?, terminal is False
                        dmodel = createMultiFrameFinalActionModel(state,
                                                                  actuation,
                                                                  x0,
                                                                  plan_to_model_ids,
                                                                  frames_in_contact,
                                                                  self.contact_planes_seq[i + 1],
                                                                  frame_targets_dict,
                                                                  gains=gains)
                        model_seqs += createFinalSequence([dmodel])
                        print(f"Applying (last) Final Sequence model at {i}")

                # ------ speed up last step before balance
                if i == (self.contact_phases - 2) and t < (i + 1) * T:
                    t += speed_up * DT
                    # if it goes past this contact phase, set it to the end of the phase
                    if t > (i + 1) * T - (speed_up * DT + 0.001):
                        gains['feet'] = get_terminal_feet_gains()
                # ------ last step speed up (end)
                elif t > (i + 1) * T - (DT + 0.001):
                    b_terminal_step = True
                    gains['feet'] = get_terminal_feet_gains()
                    t += DT
                else:
                    t += DT

                self.base_targets[self.knot_idx] = frame_targets_dict['torso']
                self.lf_targets[self.knot_idx] = frame_targets_dict['LF']
                self.rf_targets[self.knot_idx] = frame_targets_dict['RF']
                self.lh_targets[self.knot_idx] = frame_targets_dict['LH']
                self.rh_targets[self.knot_idx] = frame_targets_dict['RH']
                self.rkn_targets[self.knot_idx] = frame_targets_dict['R_knee']
                self.lkn_targets[self.knot_idx] = frame_targets_dict['L_knee']
                self.knot_idx += 1

            # add impulse model on frames in contact at the end of every contact phase
            if i != (self.contact_phases - 1):
                imp_model = createMultiFrameFinalImpulseModel(state,
                                                              x0,
                                                              plan_to_model_ids,
                                                              frames_in_contact,
                                                              self.contact_planes_seq[i + 1],
                                                              frame_targets_dict,
                                                              gains=gains)
                model_seqs = [*model_seqs, [imp_model]]
                new_contact_fr = [fr for fr in self.contact_planes_seq[i + 1].keys() if fr not in frames_in_contact.keys()]
                print(f"Applied impulse model at {i} on frame {new_contact_fr}")
            else:
                dmodel = createMultiFrameFinalActionModel(state,
                                                          actuation,
                                                          x0,
                                                          plan_to_model_ids,
                                                          frames_in_contact,
                                                          frames_in_contact,
                                                          frame_targets_dict,
                                                          gains=gains,
                                                          zero_config=zero_config,
                                                          terminal_step=True)
                model_seqs += createFinalSequence([dmodel])
                print(f"Applying Final Sequence instead of impulse at last sequence: {i}")

            problem = crocoddyl.ShootingProblem(x0, sum(model_seqs, [])[:-1], model_seqs[-1][-1])
            fddp[i] = crocoddyl.SolverFDDP(problem)

            # Adding callbacks to inspect the evolution of the solver (logs are printed in the terminal)
            fddp[i].setCallbacks([crocoddyl.CallbackLogger()])

            # Solver settings
            max_iter = 200
            fddp[i].th_stop = 1e-3

            # Set initial guess
            xs = [x0] * (fddp[i].problem.T + 1)
            us = fddp[i].problem.quasiStatic([x0] * fddp[i].problem.T)
            start_ddp_solve_time = time.time()
            print("Problem solved:", fddp[i].solve(xs, us, max_iter))
            dyn_seg_solve_time = time.time() - start_ddp_solve_time
            print("Number of iterations:", fddp[i].iter)
            print("Total cost:", fddp[i].cost)
            print("Gradient norm:", fddp[i].stoppingCriteria())
            print("Time to solve:", dyn_seg_solve_time)
            print("===============")
            dyn_solve_time += dyn_seg_solve_time

            # save data
            # if B_SAVE_DATA:
            #     for ti in range(len(fddp[i].us)):
            #         data_saver.add('time', float(i*T + ti*T/(len(fddp[i].xs)-1)))
            #         data_saver.add('q_base', list(fddp[i].xs[ti][:7]))
            #         data_saver.add('q_joints', list(fddp[i].xs[ti][7:state.nq]))
            #         data_saver.add('qd_base', list(fddp[i].xs[ti][state.nq:state.nq+6]))
            #         data_saver.add('qd_joints', list(fddp[i].xs[ti][state.nq+6:]))
            #         data_saver.add('tau_joints', list(fddp[i].us[ti]))
            #         data_saver.advance()

            # Set final state as initial state of next phase
            x0 = fddp[i].xs[-1]

            # Reset desired EE rpy and gains
            gains = copy(self._default_gains)

        # super().update_costs_from_solver()
        print("[Compute Time] Dynamic feasibility check: ", dyn_solve_time)

    def set_zero_configuration(self, joint_configuration):
        self._zero_config = joint_configuration