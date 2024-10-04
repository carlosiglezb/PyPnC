import time
from copy import copy

import numpy as np
import crocoddyl
from pnc.planner.multicontact.dyn_feasibility.HumanoidMulticontactPlanner import HumanoidMulticontactPlanner
from pnc.planner.multicontact.dyn_feasibility.humanoid_action_models import (createMultiFrameActionModel,
                                                                             createMultiFrameFinalActionModel,
                                                                             createMultiFrameFinalImpulseModel,
                                                                             createSequence,
                                                                             createFinalSequence)


def pack_current_targets(ik_cfree_planner, plan_to_model_frames, t):
    lfoot_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('LF'), t)
    lknee_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('L_knee'), t)
    rfoot_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('RF'), t)
    rknee_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('R_knee'), t)
    lhand_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('LH'), t)
    rhand_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('RH'), t)
    base_t = ik_cfree_planner.get_ee_des_pos(list(plan_to_model_frames.keys()).index('torso'), t)
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

def get_rpy_normal_left_wall():
    return [np.pi / 2, 0., 0.]

def get_rpy_normal_right_wall():
    return [-np.pi / 2, 0., 0.]

def get_terminal_feet_gains():
    return np.array([10.] * 3 + [2.5] * 3)


class G1MulticontactPlanner(HumanoidMulticontactPlanner):
    def __init__(self, robot_model, contact_seqs, time_per_phase, ik_cfree_planner):
        super().__init__(robot_model, contact_seqs, time_per_phase, ik_cfree_planner)

        self.gains = {
            'torso': np.array([4.5] * 3 + [0.5] * 3),  # (lin, ang)
            'feet': np.array([8.] * 3 + [0.00001] * 3),  # (lin, ang)
            'L_knee': np.array([3.] * 3 + [0.00001] * 3),
            'R_knee': np.array([3.] * 3 + [0.00001] * 3),
            'hands': np.array([2.] * 3 + [0.00001] * 3)
        }
        self._default_gains = copy(self.gains)

        # names of joints used in reduced states (for plotting only)
        self.lleg_jnames = ['left_hip_roll_joint', 'left_hip_pitch_joint', 'left_hip_yaw_joint',
                            'left_knee_joint', 'left_ankle_roll_joint', 'left_ankle_pitch_joint']

        self.rleg_jnames = ['right_hip_roll_joint', 'right_hip_pitch_joint', 'right_hip_yaw_joint',
                            'right_knee_joint', 'right_ankle_roll_joint', 'right_ankle_pitch_joint']


    def plan(self):
        dyn_solve_time = 0.
        b_terminal_step = False

        state = self.state
        actuation = self.actuation
        x0 = self.x0
        T = self.T
        ee_rpy = self.ee_rpy
        plan_to_model_frames = self.plan_to_model_frames
        plan_to_model_ids = self.plan_to_model_ids
        ik_cfree_planner = self.ik_cfree_planner
        gains = self.gains

        fddp = self.fddp
        for i in range(self.contact_phases):
            model_seqs = []
            frames_in_contact = self.contact_seqs[i]
            # TODO change for upper call to update_contact_params() or so
            if i == 1:
                ee_rpy['LH'] = get_rpy_normal_left_wall()
            # elif i == 3:
            #     ee_rpy['RH'] = get_rpy_normal_right_wall()
            elif i > (self.contact_phases - 1):
                raise NotImplementedError(f"Frames for contact sequence {i} not specified.")
            N_current = self.horizon_lst[i]
            DT = T / (N_current - 1)
            for t in np.linspace(i * T, (i + 1) * T, N_current):
                if t == (i + 1) * T:
                    b_terminal_step = False
                    gains['feet'] = get_terminal_feet_gains()
                frame_targets_dict = pack_current_targets(ik_cfree_planner, plan_to_model_frames, t)
                if t < (i + 1) * T:
                    dmodel = createMultiFrameActionModel(state,
                                                         actuation,
                                                         x0,
                                                         plan_to_model_ids,
                                                         frames_in_contact,
                                                         ee_rpy,
                                                         frame_targets_dict,
                                                         None,
                                                         gains=gains,
                                                         terminal_step=b_terminal_step)
                    model_seqs += createSequence([dmodel], DT, 1)
                else:   # last time knot in current contact phase
                    if i != (self.contact_phases - 1):
                        # in the last time step, we use higher weights on frame orientations
                        dmodel = createMultiFrameFinalActionModel(state,
                                                                  actuation,
                                                                  x0,
                                                                  plan_to_model_ids,
                                                                  frames_in_contact,
                                                                  ee_rpy,
                                                                  frame_targets_dict,
                                                                  None,
                                                                  gains=gains,
                                                                  terminal_step=b_terminal_step)
                        model_seqs += createFinalSequence([dmodel])
                        print(f"Applying Final Sequence model at {i}")

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
            if i != (self.contact_phases - 1):
                imp_model = createMultiFrameFinalImpulseModel(state,
                                                              x0,
                                                              plan_to_model_ids,
                                                              [self.contact_seqs[i+1][-1]],
                                                              ee_rpy,
                                                              frame_targets_dict,
                                                              gains=gains)
                model_seqs = [*model_seqs, [imp_model]]
                print(f"Applied impulse model at {i} on frame {[self.contact_seqs[i + 1][-1]]}")
            else:
                dmodel = createMultiFrameFinalActionModel(state,
                                                          actuation,
                                                          x0,
                                                          plan_to_model_ids,
                                                          frames_in_contact,
                                                          ee_rpy,
                                                          frame_targets_dict,
                                                          None,
                                                          gains=gains,
                                                          terminal_step=b_terminal_step)
                model_seqs += createFinalSequence([dmodel])
                print(f"Applying Final Sequence model at {i}")

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

            # Reset desired EE rpy and gains for next contact phase
            ee_rpy = copy(self.ee_rpy)
            gains = copy(self._default_gains)

        print("[Compute Time] Dynamic feasibility check: ", dyn_solve_time)
