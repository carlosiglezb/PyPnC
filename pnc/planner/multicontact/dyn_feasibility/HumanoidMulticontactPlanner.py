import time
import numpy as np
import crocoddyl
from pnc.planner.multicontact.dyn_feasibility.humanoid_action_models import (createMultiFrameActionModel,
                                                                             createMultiFrameFinalActionModel,
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

class HumanoidMulticontactPlanner:
    def __init__(self, robot_model, knots_lst, time_per_phase, ik_cfree_planner):
        tot_num_knots = sum(knots_lst)
        self.horizon_lst = knots_lst
        self.lh_targets = np.zeros((tot_num_knots, 3))
        self.rh_targets  = np.zeros((tot_num_knots, 3))
        self.lf_targets = np.zeros((tot_num_knots, 3))
        self.rf_targets = np.zeros((tot_num_knots, 3))
        self.lkn_targets = np.zeros((tot_num_knots, 3))
        self.rkn_targets = np.zeros((tot_num_knots, 3))
        self.base_targets = np.zeros((tot_num_knots, 3))
        self.ee_rpy = {'LH': [0., 0., 0.], 'RH': [0., 0., 0.]}

        self.contact_phases = num_contact_phases = len(knots_lst)
        self.fddp = [crocoddyl.SolverFDDP] * num_contact_phases
        self.model_seqs = []
        self.T = time_per_phase

        # TODO set some default values
        self.gains = None
        self.x0 = None
        self.plan_to_model_frames = None
        self.plan_to_model_ids = None
        self.ik_cfree_planner = ik_cfree_planner

        # Crocoddyl variables / parameters
        self.state = crocoddyl.StateMultibody(robot_model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

        self.knot_idx = 0

    def set_initial_configuration(self, x0):
        self.x0 = x0

    def set_plan_to_model_params(self, plan_to_model_frames, plan_to_model_ids):
        self.plan_to_model_frames = plan_to_model_frames
        self.plan_to_model_ids = plan_to_model_ids

    def plan(self):
        dyn_solve_time = 0.

        state = self.state
        actuation = self.actuation
        x0 = self.x0
        T = self.T
        ee_rpy = self.ee_rpy
        plan_to_model_frames = self.plan_to_model_frames
        plan_to_model_ids = self.plan_to_model_ids
        ik_cfree_planner = self.ik_cfree_planner
        gains = self.gains

        model_seqs = self.model_seqs
        fddp = self.fddp
        for i in range(self.contact_phases):
            if i == 1:
                self.update_ee_rpy_left_wall()
            N_current = self.horizon_lst[i]
            DT = T / (N_current - 1)
            for t in np.linspace(i * T, (i + 1) * T, N_current):
                if t == (i + 1) * T:
                    b_terminal_step = True
                    gains = self.get_terminal_gains()
                frame_targets_dict = pack_current_targets(ik_cfree_planner, plan_to_model_frames, t)
                if t < (i + 1) * T:
                    dmodel = createMultiFrameActionModel(state,
                                                         actuation,
                                                         x0,
                                                         plan_to_model_ids,
                                                         ['LF', 'RF'],
                                                         ee_rpy,
                                                         frame_targets_dict,
                                                         None,
                                                         gains=gains,
                                                         terminal_step=b_terminal_step)
                    model_seqs += createSequence([dmodel], DT, 1)
                else:
                    dmodel = createMultiFrameFinalActionModel(state,
                                                              actuation,
                                                              x0,
                                                              plan_to_model_ids,
                                                              ['LF', 'RF'],
                                                              ee_rpy,
                                                              frame_targets_dict,
                                                              None,
                                                              gains=gains,
                                                              terminal_step=b_terminal_step)
                    model_seqs += createFinalSequence([dmodel])
                    print(f"Applying Impulse model at {i}")

                self.base_targets[self.knot_idx] = frame_targets_dict['torso']
                self.lf_targets[self.knot_idx] = frame_targets_dict['LF']
                self.rf_targets[self.knot_idx] = frame_targets_dict['RF']
                self.lh_targets[self.knot_idx] = frame_targets_dict['LH']
                self.rh_targets[self.knot_idx] = frame_targets_dict['RH']
                self.rkn_targets[self.knot_idx] = frame_targets_dict['R_knee']
                self.lkn_targets[self.knot_idx] = frame_targets_dict['L_knee']
                self.knot_idx += 1

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

        print("[Compute Time] Dynamic feasibility check: ", dyn_solve_time)

    def get_terminal_gains(self):
        pass

    def update_ee_rpy_left_wall(self):
        self.ee_rpy['LH'] = [np.pi / 2, 0., 0.]