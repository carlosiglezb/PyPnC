import os, sys
import pickle
import numpy as np

from plot.data_saver import DataSaver

cwd = os.getcwd()
sys.path.append(cwd)

run_name = 'g1_step_on_balanceddoor'
cfree_soln_file = cwd + '/experiment_data/' + run_name


with open(cfree_soln_file + '.pkl', 'rb') as file:
    try:
        d = pickle.load(file)
        q_all_lst = d['joint_pos']
        qd_all_lst = d['joint_vel']
        tau_all_lst = d['joint_torque']
        com_all_lst = d['center_of_mass']
        torso_act = d['torso_act']
        lf_act = d['lf_act']
        rf_act = d['rf_act']
        lkn_act = d['lkn_act']
        rkn_act = d['rkn_act']
        lh_act = d['lh_act']
        rh_act = d['rh_act']
        grf_lfoot = d['grf_lfoot']
        grf_rfoot = d['grf_rfoot']
        grf_lhand = d['grf_lhand']
        grf_rhand = d['grf_rhand']
        time = d['time']
        bez_path = d['bez_path']
        bez_points = d['bez_points']
    except EOFError:
        raise NotImplementedError

# using full TO with impulse model, separate by contact phase
q_phases, qd_phases, tau_phases, com_phases, time_phases = [], [], [], [], []
torso_act_phases, lf_act_phases, rf_act_phases = [], [], []
lkn_act_phases, rkn_act_phases, lh_act_phases, rh_act_phases = [], [], [], []
i_np = 0
N_HORIZON_LST = [180, 200, 250, 220, 200]
for n in N_HORIZON_LST:
    prev_idx = sum(N_HORIZON_LST[:i_np]) + i_np
    next_idx = prev_idx + n
    q_phases.append(q_all_lst[prev_idx:next_idx])
    qd_phases.append(qd_all_lst[prev_idx:next_idx])
    tau_phases.append(tau_all_lst[prev_idx:next_idx])
    com_phases.append(com_all_lst[prev_idx:next_idx])
    torso_act_phases.append(torso_act[prev_idx:next_idx])
    lf_act_phases.append(lf_act[prev_idx:next_idx])
    rf_act_phases.append(rf_act[prev_idx:next_idx])
    lkn_act_phases.append(lkn_act[prev_idx:next_idx])
    rkn_act_phases.append(rkn_act[prev_idx:next_idx])
    lh_act_phases.append(lh_act[prev_idx:next_idx])
    rh_act_phases.append(rh_act[prev_idx:next_idx])
    # time_phases.append(time[prev_idx:next_idx])
    i_np += 1
# Optionally re-assign the *_all variables to their phased versions if needed

data_saver = DataSaver(run_name + '_parsed.pkl')
data_saver.add('bez_points', bez_points)
data_saver.add('bez_path', bez_path)
for i in range(len(N_HORIZON_LST)):
    data_saver.add('joint_pos', q_phases[i])
    data_saver.add('joint_vel', qd_phases[i])
    data_saver.add('joint_torque', tau_phases[i])
    data_saver.add('center_of_mass', com_phases[i])
    data_saver.add('torso_act', torso_act_phases[i])
    data_saver.add('lf_act', lf_act_phases[i])
    data_saver.add('rf_act', rf_act_phases[i])
    data_saver.add('lkn_act', lkn_act_phases[i])
    data_saver.add('rkn_act', rkn_act_phases[i])
    data_saver.add('lh_act', lh_act_phases[i])
    data_saver.add('rh_act', rh_act_phases[i])
    if i == len(N_HORIZON_LST) - 1:
        data_saver.add('grf_lfoot', grf_lfoot)
        data_saver.add('grf_rfoot', grf_rfoot)
        data_saver.add('grf_lhand', grf_lhand)
        data_saver.add('grf_rhand', grf_rhand)
        data_saver.add('time', time)
    data_saver.advance()
data_saver.close()