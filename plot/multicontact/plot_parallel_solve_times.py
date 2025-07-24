import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import pickle

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

n_cores = [1, 2, 3]

kin_dmin_dict = {}
kin_smooth_dict = {}
kin_sca_build_dict = {}
kin_sca_construct_dict = {}
kin_sca_solve_dict = {}
dyn_dict = {}

# Collect and organize data
for n in n_cores:
    kin_min_d_time = []
    kin_smooth_time = []
    kin_sca_build_time = []     # when using SCA
    kin_sca_construct_time = [] # when using SCA
    kin_sca_solve_time = []     # when using SCA
    dyn_time = []

    with open('experiment_data/RAL/g1_kin_dyn_sca_'+ str(n) + '_solve_stats.pkl', 'rb') as file:
        while True:
            try:
                d = pickle.load(file)
                kin_min_d_time.append(d['kin_solver_stats']['min_reach_iris_distance_cvxpy_time'])
                kin_smooth_time.append(d['kin_solver_stats']['multiple_bezier_iris_cvxpy_time'])
                kin_sca_build_time.append(d['kin_solver_stats']['multiple_bezier_iris_sca_build_time'])
                kin_sca_construct_time.append(d['kin_solver_stats']['multiple_bezier_iris_sca_construct_time'])
                kin_sca_solve_time.append(d['kin_solver_stats']['multiple_bezier_iris_sca_casadi_time'])
                dyn_time.append(sum(d['dyn_solver_stats']['contact_phases_solve_times']))
            except EOFError:
                break
        kin_dmin_dict[str(n)] = kin_min_d_time
        kin_smooth_dict[str(n)] = kin_smooth_time
        kin_sca_build_dict[str(n)] = kin_sca_build_time
        kin_sca_construct_dict[str(n)] = kin_sca_construct_time
        kin_sca_solve_dict[str(n)] = kin_sca_solve_time
        dyn_dict[str(n)] = dyn_time

#
# Create bar plots
#
x_labels = list(kin_dmin_dict.keys())
x = np.arange(len(x_labels))

# Compute mean times for each core count
kin_dmin_means = [np.mean(kin_dmin_dict[k]) / (i+1) for i, k in enumerate(x_labels)]
kin_smooth_means = [np.mean(kin_smooth_dict[k]) / (i+1) for i, k in enumerate(x_labels)]
kin_sca_build_means = [np.mean(kin_sca_build_dict[k]) / (i+1) for i, k in enumerate(x_labels)]
kin_sca_construct_means = [np.mean(kin_sca_construct_dict[k]) / (i+1) for i, k in enumerate(x_labels)]
kin_sca_solve_means = [np.mean(kin_sca_solve_dict[k]) / (i+1) for i, k in enumerate(x_labels)]
dyn_means = [np.mean(dyn_dict[k]) / (i+1) for i, k in enumerate(x_labels)]

fig, ax = plt.subplots()
bar1 = ax.bar(x, kin_dmin_means, label='kin_dmin')
bar2 = ax.bar(x, kin_smooth_means, bottom=kin_dmin_means, label='kin_smooth')
bar3 = ax.bar(x, kin_sca_build_means, bottom=np.array(kin_dmin_means) + np.array(kin_smooth_means), label='kin_sca_build')
bar4 = ax.bar(x, kin_sca_construct_means, bottom=np.array(kin_dmin_means) + np.array(kin_smooth_means) + np.array(kin_sca_build_means), label='kin_sca_construct')
bar5 = ax.bar(x, kin_sca_solve_means, bottom=np.array(kin_dmin_means) + np.array(kin_smooth_means) + np.array(kin_sca_build_means) + np.array(kin_sca_construct_means), label='kin_sca_solve')
bar6 = ax.bar(x, dyn_means, bottom=np.array(kin_dmin_means) + np.array(kin_smooth_means), label='dyn')

ax.set_xlabel('Number of Processes')
ax.set_ylabel('Avg Solve Time per WBP (s)')
ax.set_title('Planner Solve Times by Number of Cores')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()
plt.show()
