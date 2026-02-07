import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)
import pickle

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def main():

    # names of the experiment data files
    n_plans = [
        'g1_guided__no_imp_sca__sca_refine_step_over_door_knees_up',
        'g1_guided__no_imp_sca__sca_refine_step_over_door_knees_fwd',
        'g1_guided__no_imp_sca__sca_refine_step_over_door_knees_diag',
        'g1_guided__no_imp_sca__sca_refine_step_on_door_knees_up',
        'g1_guided__no_imp_sca__sca_refine_step_on_door_knees_fwd',
        'g1_guided__no_imp_sca__sca_refine_step_on_door_knees_diag',
        'g1_guided__no_imp_sca__sca_refine_step_on_balanced_door_knees_up',
        'g1_guided__no_imp_sca__sca_refine_step_on_balanced_door_knees_fwd',
        'g1_guided__no_imp_sca__sca_refine_step_on_balanced_door_knees_diag',
    ]

    n_plans_short = [
        'Over knees Up',
        'Over knees Fwd',
        'Over knees Diag',
        'On 1, knees Up',
        'On 1, knees Fwd',
        'On 1, knees Diag',
        'On 2, knees Up',
        'On 2, knees Fwd',
        'On 2, knees Diag',
    ]

    kin_dmin_dict = {}
    kin_smooth_dict = {}
    kin_sca_build_dict = {}
    kin_sca_construct_dict = {}
    kin_sca_solve_dict = {}
    dyn_dict = {}
    sca_dict = {}

    for idx, plan in enumerate(n_plans):
        kin_min_d_time = []
        kin_smooth_time = []
        kin_sca_build_time = []  # when using SCA
        kin_sca_construct_time = []  # when using SCA
        kin_sca_solve_time = []  # when using SCA
        dyn_time = []
        sca_time = []

        with open(f'experiment_data/{plan}.pkl', 'rb') as file:
            while True:
                try:
                    d = pickle.load(file)
                    kin_min_d_time.append(d['kin_solver_stats']['min_reach_iris_distance_cvxpy_time'])
                    kin_smooth_time.append(d['kin_solver_stats']['multiple_bezier_iris_cvxpy_time'])
                    kin_sca_build_time.append(d['kin_solver_stats']['multiple_bezier_iris_sca_build_time'])
                    kin_sca_construct_time.append(d['kin_solver_stats']['multiple_bezier_iris_sca_construct_time'])
                    kin_sca_solve_time.append(d['kin_solver_stats']['multiple_bezier_iris_sca_casadi_time'])
                    dyn_time.append(sum(d['dyn_solver_stats']['contacts_phases_solve_times']))
                    sca_time.append(d['dyn_solver_stats']['sca_solve_time'])
                except EOFError:
                    break

        kin_dmin_dict[plan] = kin_min_d_time
        kin_smooth_dict[plan] = kin_smooth_time
        kin_sca_build_dict[plan] = kin_sca_build_time
        kin_sca_construct_dict[plan] = kin_sca_construct_time
        kin_sca_solve_dict[plan] = kin_sca_solve_time
        dyn_dict[plan] = dyn_time
        sca_dict[plan] = sca_time
        planning_time = kin_min_d_time[-1] + kin_smooth_time[-1] + kin_sca_solve_time[-1] + dyn_time[-1] + sca_time[-1]
        print(f'Total planning time for {plan}: {planning_time}')

    #
    # Create stacked bar plots: one stacked bar per plan with all timing components
    #
    x_labels = n_plans
    x = np.arange(len(x_labels))

    # Means for each timing component per plan
    kin_dmin_means = [np.mean(kin_dmin_dict[p]) for p in x_labels]
    kin_smooth_means = [np.mean(kin_smooth_dict[p]) for p in x_labels]
    kin_sca_build_means = [np.mean(kin_sca_build_dict[p]) for p in x_labels]
    kin_sca_construct_means = [np.mean(kin_sca_construct_dict[p]) for p in x_labels]
    kin_sca_solve_means = [np.mean(kin_sca_solve_dict[p]) for p in x_labels]
    dyn_means = [np.mean(dyn_dict[p]) for p in x_labels]
    sca_means = [np.mean(sca_dict[p]) for p in x_labels]

    # Convert to numpy arrays for easy stacking
    a = np.array(kin_dmin_means)
    b = np.array(kin_smooth_means)
    c = np.array(kin_sca_build_means)
    d = np.array(kin_sca_construct_means)
    e = np.array(kin_sca_solve_means)
    f = np.array(dyn_means)
    g = np.array(sca_means)
    s1 = a + b

    fig, ax = plt.subplots()
    # bar1 = ax.bar(x, a, label='kin_min_d')
    bar2 = ax.bar(x, s1, label='Stage 1')
    bar3 = ax.bar(x, c, bottom=s1, label='Stage 2 (build)', color='gray')
    bar4 = ax.bar(x, d, bottom=s1 + c, label='Stage 2 (construct)')
    bar5 = ax.bar(x, e, bottom=s1 + c + d, label='Stage 2 (solve)')
    bar6 = ax.bar(x, f, bottom=s1 + c + d + e, label='Stage 3')
    bar7 = ax.bar(x, g, bottom=s1 + c + d + e + f, label='Full SCA')

    ax.set_xlabel('Locomotion Plan', fontsize=14)
    ax.set_ylabel('Solve Time (s)', fontsize=14)
    ax.set_title('Planner Solve Times Breakdown', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(n_plans_short, rotation=45, ha='right', fontsize=14)
    ax.legend(fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()


if __name__ == '__main__':
    main()