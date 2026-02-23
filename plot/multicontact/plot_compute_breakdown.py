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
    sca_plans = [
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

    sca_plans_short = [
        'Up',
        'Fwd',
        'Diag',
        'Up',
        'Fwd',
        'Diag',
        'Up',
        'Fwd',
        'Diag',
    ]

    kin_dmin_dict = {}
    kin_smooth_dict = {}
    kin_sca_build_dict = {}
    kin_sca_construct_dict = {}
    kin_sca_solve_dict = {}
    dyn_dict = {}
    sca_dict = {}

    for idx, plan in enumerate(sca_plans):
        kin_min_d_time = []
        kin_smooth_time = []
        kin_sca_build_time = []  # when using SCA
        kin_sca_construct_time = []  # when using SCA
        kin_sca_solve_time = []  # when using SCA
        dyn_time = []
        sca_time = []

        with open(f'experiment_data/RAL/DoorShort/{plan}.pkl', 'rb') as file:
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
    #  Load solution with NO KIN-SCA
    #
    mfpp_plans = [
        'g1_guided__no_imp__sca_refine_step_over_door_knees_up',
        'g1_guided__no_imp__sca_refine_step_over_door_knees_fwd',
        'g1_guided__no_imp__sca_refine_step_over_door_knees_diag',
        'g1_guided__no_imp__sca_refine_step_on_door_knees_up',
        'g1_guided__no_imp__sca_refine_step_on_door_knees_fwd',
        'g1_guided__no_imp__sca_refine_step_on_door_knees_diag',
        'g1_guided__no_imp__sca_refine_step_on_balanced_door_knees_up',
        'g1_guided__no_imp__sca_refine_step_on_balanced_door_knees_fwd',
        'g1_guided__no_imp__sca_refine_step_on_balanced_door_knees_diag',
    ]

    mfpp_kin_dmin_dict = {}
    mfpp_kin_smooth_dict = {}
    mfpp_dyn_dict = {}
    mfpp_sca_dict = {}

    for idx, plan in enumerate(mfpp_plans):
        mfpp_kin_min_d_time = []
        mfpp_kin_smooth_time = []
        mfpp_dyn_time = []
        mfpp_sca_time = []

        with open(f'experiment_data/RAL/Comparisons/MFPP_single_full-SCA/{plan}.pkl', 'rb') as file:
            while True:
                try:
                    d = pickle.load(file)
                    mfpp_kin_min_d_time.append(d['kin_solver_stats']['min_reach_iris_distance_cvxpy_time'])
                    mfpp_kin_smooth_time.append(d['kin_solver_stats']['multiple_bezier_iris_cvxpy_time'])
                    mfpp_dyn_time.append(sum(d['dyn_solver_stats']['contacts_phases_solve_times']))
                    mfpp_sca_time.append(d['dyn_solver_stats']['sca_solve_time'])
                except EOFError:
                    break

        mfpp_kin_dmin_dict[plan] = mfpp_kin_min_d_time
        mfpp_kin_smooth_dict[plan] = mfpp_kin_smooth_time
        mfpp_dyn_dict[plan] = mfpp_dyn_time
        mfpp_sca_dict[plan] = mfpp_sca_time
        if idx == 6:
            mfpp_sca_dict[plan] = [0]
        planning_time = mfpp_kin_min_d_time[-1] + mfpp_kin_smooth_time[-1] + mfpp_dyn_time[-1] + mfpp_sca_time[-1]
        # planning_time = kin_min_d_time[-1] + kin_smooth_time[-1] + kin_sca_solve_time[-1] + dyn_time[-1] + sca_time[-1]
        print(f'Total planning time for {plan}: {planning_time}')


    #
    # Create stacked bar plots: one stacked bar per plan with all timing components
    #
    x_labels = sca_plans
    x = np.arange(len(x_labels))

    # Timing component per SCA plan
    kin_dmin_means = [kin_dmin_dict[p] for p in x_labels]
    kin_smooth_means = [kin_smooth_dict[p] for p in x_labels]
    kin_sca_build_means = [kin_sca_build_dict[p] for p in x_labels]
    kin_sca_construct_means = [kin_sca_construct_dict[p] for p in x_labels]
    kin_sca_solve_means = [kin_sca_solve_dict[p] for p in x_labels]
    dyn_means = [dyn_dict[p] for p in x_labels]
    sca_means = [sca_dict[p] for p in x_labels]

    # Convert to numpy arrays for easy stacking
    a1 = np.array(kin_dmin_means)
    b1 = np.array(kin_smooth_means)
    c1 = np.array(kin_sca_build_means)
    d1 = np.array(kin_sca_construct_means)
    e1 = np.array(kin_sca_solve_means)
    f1 = np.array(dyn_means)
    g1 = np.array(sca_means)
    s1 = a1 + b1

    # Timing component per MFPP plan
    mfpp_kin_dmin_means = [mfpp_kin_dmin_dict[p] for p in mfpp_plans]
    mfpp_kin_smooth_means = [mfpp_kin_smooth_dict[p] for p in mfpp_plans]
    mfpp_dyn_means = [mfpp_dyn_dict[p] for p in mfpp_plans]
    mfpp_sca_means = [mfpp_sca_dict[p] for p in mfpp_plans]

    a2 = np.array(mfpp_kin_dmin_means)
    b2 = np.array(mfpp_kin_smooth_means)
    f2 = np.array(mfpp_dyn_means)
    g2 = np.array(mfpp_sca_means)
    s2 = a2 + b2

    # ----------------
    # # 2D bar plots for SCA plans
    # fig, ax_sca = plt.subplots()
    # sca_bar1 = ax_sca.bar(x, a1, label='kin_min_d')
    # sca_bar2 = ax_sca.bar(x, s1, label='Stage 1')
    # # sca_bar3 = ax_sca.bar(x, c1, bottom=s1, label='Stage 2 (build)', color='gray')
    # # sca_bar4 = ax_sca.bar(x, d1, bottom=s1 + c, label='Stage 2 (construct)')
    # sca_bar5 = ax_sca.bar(x, e1, bottom=s1, label='Stage 2')     # (solve)
    # sca_bar6 = ax_sca.bar(x, f1, bottom=s1 + e1, label='Stage 3 (dyn)')
    # sca_bar7 = ax_sca.bar(x, g1, bottom=s1 + e1 + f1, label='Stage 3 (full SCA)')
    #
    # # 2D bar plots for MFPP plans
    # fig, ax_mfpp = plt.subplots()
    # mfpp_bar1 = ax_mfpp.bar(x, a2, label='kin_min_d')
    # mfpp_bar2 = ax_mfpp.bar(x, s2, label='Stage 1')
    # mfpp_bar6 = ax_mfpp.bar(x, f2, bottom=s2, label='Stage 3 (dyn)', color='#2ca02c')
    # mfpp_bar7 = ax_mfpp.bar(x, g2, bottom=s2 + f2, label='Stage 3 (full SCA)', color='#d62728')
    # ax.set_xlabel('Locomotion Plan', fontsize=14)
    # ax.set_ylabel('Solve Time (s)', fontsize=14)
    # ax.set_title('Planner Solve Times Breakdown', fontsize=14)
    # ax.set_xticks(x)
    # ax.set_xticklabels(sca_plans_short, rotation=45, ha='right', fontsize=14)
    # ax.legend(fontsize=14)
    # plt.tick_params(axis='both', which='major', labelsize=14)
    # plt.show()
    # -----------

    # Prepare 3D Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_labels = sca_plans_short
    x_pos = np.arange(len(x_labels))  # [0, 1, 2, ... 8]
    y_pos = np.array([0, 1])  # 0 for SCA, 1 for MFPP

    # Bar width and depth
    width = 0.8
    depth = 0.1

    # Colors for consistency
    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c', '#d62728']
    labels = ['Kin Min D', 'Stage 1', 'Stage 2', 'Stage 3 (Dyn)', 'Stage 3 (Full SCA)']

    def plot_stacked_3d_bars(y_val, data_layers, bar_label_prefix):
        """Helper to stack bars along the Z axis at a specific Y position."""
        bottom = np.zeros(len(x_pos))
        for i, layer in enumerate(data_layers):
            # layer is a 1D array of values for each x_pos
            flattened_layer = layer.flatten()
            ax.bar3d(x_pos - width / 2, np.full_like(x_pos, y_val) - depth / 2, bottom,
                     width, depth, flattened_layer,
                     color=colors[i], alpha=0.8)
            bottom += flattened_layer

    # --- Plot SCA Bars at Y = 0 ---
    # Stack: a1 (min_d), b1 (stage1), e1 (stage2), f1 (dyn), g1 (sca)
    plot_stacked_3d_bars(0, [a1, b1, e1, f1, g1], "SCA")

    # --- Plot MFPP Bars at Y = 1 ---
    # Stack: a2 (min_d), b2 (stage1), zero_layer (no stage 2), f2 (dyn), g2 (sca)
    zero_layer = np.zeros_like(a2)
    plot_stacked_3d_bars(1, [a2, b2, zero_layer, f2, g2], "MFPP")

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(['SCA Plans', 'MFPP Plans'])

    ax.set_xlabel('Plan Type', fontsize=12, labelpad=10)
    ax.set_ylabel('Method', fontsize=12, labelpad=10)
    ax.set_zlabel('Solve Time (s)', fontsize=12)
    ax.set_title('3D Breakdown of Planner Solve Times', fontsize=14)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, lw=4, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()