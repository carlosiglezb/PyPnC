import copy
from typing import List

import cvxpy as cp
import numpy as np
from scipy.special import binom

from pnc.planner.multicontact.kin_feasibility.cvx_mfpp_tools import get_aux_frame_idx, create_bezier_cvx_norm_eq_relaxation, \
    add_vel_acc_constr
from pnc.planner.multicontact.path_parameterization import BezierCurve, CompositeBezierCurve
from vision.iris.iris_regions_manager import IrisRegionsManager

from pnc.planner.multicontact.kin_feasibility.test.test_updater import reach_updater

def optimize_multiple_bezier_iris(reach_region: dict[str: np.array, str: np.array],
                                  aux_frames: List[dict],
                                  iris_regions: dict[str: IrisRegionsManager],
                                  durations: List[dict[str, np.array]],
                                  alpha: dict[int: float],
                                  safe_points_lst: List[dict[str, np.array]],
                                  fixed_frames=None,
                                  surface_normals_lst=None,
                                  weights_rigid_link=None,
                                  n_points=None, **kwargs):
    stance_foot = 'RF'

    if weights_rigid_link is None:
        weights_rigid_link = np.array([3500., 0.5, 10.])     # default for g1

    # number of frames
    n_frames = len(safe_points_lst[0].keys())

    # Problem size. Assume for now same number of boxes for all frames
    first_fr_iris = next(iter(iris_regions.values()))
    d = first_fr_iris.iris_list[0].iris_region.ambient_dimension()
    num_iris_tot = 0
    for seg_dur in durations:
        num_iris_tot += len(seg_dur[next(iter(seg_dur))])
    D = max(alpha)

    # default number of points for Bezier curve
    if n_points is None:
        n_points = (D + 1) * 2

    # Control points of the curves and their derivatives.
    points = {}
    for k in range(num_iris_tot * n_frames):
        points[k] = {}
        for i in range(D + 1):
            size = (n_points - i, d)
            points[k][i] = cp.Variable(size)

    frame_list = list(safe_points_lst[0].keys())
    constraints = []

    # Loop through boxes.
    cost = 0
    continuity = {}
    frame_idx, fr_seg_k_box = 0, 0
    seg_idx, k = 0, 0
    for k in range(num_iris_tot * n_frames):
        continuity[k] = {}

        # Update frame name and number of boxes within segment/interval
        f_name = frame_list[frame_idx]
        sequenced_idx = iris_regions[f_name].iris_idx_seq[seg_idx][fr_seg_k_box]
        A = iris_regions[f_name].iris_list[sequenced_idx].iris_region.A()
        b = iris_regions[f_name].iris_list[sequenced_idx].iris_region.b()
        b = np.reshape(b, (len(b), 1))
        b = np.repeat(b, n_points, axis=1)
        constraints.append(A @ points[k][0].T <= b)
        num_iris_current = len(iris_regions[f_name].iris_idx_seq[seg_idx])

        # Enforce given positions
        if k % num_iris_tot == 0:          # initial position for each frame
            # if also a fixed frame, repeat for entire segment duration
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points, axis=0)
                constraints.append(points[k][0] == fixed_frame_pos_mat)
            else:   # assign for just the first time instant
                constraints.append(points[k][0][0] == safe_points_lst[0][f_name])   # initial position
                # check if it has a final safe point assigned
                if fr_seg_k_box == (num_iris_current-1) and f_name in safe_points_lst[seg_idx+1].keys():
                    constraints.append(points[k][0][-1] == safe_points_lst[seg_idx+1][f_name])
                    add_vel_acc_constr(f_name, surface_normals_lst[seg_idx], points[k], constraints)
        elif (k + 1) % num_iris_tot == 0:  # final position for each frame
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points-1, axis=0)
                constraints.append(points[k][0][1:] == fixed_frame_pos_mat)
            else:
                constraints.append(points[k][0][-1] == safe_points_lst[-1][f_name])
                add_vel_acc_constr(f_name, surface_normals_lst[-1], points[k], constraints)
        else:       # safe and fixed positions at other times
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points-1, axis=0)
                constraints.append(points[k][0][1:] == fixed_frame_pos_mat)
            # Check if safe_point is available for the current frame
            elif f_name in safe_points_lst[seg_idx+1].keys():
                # Enforce (pre-computed) safe points at the end of each desired motion
                # note: the initial point within a segment is defined by the continuity constraint below
                if fr_seg_k_box == (num_iris_current-1):
                    constraints.append(points[k][0][-1] == safe_points_lst[seg_idx+1][f_name])  # pos
                    add_vel_acc_constr(f_name, surface_normals_lst[seg_idx], points[k], constraints)

        # Bezier dynamics.
        for i in range(D):
            h = n_points - i - 1
            ci = durations[seg_idx][f_name][fr_seg_k_box] / h
            constraints.append(points[k][i][1:] - points[k][i][:-1] == ci * points[k][i + 1])

        # if we are in the same frame, enforce dynamics, continuity, differentiability, and cost
        if (k+1) % num_iris_tot != 0:
            # Continuity and differentiability.
            if fr_seg_k_box < num_iris_current:
                for i in range(D + 1):
                    constraints.append(points[k][i][-1] == points[k + 1][i][0])
                    if i > 0:
                        continuity[k][i] = constraints[-1]

        # Cost function
        for i, ai in alpha.items():
            h = n_points - 1 - i
            A = np.zeros((h + 1, h + 1))
            for m in range(h + 1):
                for n in range(h + 1):
                    A[m, n] = binom(h, m) * binom(h, n) / binom(2 * h, m + n)
            A *= durations[seg_idx][f_name][fr_seg_k_box] / (2 * h + 1)
            A = np.kron(A, np.eye(d))
            p = cp.vec(points[k][i], order='C')
            cost += ai * cp.quad_form(p, A)

        # Adjust frame name, segment and box numbers
        if (k+1) % num_iris_tot == 0:
            frame_idx += 1
            seg_idx = 0
            fr_seg_k_box = 0
        else:           # move to next segment if this is the last box
            if fr_seg_k_box == (num_iris_current - 1):   # or (k % num_iris_current == 0)
                fr_seg_k_box = 0        # reset the box count
                seg_idx += 1            # increase segment
            else:
                fr_seg_k_box += 1

    # Rigid links (e.g., shin link length) constraint relaxation
    soc_constraint, cost_log_abs = [], []
    cost_log_abs_sum = 0.
    if bool(aux_frames):     # check if empy dictionary
        # apply auxiliary rigid link constraint throughout all safe regions
        for aux_fr in aux_frames:
            prox_fr_idx, dist_fr_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot)

            # loop through all safe boxes
            for nb in range(1, num_iris_tot-1):
                for pnt in range(n_points-1):
                # for pnt in range(1):
                    link_proximal_point = points[prox_fr_idx+nb][0][pnt]
                    link_distal_point = points[dist_fr_idx+nb][0][pnt]
                    create_bezier_cvx_norm_eq_relaxation(link_length, link_proximal_point,
                                             link_distal_point, soc_constraint, cost_log_abs,
                                                         wi=weights_rigid_link)

        cost_log_abs_sum = -(cp.sum(cost_log_abs))

    # Reachability constraints
    if reach_region is not None:
        k_fr_iris = 0
        for fr_idx, frame_name in enumerate(frame_list):
            fr_iris_counter = 0
            for seg in range(len(durations)):

                num_iris_current = len(iris_regions[frame_name].iris_idx_seq[seg])
                for si in range(num_iris_current):
                    z_t = points[0 * num_iris_tot + fr_iris_counter + si][0]

                    if frame_name == 'torso':
                        # get corresponding ee frame points to consider for stance reachability constraint
                        if seg < 1:         # stance_foot = 'RF'
                            rf_idx = 2      # RF is 3rd in the list
                            coeffs = reach_region['RF']
                            z_ee_seg = points[rf_idx * num_iris_tot + fr_iris_counter + si][0]
                        else:               # stance_foot = 'LF'
                            lf_idx = 1      # LF is 2nd in the list
                            coeffs = reach_region['LF']
                            z_ee_seg = points[lf_idx * num_iris_tot + fr_iris_counter + si][0]

                    else:
                        coeffs = reach_region[frame_name]
                        z_ee_seg = points[fr_idx * num_iris_tot + fr_iris_counter + si][0]

                    # reachable constraint
                    H = coeffs['H']
                    d_vec = np.reshape(coeffs['d'], (len(H), 1))
                    d_mat = np.repeat(d_vec, n_points, axis=1)
                    if frame_name != 'torso':
                        constraints.append(H @ (z_ee_seg.T - z_t.T) <= -d_mat)

                fr_iris_counter += num_iris_current
                k_fr_iris += num_iris_current

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost + cost_log_abs_sum), constraints + soc_constraint)
    prob.solve(solver='CLARABEL')

    if prob.status == 'infeasible':
        print('Problem was infeasible with CLARABEL solver. Retrying with relaxed SCS.')
        prob.solve(solver='SCS', eps_rel=1e-1, eps_abs=1e-1)

    # Reconstruct trajectory.
    beziers, path = [], []
    a = 0
    fr_seg_k_box, frame_idx, seg_idx = 0, 0, 0
    frame_name = frame_list[frame_idx]
    for k in range(num_iris_tot * n_frames):
        val = points[k][0].value

        num_iris_current = len(iris_regions[frame_name].iris_idx_seq[seg_idx])
        # move on to next segment after the current number of safe boxes
        if (fr_seg_k_box != 0) and fr_seg_k_box % num_iris_current == 0 and seg_idx != (num_iris_tot-1):
            seg_idx += 1
            fr_seg_k_box = 0

        # move on to next frame after all boxes processed for each frame
        if k != 0 and (k % num_iris_tot) == 0:
            frame_idx += 1
            frame_name = frame_list[frame_idx]
            fr_seg_k_box = 0

        b = a + durations[seg_idx][frame_name][fr_seg_k_box]
        beziers.append(BezierCurve(points[k][0].value, a, b))
        a = b
        fr_seg_k_box += 1
        # skip the final positions, those are assigned later
        if (k + 1) % num_iris_tot == 0:
            fr_seg_k_box = 0  # might be redundant
            seg_idx = 0
            path.append(copy.deepcopy(CompositeBezierCurve(beziers)))
            beziers.clear()
            a = 0

    retiming_weights = {}

    # Reconstruct costs.
    cost_breakdown = {}

    # Solution statistics.
    sol_stats = {}
    sol_stats['cost'] = prob.value
    sol_stats['runtime'] = prob.solver_stats.solve_time
    sol_stats['cost_breakdown'] = cost_breakdown
    sol_stats['retiming_weights'] = retiming_weights

    return path, sol_stats, points
