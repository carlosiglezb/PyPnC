import copy
from typing import List

import casadi as ca
import cvxpy as cp
import numpy as np
from casadi import nlpsol
from scipy.special import binom
from scipy.optimize import minimize

from pnc.planner.multicontact.kin_feasibility.casadi_ocp_constraints.casadi_ocp_functions import \
    DColIndexedPolytopesConstraint
from pnc.planner.multicontact.kin_feasibility.cvx_mfpp_tools import get_aux_frame_idx, \
    create_bezier_cvx_norm_eq_relaxation, add_vel_acc_constr
from pnc.planner.multicontact.kin_feasibility.scipy_ocp_constraints.scipy_ocp_functions import \
    LinearBezierIneqConstraint, LinearBezierEqConstraint
from pnc.planner.multicontact.path_parameterization import BezierCurve, CompositeBezierCurve
from vision.iris.iris_regions_manager import IrisRegionsManager


def optimize_multiple_bezier_iris(reach_region: dict[str: np.array, str: np.array],
                                  aux_frames: List[dict],
                                  iris_regions: dict[str: IrisRegionsManager],
                                  durations: List[dict[str, np.array]],
                                  alpha: dict[int: float],
                                  safe_points_lst: List[dict[str, np.array]],
                                  fixed_frames=None,
                                  contact_sequence=None,
                                  surface_normals_lst=None,
                                  weights_rigid_link=None,
                                  n_points=None, **kwargs):
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
        link_threshold = 0.05
        # apply auxiliary rigid link constraint throughout all safe regions
        for aux_fr in aux_frames:
            prox_fr_idx, dist_fr_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot)

            # loop through all safe boxes
            link_length += link_threshold     # threshold for relaxation
            for nb in range(1, num_iris_tot-1):
                # for pnt in range(n_points-1):
                for pnt in range(1):
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
                        continue

                    else:
                        coeffs = reach_region[frame_name]
                        z_ee_seg = points[fr_idx * num_iris_tot + fr_iris_counter + si][0]

                    # reachable constraint
                    H = coeffs['H']
                    d_vec = np.reshape(coeffs['d'], (len(H), 1))
                    d_mat = np.repeat(d_vec, n_points, axis=1)
                    if frame_name == 'torso' or frame_name == 'LF' or frame_name == 'RF':
                        constraints.append(H @ (z_ee_seg.T - z_t.T) <= -d_mat)
                    # constraints.append(H @ (z_ee_seg.T - z_t.T) <= -d_mat)

                fr_iris_counter += num_iris_current
                k_fr_iris += num_iris_current

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost + cost_log_abs_sum), constraints + soc_constraint)
    prob.solve(solver='SCS')

    if prob.status == 'infeasible':
        print('***** Problem was infeasible with CLARABEL solver. Retrying with relaxed SCS.')
        prob.solve(solver='SCS', eps_rel=1e-1, eps_abs=1e-1)

    # check link constraints values
    if bool(aux_frames):     # check if empy dictionary
        # apply auxiliary rigid link constraint throughout all safe regions
        for aux_fr in aux_frames:
            prox_fr_idx, dist_fr_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot)

            # loop through all safe boxes
            link_length += link_threshold     # threshold for relaxation
            for nb in range(1, num_iris_tot-1):
                # for pnt in range(n_points-1):
                for pnt in range(1):
                    link_proximal_point = points[prox_fr_idx+nb][0][pnt]
                    link_distal_point = points[dist_fr_idx+nb][0][pnt]
                    print(f"{aux_fr['parent_frame']} Link length discrepancy: {np.linalg.norm(link_proximal_point.value - link_distal_point.value) - link_length}")

    # Reconstruct trajectory.
    beziers, path = [], []
    a = 0
    fr_seg_k_box, frame_idx, seg_idx = 0, 0, 0
    frame_name = frame_list[frame_idx]
    for k in range(num_iris_tot * n_frames):
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
    dual_vars = {}
    dual_vars['lam_g0'] = prob.solution.dual_vars

    return path, sol_stats, points, dual_vars

def unpack_sol_to_points(x_sol, num_iris_all_frames, n_points, D, d):
    sol_points = [None] * num_iris_all_frames
    for k in range(num_iris_all_frames):
        sol_points[k] = [None] * (D + 1)
    curr_idx = 0
    for k in range(num_iris_all_frames):
        for i in range(D + 1):
            next_idx = curr_idx + (n_points - i) * d
            sol_points[k][i] = x_sol[curr_idx:next_idx].reshape(n_points-i, d, order='F')
            curr_idx = next_idx

    return sol_points


def pack_points_to_single_vector(points, vec_type: str):
    """
    Casadi's reshape method follows a column-major order, so we reshape the matrices
    accordingly to output:
    x = [p0_x, p0_y, p0_z, v0_x, v0_y, v0_z, ..., p1_x, p1_y, p1_z, v1_x, v1_y, v1_z, ...]
    """
    num_iris_traversed = len(points)
    num_points = points[0][0].shape[0]
    alpha_deg = len(points[0])
    d = alpha_deg - 1
    vector_out = []
    for ir in range(num_iris_traversed):
        for i in range(alpha_deg):
            vec_size = (num_points - i) * d
            if vec_type == 'casadi':
                transposed_mat = ca.reshape(points[ir][i], 3, num_points - i)
                vector_out = ca.vertcat(vector_out, ca.reshape(transposed_mat, vec_size, 1))
            elif vec_type == 'cxvpy':
                parsed_vec = np.reshape(points[ir][i].value, (vec_size, 1), order='C')
                vector_out = np.concatenate((vector_out, *parsed_vec))
            elif vec_type == 'numpy':
                parsed_vec = np.reshape(points[ir][i], (vec_size, 1), order='C')
                vector_out = np.concatenate((vector_out, *parsed_vec))
            else:
                raise ValueError('Invalid vector type specified. Use either casadi or numpy.')

    return vector_out


def optimize_multiple_bezier_iris_casadi(reach_region: dict[str: np.array, str: np.array],
                                  aux_frames: List[dict],
                                  iris_regions: dict[str: IrisRegionsManager],
                                  durations: List[dict[str, np.array]],
                                  alpha: dict[int: float],
                                  safe_points_lst: List[dict[str, np.array]],
                                  robot_geom_data: dict[str: np.array]=None,
                                  fixed_frames=None,
                                  contact_sequence=None,
                                  surface_normals_lst=None,
                                  initial_guess=None,
                                  weights_rigid_link=None,
                                  n_points=None, **kwargs):
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
            points[k][i] = ca.MX.sym("p" + str(k), size[0], size[1])

    frame_list = list(safe_points_lst[0].keys())
    constraints = []
    lbg = []
    ubg = []

    # Loop through IRIS regions
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
        parse_mat_leq_constr(A, b, points[k], constraints, lbg, ubg)
        num_iris_current = len(iris_regions[f_name].iris_idx_seq[seg_idx])

        # Enforce given positions
        if k % num_iris_tot == 0:          # initial position for each frame
            # if also a fixed frame, repeat for entire segment duration
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                parse_repvec_eq_constr(np.array([safe_points_lst[seg_idx][f_name]]), points[k][0], constraints, lbg, ubg)
            else:   # assign for just the first time instant
                parse_vec_eq_constr(safe_points_lst[0][f_name], points[k][0][0,:], constraints, lbg, ubg)
                # check if it has a final safe point assigned
                if fr_seg_k_box == (num_iris_current-1) and f_name in safe_points_lst[seg_idx+1].keys():
                    parse_vec_eq_constr(safe_points_lst[seg_idx+1][f_name], points[k][0][-1,:], constraints, lbg, ubg)
                    # TODO add vel constraint
                    # add_vel_acc_constr(f_name, surface_normals_lst[seg_idx], points[k], constraints)
        elif (k + 1) % num_iris_tot == 0:  # final position for each frame
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                parse_repvec_eq_constr(np.array([safe_points_lst[seg_idx][f_name]]), points[k][0][1:, :], constraints, lbg, ubg)
            else:
                parse_vec_eq_constr(safe_points_lst[-1][f_name], points[k][0][-1,:], constraints, lbg, ubg)
                # TODO add vel constraint
                # add_vel_acc_constr(f_name, surface_normals_lst[-1], points[k], constraints)
        else:       # safe and fixed positions at other times
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                parse_repvec_eq_constr(np.array([safe_points_lst[seg_idx][f_name]]), points[k][0][1:, :], constraints, lbg, ubg)
            # Check if safe_point is available for the current frame
            elif f_name in safe_points_lst[seg_idx+1].keys():
                # Enforce (pre-computed) safe points at the end of each desired motion
                # note: the initial point within a segment is defined by the continuity constraint below
                if fr_seg_k_box == (num_iris_current-1):
                    parse_vec_eq_constr(safe_points_lst[seg_idx+1][f_name], points[k][0][-1,:], constraints, lbg, ubg)
                    # TODO add vel contraint
                    # add_vel_acc_constr_casadi(f_name, surface_normals_lst[seg_idx], points[k], constraints, lbg, ubg)

        # Bezier dynamics.
        for i in range(D):
            h = n_points - i - 1
            ci = durations[seg_idx][f_name][fr_seg_k_box] / h
            parse_mat_eq_constr(ci * points[k][i + 1], points[k][i][1:, :] - points[k][i][:-1, :], constraints, lbg, ubg)

        # if we are in the same frame, enforce dynamics, continuity, differentiability, and cost
        if (k+1) % num_iris_tot != 0:
            # Continuity and differentiability.
            if fr_seg_k_box < num_iris_current:
                for i in range(D + 1):
                    parse_vec_eq_constr(points[k + 1][i][0,:], points[k][i][-1,:].T, constraints, lbg, ubg)
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
            p = ca.vec(points[k][i].T)
            cost += ai * ca.bilin(A, p, p)

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
        link_threshold = 0.05
        # apply auxiliary rigid link constraint throughout all safe regions
        for aux_fr in aux_frames:
            prox_fr_idx, dist_fr_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot)

            # loop through all safe boxes
            link_length += link_threshold     # threshold for relaxation
            for nb in range(1, num_iris_tot-1):
                # for pnt in range(n_points-1):
                for pnt in range(1):
                    link_proximal_point = points[prox_fr_idx+nb][0][pnt]
                    link_distal_point = points[dist_fr_idx+nb][0][pnt]
                    create_bezier_cvx_norm_eq_relaxation(link_length, link_proximal_point,
                                                         link_distal_point, soc_constraint, cost_log_abs,
                                                         wi=weights_rigid_link)

        cost_log_abs_sum = -(cp.sum(cost_log_abs))

    # Reachability constraints
    # if reach_region is not None:
    #     k_fr_iris = 0
    #     for fr_idx, frame_name in enumerate(frame_list):
    #         fr_iris_counter = 0
    #         for seg in range(len(durations)):
    #
    #             num_iris_current = len(iris_regions[frame_name].iris_idx_seq[seg])
    #             for si in range(num_iris_current):
    #                 z_t = points[0 * num_iris_tot + fr_iris_counter + si][0]
    #
    #                 if frame_name == 'torso':
    #                     continue
    #
    #                 else:
    #                     coeffs = reach_region[frame_name]
    #                     z_ee_seg = points[fr_idx * num_iris_tot + fr_iris_counter + si][0]
    #
    #                 # reachable constraint
    #                 H = coeffs['H']
    #                 d_vec = np.reshape(coeffs['d'], (len(H), 1))
    #                 d_mat = np.repeat(d_vec, n_points, axis=1)
    #                 if frame_name == 'torso' or frame_name == 'LF' or frame_name == 'RF':
    #                     constraints.append(H @ (z_ee_seg.T - z_t.T) <= -d_mat)
    #                 # constraints.append(H @ (z_ee_seg.T - z_t.T) <= -d_mat)
    #
    #             fr_iris_counter += num_iris_current
    #             k_fr_iris += num_iris_current

    # Collect points into a single vector
    points_all = pack_points_to_single_vector(points, 'casadi')

    opts = {
        "ipopt": {
            "hessian_approximation": "exact",   # limited-memory
            "max_iter": 100,
            "mu_init": 1e-5,
            "tol": 1e-3,
            # "derivative_test": "first-order",
            # "derivative_test_print_all": "no",
            # "derivative_test_perturbation": 1e-6,
            # "derivative_test_tol": 0.0005
        }
    }

    sca_constraints = []
    if robot_geom_data is not None:
        # Simplified no self-collision function and constraints bounds for specified index pairs
        mfpp_bezier_data = {'current_frames': (0, 1),   # make torso and RK frames SCA
                            'n_points': n_points,
                            'num_derivatives': D,
                            'num_iris_per_frame': num_iris_tot,
                            'num_frames': n_frames     # 2
                            }
        f_dist = {}
        sca_bez_points = range(0, num_iris_tot * n_points, 1)
        for i in sca_bez_points:
            mfpp_bezier_data['current_point'] = i
            i_name = 'f_dist'+ str(i)
            current_mfpp_data = copy.deepcopy(mfpp_bezier_data)
            f_dist[i_name] = DColIndexedPolytopesConstraint(i_name, robot_geom_data, current_mfpp_data)
            sca_constraints.append(f_dist[i_name](points_all))
            lbg.append(1.0)
            ubg.append(ca.inf)

        # assume lagrange multipliers of SCA constraints are zero
        initial_guess['lam_g0'] = np.vstack((initial_guess['lam_g0'], np.zeros((len(sca_bez_points),1))))

        opts["ipopt"]["warm_start_init_point"] = "yes"
        opts["ipopt"]["warm_start_mult_bound_push"] = 1e-6
        opts["ipopt"]["warm_start_slack_bound_push"] = 1e-6
        opts["ipopt"]["warm_start_bound_push"] = 1e-6

        # Solve problem
    nlp = {'x': points_all,
           'f': cost + cost_log_abs_sum,
           'g': ca.vertcat(*constraints, *sca_constraints)
           }
    solver = nlpsol('solver', 'ipopt', nlp, opts)

    if initial_guess is not None:
        sol = solver(x0=initial_guess['x0'],
                     lam_g0=initial_guess['lam_g0'],
                     lam_x0=initial_guess['lam_x0'],
                     lbg=lbg, ubg=ubg)
    else:
        sol = solver(lbg=lbg, ubg=ubg)

    x_sol = sol['x'].full()

    sol_points = unpack_sol_to_points(x_sol, num_iris_tot * n_frames, n_points, D, d)

    # if prob.status == 'infeasible':
    #     print('***** Problem was infeasible with CLARABEL solver. Retrying with relaxed SCS.')
    #     prob.solve(solver='SCS', eps_rel=1e-1, eps_abs=1e-1)

    # check link constraints values
    if bool(aux_frames):     # check if empy dictionary
        # apply auxiliary rigid link constraint throughout all safe regions
        for aux_fr in aux_frames:
            prox_fr_idx, dist_fr_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot)

            # loop through all safe boxes
            link_length += link_threshold     # threshold for relaxation
            for nb in range(1, num_iris_tot-1):
                # for pnt in range(n_points-1):
                for pnt in range(1):
                    link_proximal_point = points[prox_fr_idx+nb][0][pnt]
                    link_distal_point = points[dist_fr_idx+nb][0][pnt]
                    print(f"{aux_fr['parent_frame']} Link length discrepancy: {np.linalg.norm(link_proximal_point.value - link_distal_point.value) - link_length}")

    # Reconstruct trajectory.
    beziers, path = [], []
    a = 0
    fr_seg_k_box, frame_idx, seg_idx = 0, 0, 0
    frame_name = frame_list[frame_idx]
    for k in range(num_iris_tot * n_frames):
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
        beziers.append(BezierCurve(sol_points[k][0], a, b))
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
    # sol_stats_all = sol.stats()
    sol_stats = {}
    # sol_stats['cost'] = prob.value
    # sol_stats['runtime'] = sol_stats_all['t_wall_total']
    # sol_stats['cost_breakdown'] = cost_breakdown
    # sol_stats['retiming_weights'] = retiming_weights
    dual_vars = {'lam_g0': sol['lam_g'].full(),
                 'lam_x0': sol['lam_x'].full()}

    return path, sol_stats, sol_points, dual_vars


def get_ci_from_global_ir_idx(total_iris_regions, k_ir, durations, f_name, h):
    # find the current iris index in the global list
    c_seq, curr_ir_num = 0, 0
    for k in range(total_iris_regions):
        if k == k_ir:
            # we have arrived at the correct iris region
            return durations[c_seq][f_name][curr_ir_num] / h
        else:
            # we have to increase both counters accordingly
            if curr_ir_num < len(durations[c_seq][f_name]) - 1:
                curr_ir_num += 1
            else:
                c_seq += 1
                curr_ir_num = 0

    return durations[c_seq][f_name][curr_ir_num] / h


def optimize_multiple_sca_bezier_iris(reach_region: dict[str: np.array, str: np.array],
                                  aux_frames: List[dict],
                                  iris_regions: dict[str: IrisRegionsManager],
                                  durations: List[dict[str, np.array]],
                                  alpha: dict[int: float],
                                  safe_points_lst: List[dict[str, np.array]],
                                  fixed_frames=None,
                                  contact_sequence=None,
                                  surface_normals_lst=None,
                                  weights_rigid_link=None,
                                  n_points=None, **kwargs):
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
    frame_list = list(safe_points_lst[0].keys())

    # initialize size of optimization variable
    # x_dim = 3 * num_iris_tot * n_frames * (n_points - 1) * (D + 1)
    x_dim = 3 * num_iris_tot * n_frames * (4 * n_points - 6)

    # initialize size of collision-free (IRIS) constraints in OCP
    num_iris_halfspaces = 0
    for f_name in frame_list:
        for c_seq in range(len(iris_regions[f_name].iris_idx_seq)):
            for curr_ir_num in iris_regions[f_name].iris_idx_seq[c_seq]:
                # ------------- collision-free constraint (inequality)
                num_iris_halfspaces += iris_regions[f_name].iris_list[curr_ir_num].iris_region.A().shape[0]

    # ------------- Bezier dynamics constraint (equality)
    num_bezier_eq_constraints_dyn = (D * n_points - 6) * (num_iris_tot * n_frames) * 3
    # ------------- Bezier continuous and differentiably continuous constraint (equality)
    num_bezier_eq_constraints_cont = (n_points - 1) * n_frames * (D + 1) * 3
    num_bezier_eq_constraints_tot = num_bezier_eq_constraints_dyn + num_bezier_eq_constraints_cont

    bez_lin_ineq_constraints = LinearBezierIneqConstraint(x_dim, num_iris_halfspaces * n_points)

    # Populate constraints
    x_idx = 0       # index of corresponding optimization variable, x
    for f_name in frame_list:
        for c_seq in range(len(iris_regions[f_name].iris_idx_seq)):
            for curr_ir_num in iris_regions[f_name].iris_idx_seq[c_seq]:

                # ------------- collision-free constraint
                A_safe = iris_regions[f_name].iris_list[curr_ir_num].iris_region.A()
                b_safe = iris_regions[f_name].iris_list[curr_ir_num].iris_region.b()

                # apply to all control points
                # constraints.append(points[k][i][1:] - points[k][i][:-1] == ci * points[k][i + 1])
                for n_cp in range(n_points):
                    bez_lin_ineq_constraints.add_lin_ineq(A_safe, b_safe, x_idx)
                    x_idx += 3


    # get all constraints
    collision_free_constraints = bez_lin_ineq_constraints.get_constraints()

    # Initialize constant matrices in equality constraints
    bez_lin_eq_constraints = LinearBezierEqConstraint(x_dim, num_bezier_eq_constraints_tot)
    i_prev = 0
    # ------------- Bezier dynamics
    for k_f in range(n_frames):
        b_first_visit = True
        f_name = frame_list[k_f]
        start_neye_idx = k_f * (x_dim // n_frames)
        start_Cmat_idx = start_neye_idx + n_points * num_iris_tot * 3
        for i in range(D):
            h = n_points - i - 1

            # matrices that remain fixed per degree of differentiation i = 0, ..., D-1
            upper_banded_eye = np.zeros((3 * (n_points - (i + 1)), 3 * (n_points - i)))
            eye_npmi = np.eye(3 * (n_points - (i + 1)))     # identity of corresponding size
            upper_banded_eye[:, :-3] -= eye_npmi
            upper_banded_eye[:, 3:] += eye_npmi

            for k_ir in range(num_iris_tot):
                A_bez_dyn = np.zeros((3 * (n_points - (i + 1)), x_dim))     # set to zero at every iteration

                # add identity-banded matrix in respective location of A_bez_dyn
                if b_first_visit:
                    pass
                else:
                    if i == i_prev:      # the very first time instant
                        start_neye_idx += 3 * (n_points - i)
                    else:
                        start_neye_idx += 3 * (n_points - (i - 1))
                end_neye_idx = start_neye_idx + 3 * (n_points - (i + 1))
                A_bez_dyn[:, start_neye_idx:end_neye_idx] -= eye_npmi
                A_bez_dyn[:, start_neye_idx + 3:end_neye_idx + 3] += eye_npmi

                # add -ci * I matrix corresponding to next derivative
                ci = get_ci_from_global_ir_idx(num_iris_tot, k_ir, durations, f_name, h)
                C_mat = -ci * eye_npmi

                if b_first_visit:
                    b_first_visit = False
                else:
                    if i == i_prev:      # if we still haven't changed to higher degree, use previous i
                        start_Cmat_idx += (n_points - (i+1)) * 3
                    else:               # we just changed ti higher degree, i
                        start_Cmat_idx += (n_points - i) * 3
                next_Cmat_idx = start_Cmat_idx + (n_points - (i+1)) * 3
                A_bez_dyn[:, start_Cmat_idx:next_Cmat_idx] = copy.copy(C_mat)
                bez_lin_eq_constraints.add_lin_eq(A_bez_dyn)

                i_prev = i      # store previous degree for house-keeping of variables/constraints order

    # ------------- Bezier continuity constraints
    A_bez_cont = np.zeros((3 * (num_iris_tot - 1) * (D+1) * n_frames, x_dim))
    y_Acond_curr_idx = 0
    b_first_visit = True
    for k_f in range(n_frames):
        for i in range(D+1):
            # matrices that remain fixed per degree of differentiation i = 0, ..., D-1
            Acont = np.zeros((3 * (num_iris_tot - 1), 3 * (n_points - i) * num_iris_tot))

            for k_ir in range(num_iris_tot-1):
                Acont[D*k_ir, D * k_ir * n_points + D*(n_points-1):D*n_points] = np.eye(D)
                Acont[D*k_ir, D * k_ir * n_points + D*n_points:D*(n_points+1)] = -np.eye(D)

            x_Acond_curr_idx = k_f * D * (num_iris_tot-1) * (D+1) + D * (num_iris_tot - 1) * i
            x_Acond_next_idx = x_Acond_curr_idx + D * (num_iris_tot - 1)
            if b_first_visit:
                b_first_visit = False
            else:
                y_Acond_curr_idx += Acont.shape[1]
            y_Acond_next_idx = y_Acond_curr_idx + Acont.shape[1]
            A_bez_cont[x_Acond_curr_idx:x_Acond_next_idx, y_Acond_curr_idx:y_Acond_next_idx] = copy.copy(Acont)
            bez_lin_eq_constraints.add_lin_eq(A_bez_cont)

    constraints = []
    cost = 0
    continuity = {}
    frame_idx, fr_seg_k_box = 0, 0
    seg_idx, k = 0, 0

    for k in range(num_iris_tot * n_frames):
        continuity[k] = {}

        # Update frame name and number of boxes within segment/interval
        f_name = frame_list[frame_idx]
        num_iris_current = len(iris_regions[f_name].iris_idx_seq[seg_idx])

        # TODO Enforce given positions

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
        link_threshold = 0.05
        # apply auxiliary rigid link constraint throughout all safe regions
        for aux_fr in aux_frames:
            prox_fr_idx, dist_fr_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot)

            # loop through all safe boxes
            link_length += link_threshold     # threshold for relaxation
            for nb in range(1, num_iris_tot-1):
                # for pnt in range(n_points-1):
                for pnt in range(1):
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
                        continue

                    else:
                        coeffs = reach_region[frame_name]
                        z_ee_seg = points[fr_idx * num_iris_tot + fr_iris_counter + si][0]

                    # reachable constraint
                    H = coeffs['H']
                    d_vec = np.reshape(coeffs['d'], (len(H), 1))
                    d_mat = np.repeat(d_vec, n_points, axis=1)
                    if frame_name == 'torso' or frame_name == 'LF' or frame_name == 'RF':
                        constraints.append(H @ (z_ee_seg.T - z_t.T) <= -d_mat)
                    # constraints.append(H @ (z_ee_seg.T - z_t.T) <= -d_mat)

                fr_iris_counter += num_iris_current
                k_fr_iris += num_iris_current

    prob = cp.Problem(cp.Minimize(cost + cost_log_abs_sum), constraints + soc_constraint)
    prob.solve(solver='SCS')

    if prob.status == 'infeasible':
        print('***** Problem was infeasible with CLARABEL solver. Retrying with relaxed SCS.')
        prob.solve(solver='SCS', eps_rel=1e-1, eps_abs=1e-1)

    # check link constraints values
    if bool(aux_frames):     # check if empy dictionary
        # apply auxiliary rigid link constraint throughout all safe regions
        for aux_fr in aux_frames:
            prox_fr_idx, dist_fr_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot)

            # loop through all safe boxes
            link_length += link_threshold     # threshold for relaxation
            for nb in range(1, num_iris_tot-1):
                # for pnt in range(n_points-1):
                for pnt in range(1):
                    link_proximal_point = points[prox_fr_idx+nb][0][pnt]
                    link_distal_point = points[dist_fr_idx+nb][0][pnt]
                    print(f"{aux_fr['parent_frame']} Link length discrepancy: {np.linalg.norm(link_proximal_point.value - link_distal_point.value) - link_length}")

    # Solve problem using scipy
    np_points = convert_cvxvar_to_numpy(points)
    res = minimize(cost + cost_log_abs_sum, points[k])

    # Reconstruct trajectory.
    beziers, path = [], []
    a = 0
    fr_seg_k_box, frame_idx, seg_idx = 0, 0, 0
    frame_name = frame_list[frame_idx]
    for k in range(num_iris_tot * n_frames):
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
