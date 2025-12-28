import copy
import time
from typing import List

import casadi as ca
import cvxpy as cp
import numpy as np
from casadi import nlpsol
from scipy.special import binom
from scipy.optimize import minimize

from ..casadi_ocp_constraints.casadi_ocp_functions import \
    IndexedPolytopeEllipsoidConstraint, IndexedPolytopePolytopeConstraint, IndexedCapsuleEllipsoidConstraint
from ..constraint_parsers import parse_mat_leq_constr, parse_repvec_eq_constr, \
    parse_vec_eq_constr, parse_mat_eq_constr
from ..cvx_mfpp_tools import get_aux_frame_idx, \
    create_bezier_cvx_norm_eq_relaxation, add_vel_acc_constr, add_vel_acc_constr_casadi
from ..scipy_ocp_constraints.scipy_ocp_functions import \
    LinearBezierIneqConstraint, LinearBezierEqConstraint
from util.path_parameterization import BezierCurve, CompositeBezierCurve
from ..self_collision_avoidance.sca_robot_geometry import SCARobotGeometry
from vision.iris.iris_regions_manager import IrisRegionsManager


def has_safe_point_at(points_sequence_order: List[np.ndarray],
                      num_iris_tot: int,
                      safe_points_lst: List[dict[str: np.ndarray]],
                      index: int,
                      frame_name:str):
    # determine the current segment based on the index
    ir_idx = index % num_iris_tot
    counter, seg_idx = 0, 0
    for dur in points_sequence_order:
        # check if the current index is within the segment
        if counter < ir_idx:
            counter += len(dur)
            seg_idx += 1
        else:
            break

    b_end_of_current_seg = ir_idx == counter
    if ir_idx == 0:
        raise ValueError('Initial positions are handled separately before here.')
    else:
        # apply desired safe points at the end of each segment
        if b_end_of_current_seg:
            # check if the current index is within the segment
            if frame_name in safe_points_lst[seg_idx].keys():
                return safe_points_lst[seg_idx][frame_name]
            # elif seg_idx < len(safe_points_lst) - 1 and frame_name in safe_points_lst[seg_idx + 1].keys():
            #     return safe_points_lst[seg_idx + 1][frame_name]

    # if num_iris_in_curr_seg == 1:
    #     # we only need to check if safe points is specified
    #     if frame_name in safe_points_lst[seg_idx].keys():
    #         return safe_points_lst[seg_idx][frame_name]
    # else:
    #     # check if the current index is within the segment
    #     if frame_name in safe_points_lst[seg_idx].keys():
    #         return safe_points_lst[seg_idx][frame_name]
    #     else:
    #         # check if the next segment has a safe point
    #         if seg_idx < len(safe_points_lst) - 1 and frame_name in safe_points_lst[seg_idx + 1].keys():
    #             return safe_points_lst[seg_idx + 1][frame_name]

    return [False]

def optimize_multiple_bezier_iris(reach_region: dict[str: np.array, str: np.array],
                                  aux_frames: List[dict],
                                  iris_regions: dict[str: IrisRegionsManager],
                                  durations: List[dict[str, np.array]],
                                  alpha: dict[int: float],
                                  safe_points_lst: List[dict[str, np.array]],
                                  fixed_frames=None,
                                  b_final_vel_constr=False,
                                  contact_sequence=None,
                                  surface_normals_lst=None,
                                  weights_rigid_link=None,
                                  b_use_knees_in_smooth_plan=True,
                                  n_points=None, **kwargs):
    if weights_rigid_link is None:
        weights_rigid_link = np.array([3500., 0.5, 10.])     # default for g1

    # number of frames
    n_frames = len(safe_points_lst[0].keys())

    # point segment order
    point_seg_order = []
    for dseg_dict in durations:
        point_seg_order.append(dseg_dict['torso'])

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
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points-1, axis=0)
                constraints.append(points[k][0][:-1] == fixed_frame_pos_mat)
            else:   # assign for just the first time instant
                constraints.append(points[k][0][0] == safe_points_lst[0][f_name])   # initial position
                # check if it has a final safe point assigned
                if fr_seg_k_box == (num_iris_current-1) and f_name in safe_points_lst[seg_idx+1].keys():
                    constraints.append(points[k][0][-1] == safe_points_lst[seg_idx+1][f_name])
                    # TODO uncomment below after fixing casadi version
                    # add_vel_acc_constr(f_name, surface_normals_lst[seg_idx], points[k], constraints, False)
        elif (k + 1) % num_iris_tot == 0:  # final position for each frame
            safe_pnt = has_safe_point_at(point_seg_order, num_iris_tot, safe_points_lst, k, f_name)
            if any(safe_pnt):
                constraints.append(points[k][0][0] == safe_pnt) # pos
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points-1, axis=0)
                constraints.append(points[k][0][1:] == fixed_frame_pos_mat)
            else:
                constraints.append(points[k][0][-1] == safe_points_lst[-1][f_name])
            # TODO check if below is needed since the last motion is taken into account below
            # add_vel_acc_constr(f_name, surface_normals_lst[-1], points[k], constraints)
        else:       # safe and fixed positions at other times
            safe_pnt = has_safe_point_at(point_seg_order, num_iris_tot, safe_points_lst, k, f_name)
            if any(safe_pnt):
                constraints.append(points[k][0][0] == safe_pnt) # pos
                # ignore if at initial stance
                # TODO add flag to toggle this or to customize epsilon value
                if b_final_vel_constr and (k-1) % num_iris_tot != 0:
                    add_vel_acc_constr(f_name, surface_normals_lst[seg_idx-1], points[k-1], constraints, False)
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points-2, axis=0)
                constraints.append(points[k][0][1:-1] == fixed_frame_pos_mat)

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

    # Reachability constraints
    reach_constr = []
    if reach_region is not None:
        for fr_idx, frame_name in enumerate(frame_list):
            if frame_name == 'torso':
                continue

            coeffs = reach_region[frame_name]
            H = coeffs['H']
            d_vec = np.reshape(coeffs['d'], (len(H), 1))
            d_mat = np.repeat(d_vec, n_points, axis=1)
            for ti in range(num_iris_tot):
                # torso index
                z_t = points[0 * num_iris_tot + ti][0]

                # current frame index
                z_ee_seg = points[fr_idx * num_iris_tot + ti][0]

                # reachable constraint
                if frame_name == 'LF' or frame_name == 'RF' or frame_name == 'LH' or frame_name == 'RH':
                    reach_constr.append(H @ (z_ee_seg.T - z_t.T) <= -d_mat)
                if b_use_knees_in_smooth_plan:
                    if frame_name == 'L_knee' or frame_name == 'R_knee':
                        # note: in some cases, scaling the reach polytope for knees helps the solver
                        reach_constr.append(H @ (z_ee_seg.T - z_t.T) <= -d_mat)

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

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost + cost_log_abs_sum), constraints + reach_constr + soc_constraint)
    try:
        prob.solve(solver='CLARABEL')
    except Exception as e:
        print("[MFPP Smooth WARNING] ", e)
        prob.solve(solver='SCS')

    if prob.status == 'infeasible':
        print(f'{"*" * 5} Smooth Problem was infeasible. Retrying with relaxed tolerances.')
        prob.solve(solver='SCS', eps_rel=5e-1, eps_abs=5e-1)
        if prob.status == 'infeasible':
            print(f'{"*" * 5} Smooth Problem was infeasible. Retrying without reachability constraints.')
            prob = cp.Problem(cp.Minimize(cost + cost_log_abs_sum), constraints + soc_constraint)
            prob.solve(solver='CLARABEL')
            if prob.status == 'infeasible':
                print('***** Smooth Problem was infeasible with CLARABEL solver. Retrying with relaxed SCS.')
                prob.solve(solver='SCS', eps_rel=5e-2, eps_abs=5e-2)
                if prob.status == 'infeasible':
                    print('***** Smooth (2nd Attempt) Problem was infeasible with CLARABEL solver. Retrying with relaxed SCS.')
                    prob.solve(solver='SCS', eps_rel=5e-1, eps_abs=5e-1)


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
    print(f"[Smooth] Cost: {cost.value:.3f}")
    sol_stats = {}
    sol_stats['cost'] = prob.value
    sol_stats['runtime'] = prob.solver_stats.solve_time
    sol_stats['cost_breakdown'] = cost_breakdown
    sol_stats['retiming_weights'] = retiming_weights
    dual_vars = {}
    dual_vars['lam_g0'] = prob.solution.dual_vars
    dual_vars['constraints_idx'] = sum(
        c.size if hasattr(c, 'size') else (c.shape[0] if hasattr(c, 'shape') else 1)
        for c in constraints
    )
    dual_vars['reach_constr_idx'] = sum(
        c.size if hasattr(c, 'size') else (c.shape[0] if hasattr(c, 'shape') else 1)
        for c in reach_constr
    )
    dual_vars['soc_constr_idx'] = sum(
        c.size if hasattr(c, 'size') else (c.shape[0] if hasattr(c, 'shape') else 1)
        for c in soc_constraint
    )
    dual_vars['lam_x0'] = np.zeros(prob.size_metrics.num_scalar_variables)

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
            elif vec_type == 'cvxpy':
                parsed_vec = np.reshape(points[ir][i].value, (vec_size, 1), order='F')
                vector_out = np.concatenate((vector_out, *parsed_vec))
            elif vec_type == 'numpy':
                parsed_vec = np.reshape(points[ir][i], (vec_size, 1), order='C')
                vector_out = np.concatenate((vector_out, *parsed_vec))
            else:
                raise ValueError('Invalid vector type specified. Use either casadi or numpy.')

    return vector_out


def pack_points_for_single_vector(points, vec_type: str):
    """
    Casadi's reshape method follows a column-major order, so we reshape the matrices
    accordingly to output:
    x = [p0_x, p0_y, p0_z, v0_x, v0_y, v0_z, ..., p1_x, p1_y, p1_z, v1_x, v1_y, v1_z, ...]
    where p0_x in mathbb{R}^(n_points)
    """
    vector_out = []
    if type(points) is dict:
        for k, v in points.items():      # loop through all IRIS regions
            if type(v) is dict:
                for k_deg in v.keys():
                    n_pnt, x_dim = points[k][k_deg].shape
                    vec_size = (n_pnt * x_dim)
                    if vec_type == 'casadi':
                        transposed_mat = ca.reshape(points[k][k_deg], x_dim, n_pnt)
                        vector_out = ca.vertcat(vector_out, ca.reshape(transposed_mat, vec_size, 1))
                    elif vec_type == 'cvxpy':
                        parsed_vec = np.reshape(points[k][k_deg].value, (vec_size, 1), order='F')
                        vector_out = np.concatenate((vector_out, *parsed_vec))
                    elif vec_type == 'numpy':
                        parsed_vec = np.reshape(points[k][k_deg], (vec_size, 1), order='C')
                        vector_out = np.concatenate((vector_out, *parsed_vec))
                    else:
                        raise ValueError(f'Invalid vector type {vec_type}. Use either casadi, numpy, or cvxpy.')
            elif type(v) is np.ndarray:     # this can be the dual variables from cvxpy
                vec_size = np.prod(v.shape)
                if vec_type == 'cvxpy':
                    parsed_vec = np.reshape(v, (vec_size, 1), order='C')
                    vector_out = np.concatenate((vector_out, *parsed_vec))
                else:
                    raise NotImplementedError(f'Parsing of {vec_type} into single vector not implemented.')

    elif type(points) is list:      # solution from casadi comes as list of lists
        for (_, pnt_mats) in enumerate(points):
            for (_, p_mat) in enumerate(pnt_mats):
                vec_size = np.prod(p_mat.shape)
                if vec_type == 'casadi':
                    parsed_vec = np.reshape(p_mat, (vec_size, 1), order='F')
                    vector_out = np.concatenate((vector_out, *parsed_vec))
                else:
                    raise NotImplementedError(f'Parsing of {vec_type} into single vector not implemented.')
    return vector_out

def optimize_multiple_bezier_iris_casadi(reach_region: dict[str: np.array, str: np.array],
                                  aux_frames: List[dict],
                                  iris_regions: dict[str: IrisRegionsManager],
                                  durations: List[dict[str, np.array]],
                                  alpha: dict[int: float],
                                  safe_points_lst: List[dict[str, np.array]],
                                  robot_geom_data: SCARobotGeometry=None,
                                  fixed_frames=None,
                                  b_final_vel_constr=False,
                                  contact_sequence=None,
                                  surface_normals_lst=None,
                                  initial_guess=None,
                                  weights_rigid_link=None,
                                  b_use_knees_in_smooth_plan=True,
                                  b_skip_sca=True,
                                  n_points=None, **kwargs):
    if weights_rigid_link is None:
        weights_rigid_link = np.array([3500., 0.5, 10.])     # default for g1

    # number of frames
    n_frames = len(safe_points_lst[0].keys())

    # point segment order
    point_seg_order = []
    for dseg_dict in durations:
        point_seg_order.append(dseg_dict['torso'])

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
                parse_repvec_eq_constr(np.array([safe_points_lst[seg_idx][f_name]]), points[k][0][:-1,:], constraints, lbg, ubg)
            else:   # assign for just the first time instant
                parse_vec_eq_constr(safe_points_lst[0][f_name], points[k][0][0,:], constraints, lbg, ubg)
                # check if it has a final safe point assigned
                if fr_seg_k_box == (num_iris_current-1) and f_name in safe_points_lst[seg_idx+1].keys():
                    parse_vec_eq_constr(safe_points_lst[seg_idx+1][f_name], points[k][0][-1,:], constraints, lbg, ubg)
                    # TODO add vel constraint
                    # add_vel_acc_constr_casadi(f_name, surface_normals_lst[seg_idx], points[k][], constraints, lbg, ubg)
        elif (k + 1) % num_iris_tot == 0:  # final position for each frame
            safe_pnt = has_safe_point_at(point_seg_order, num_iris_tot, safe_points_lst, k, f_name)
            if any(safe_pnt):
                parse_repvec_eq_constr(np.array([safe_pnt]), points[k][0][0,:], constraints, lbg, ubg)
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                parse_repvec_eq_constr(np.array([safe_points_lst[seg_idx][f_name]]), points[k][0][1:, :], constraints, lbg, ubg)
            else:
                parse_vec_eq_constr(safe_points_lst[-1][f_name], points[k][0][-1,:], constraints, lbg, ubg)
                # TODO add vel constraint
                # add_vel_acc_constr_casadi(f_name, surface_normals_lst[-1], points[k], constraints, lbg, ubg)
        else:       # safe and fixed positions at other times
            safe_pnt = has_safe_point_at(point_seg_order, num_iris_tot, safe_points_lst, k, f_name)
            if any(safe_pnt):
                # constraints.append(points[k][0][0] == safe_pnt) # pos
                parse_repvec_eq_constr(np.array([safe_pnt]), points[k][0][0,:], constraints, lbg, ubg)
                # ignore if at initial stance
                # TODO add flag to toggle this or to customize epsilon value
                if b_final_vel_constr and (k-1) % num_iris_tot != 0:
                    add_vel_acc_constr_casadi(f_name, surface_normals_lst[seg_idx-1], points[k-1], constraints, lbg, ubg, False)
            if (fixed_frames[seg_idx] is not None) and (f_name in fixed_frames[seg_idx]):
                parse_repvec_eq_constr(np.array([safe_points_lst[seg_idx][f_name]]), points[k][0][1:-1, :], constraints, lbg, ubg)

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

    # Reachability constraints
    if reach_region is not None:
        for fr_idx, frame_name in enumerate(frame_list):
            if frame_name == 'torso':
                continue

            coeffs = reach_region[frame_name]
            H = coeffs['H']
            d_vec = np.reshape(coeffs['d'], (len(H), 1))
            for ti in range(num_iris_tot):
                # torso index
                z_t = points[0 * num_iris_tot + ti][0]

                # current frame index
                z_ee_seg = points[fr_idx * num_iris_tot + ti][0]

                # reachable constraint
                if frame_name == 'LF' or frame_name == 'RF' or frame_name == 'LH' or frame_name == 'RH':
                    parse_mat_leq_constr(H, -d_vec, (z_ee_seg - z_t), constraints, lbg, ubg)
                if b_use_knees_in_smooth_plan:
                    if frame_name == 'L_knee' or frame_name == 'R_knee':
                        parse_mat_leq_constr(H, -d_vec, (z_ee_seg - z_t), constraints, lbg, ubg)

    # Rigid links (e.g., shin link length) constraint relaxation
    if bool(aux_frames):     # check if empty dictionary
        link_threshold = 0.05
        # apply auxiliary rigid link constraint throughout all safe regions
        for aux_fr in aux_frames:
            prox_fr_idx, dist_fr_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot)

            # loop through all safe boxes
            for nb in range(1, num_iris_tot-1):
                # for pnt in range(n_points-1):
                for pnt in range(1):
                    link_proximal_point = points[prox_fr_idx+nb][0][pnt,:]
                    link_distal_point = points[dist_fr_idx+nb][0][pnt,:]
                    # --- as equality constraint
                    constraints.append(ca.norm_2(link_proximal_point - link_distal_point))
                    lbg.append(link_length - link_threshold/2)
                    ubg.append(link_length + link_threshold/2)
                    if initial_guess is not None:
                        # initial_guess['lam_g0'] = np.insert(initial_guess['lam_g0'], initial_guess['constraints_idx'], 0.)
                        initial_guess['lam_g0'] = np.concatenate((initial_guess['lam_g0'], np.array([0.])))

    # Collect points into a single vector
    points_all = pack_points_for_single_vector(points, 'casadi')

    opts = {
        "ipopt": {
            "print_level": 5,   # {0: none; 1: final compute statistics; 3: num of vars, *5, EXIT; 12: all}
            "hessian_approximation": "limited-memory",   # exact
            "max_iter": 200,
            "mu_init": 1e-6,    # *0.1 (applicable if monotone strategy)
            "tol": 1e-1,
            "constr_viol_tol": 1e-2,    # *0.0001
            # "slack_bound_frac": 0.1,          # *0.01
            "mu_strategy":"adaptive",   # {monotone, adaptive}
            "nlp_scaling_method": "gradient-based", # {none, user-scaling, *gradient-based, equilibration-based}
            "jacobian_regularization_value": 1e-4,  # 1e-6, 2e-4  * 1e-8
            # "derivative_test": "first-order",
            # "derivative_test_print_all": "no",
            # "derivative_test_perturbation": 1e-6,
            # "derivative_test_tol": 0.0005
        }
    }

    sca_constraints = []
    if robot_geom_data is not None and not b_skip_sca:
        print(f'{"*" * 10} Solving with Primitive Self Collision Avoidance! {"*" * 10}')
        f_dist = {}
        Q = np.eye(3)

        # get indices of links to check for simplified rigid body collisions
        torso_idx = None
        sca_col_link_idxs = []
        for i, fr in enumerate(frame_list):
            if fr == 'torso':
                torso_geom_type = robot_geom_data.get_primitive_shape_type(fr)
                if torso_geom_type == 'box':
                    A1 = robot_geom_data.get_box_representation(fr)['A']
                    b1 = robot_geom_data.get_box_representation(fr)['b']
                elif torso_geom_type == 'capsule':
                    R = robot_geom_data.get_box_representation(fr)['R']
                    L = robot_geom_data.get_box_representation(fr)['L']
                torso_idx = i
            elif robot_geom_data.is_link_in_sca_list(fr):
                sca_col_link_idxs.append(i)
        print(f'Checking for self-collision with: {[frame_list[i] for i in sca_col_link_idxs]}')
        for col_idx in sca_col_link_idxs:
            # Simplified no self-collision function and constraints bounds for specified index pairs
            mfpp_bezier_data = {'current_frames': (torso_idx, col_idx),   # make torso and RK frames SCA
                                'n_points': n_points,
                                'num_derivatives': D,
                                'num_iris_per_frame': num_iris_tot,
                                'num_frames': n_frames
                                }
            sca_bez_points = range(0, num_iris_tot * n_points, 2)

            # populate col_pair_geom_data with respective primitive shape pair type information
            ee_geom_type = robot_geom_data.get_primitive_shape_type(frame_list[col_idx])
            if ee_geom_type == 'box':
                A2 = robot_geom_data.get_box_representation(frame_list[col_idx])['A']
                b2 = robot_geom_data.get_box_representation(frame_list[col_idx])['b']
                col_pair_geom_data = {'A1': A1, 'b1': b1, 'A2': A2, 'b2': b2, 'Q': Q}
            elif ee_geom_type == 'sphere':
                U = robot_geom_data.get_sphere_representation(frame_list[col_idx])['U']
                if torso_geom_type == 'box':
                    col_pair_geom_data = {'A1': A1, 'b1': b1, 'U': U, 'Q': Q}
                elif torso_geom_type == 'capsule':
                    col_pair_geom_data = {'A1': None, 'b1': None, 'R': R, 'L': L, 'U': U, 'Q': Q}
            else:
                raise ValueError(f'Invalid primitive shape {ee_geom_type} type specified for SCA.')

            sca_build_start_time = time.time()
            for i in sca_bez_points:
                mfpp_bezier_data['current_point'] = i
                i_name = 'f_dist_' + str(frame_list[col_idx]) + str(i)
                current_mfpp_data = copy.deepcopy(mfpp_bezier_data)
                if ee_geom_type == 'box':
                    f_dist[i_name] = IndexedPolytopePolytopeConstraint(i_name, col_pair_geom_data, current_mfpp_data)
                elif ee_geom_type == 'sphere':
                    if torso_geom_type == 'box':
                        f_dist[i_name] = IndexedPolytopeEllipsoidConstraint(i_name, col_pair_geom_data, current_mfpp_data)
                    if torso_geom_type == 'capsule':
                        f_dist[i_name] = IndexedCapsuleEllipsoidConstraint(i_name, col_pair_geom_data, current_mfpp_data)
                else:
                    raise ValueError(f'Invalid primitive shape type {ee_geom_type} specified for SCA.')
                # f_dist[i_name] = DColIndexedPolytopesConstraint(i_name, col_pair_geom_data, current_mfpp_data)
                sca_constraints.append(f_dist[i_name](points_all))
                lbg.append(1.0)
                ubg.append(ca.inf)
            sca_build_time = time.time() - sca_build_start_time

            # assume lagrange multipliers of SCA constraints are zero
            # initial_guess['lam_g0'] = np.concatenate((initial_guess['lam_g0'], np.zeros((len(sca_bez_points),1))))
            initial_guess['lam_g0'] = np.concatenate((initial_guess['lam_g0'].reshape(-1, 1), np.zeros((len(sca_bez_points),1))))

    opts["ipopt"]["max_iter"] = 200
    opts["ipopt"]["warm_start_init_point"] = "yes"
    opts["ipopt"]["warm_start_mult_bound_push"] = 1e-8
    opts["ipopt"]["warm_start_slack_bound_push"] = 1e-8
    opts["ipopt"]["warm_start_bound_push"] = 1e-8

    # Solve problem
    prob_construct_start_time = time.time()
    nlp = {'x': points_all,
           'f': cost,
           'g': ca.vertcat(*constraints, *sca_constraints)
           }
    solver = nlpsol('solver', 'ipopt', nlp, opts)
    prob_construct_time = time.time() - prob_construct_start_time

    solver_start_time = time.time()
    if initial_guess is not None:
        # sol_ig = solver(x0=initial_guess['x0'], lbg=lbg, ubg=ubg)
        # print(f"[Smooth CasADi] Cost Without LAM: {sol_ig['f'].full()[0][0]:.3f}")
        sol = solver(x0=initial_guess['x0'],
                     lam_g0=initial_guess['lam_g0'],
                     lam_x0=initial_guess['lam_x0'],
                     lbg=lbg, ubg=ubg)
    else:
        sol = solver(lbg=lbg, ubg=ubg)
    solver_compute_time = time.time() - solver_start_time

    x_sol = sol['x'].full()

    sol_points = unpack_sol_to_points(x_sol, num_iris_tot * n_frames, n_points, D, d)
    ig_points = unpack_sol_to_points(initial_guess['x0'], num_iris_tot * n_frames, n_points, D, d)

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
                    link_proximal_point = sol_points[prox_fr_idx+nb][0][pnt]
                    link_distal_point = sol_points[dist_fr_idx+nb][0][pnt]
                    print(f"{aux_fr['parent_frame']} Link length discrepancy: {np.linalg.norm(link_proximal_point - link_distal_point) - link_length}")

    # Reconstruct trajectory.
    beziers, path = [], []
    a = 0
    fr_seg_k_box, frame_idx, seg_idx = 0, 0, 0
    frame_name = frame_list[frame_idx]
    cost_prime = 0
    for k in range(num_iris_tot * n_frames):
        f_name = frame_list[frame_idx]
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

        # check cost of initial guess
        for i, ai in alpha.items():
            h_prime = n_points - 1 - i
            A_prime = np.zeros((h_prime + 1, h_prime + 1))
            for m in range(h_prime + 1):
                for n in range(h_prime + 1):
                    A_prime[m, n] = binom(h_prime, m) * binom(h_prime, n) / binom(2 * h_prime, m + n)
            A_prime *= durations[seg_idx][f_name][fr_seg_k_box] / (2 * h_prime + 1)
            A_prime = np.kron(A_prime, np.eye(d))
            p_prime = ca.vec(ig_points[k][i].T)
            cost_prime += ai * ca.bilin(A_prime, p_prime, p_prime)

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
    print(f"[Smooth CasADi] Initial Guess Cost: {cost_prime.full()[0][0]:.3f}")
    print(f"[Smooth CasADi] Cost: {sol['f'].full()[0][0]:.3f}")
    sol_stats = {'runtime': solver_compute_time,
                 }
    if not b_skip_sca:
        sol_stats['sca_build_time'] = sca_build_time if bool(aux_frames) else 0.0
        sol_stats['prob_construct_time'] = prob_construct_time

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
