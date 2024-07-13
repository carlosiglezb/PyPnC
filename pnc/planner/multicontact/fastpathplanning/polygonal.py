import copy
from typing import List

import numpy as np
import cvxpy as cp

from pnc.planner.multicontact.cvx_mfpp_tools import create_cvx_norm_eq_relaxation, get_aux_frame_idx
from vision.iris.iris_regions_manager import IrisRegionsManager

b_debug = False

if b_debug:
    from ruamel.yaml import YAML
    import os
    import sys

    cwd = os.getcwd()
    sys.path.append(cwd)


def solve_min_distance(B, box_seq, start, goal):
    x = cp.Variable((len(box_seq) + 1, B.d))

    boxes = [B.boxes[i] for i in box_seq]
    l = np.array([np.maximum(b.l, c.l) for b, c in zip(boxes[:-1], boxes[1:])])
    u = np.array([np.minimum(b.u, c.u) for b, c in zip(boxes[:-1], boxes[1:])])

    cost = cp.sum(cp.norm(x[1:] - x[:-1], 2, axis=1))
    constr = [x[0] == start, x[1:-1] >= l, x[1:-1] <= u, x[-1] == goal]

    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver='CLARABEL')

    length = prob.value
    traj = x.value
    solver_time = prob.solver_stats.solve_time

    return traj, length, solver_time


def solve_min_reach_iris_distance(reach: dict[str: np.array, str: np.array],
                                  iris_regions: dict[str: IrisRegionsManager],
                                  iris_seq: List[dict[str: int]],
                                  safe_points_list: List[dict[str: np.array]],
                                  aux_frames=None) -> [np.array, np.float64, np.float64]:
    stance_foot = 'LF'

    # Make copy of ee reachability region with only end effectors (e.g., excluding torso)
    ee_reach = {}
    for frame in iris_regions.keys():
        if frame != 'torso':
            ee_reach[frame] = iris_regions[frame]

    # parameters needed for state dimensions
    first_fr_iris = next(iter(iris_regions.values()))
    d = first_fr_iris.iris_list[0].iris_region.ambient_dimension()
    if reach is not None:
        N_planes = len(next(iter(reach.values()))['H'])
    n_ee = len(ee_reach)
    n_f = len(iris_regions)

    num_iris_tot = 0
    for irs_lst in iris_seq:
        num_iris_tot += len(next(iter(irs_lst.values())))

    # we write the problem as
    # x = [p_torso^{(i)}, p_lfoot^{(i)}, p_rfoot^{(i)}, p_lknee^{(i)}, p_rknee^{(i)}, ... , t^{(i)}]
    # containing all "i" curve points + auxiliary variables t^(i) that assimilate constant shin
    # lengths in paths for each leg
    x = cp.Variable(d * n_f * (num_iris_tot + 1))

    constr = []
    x_init_idx = 0
    # re-write multi-stage goal points (locations) in terms of optimization variables
    for f_name in iris_regions.keys():  # go in order (hence, refer to an OrderedDict)
        seg_idx = 0
        for sp_lst in safe_points_list:
            # process initial positions when in new frame
            if seg_idx == 0:
                constr.append(x[x_init_idx:x_init_idx + d] == sp_lst[f_name])
            else:
                if f_name in sp_lst:
                    constr.append(x[x_init_idx:x_init_idx + d] == sp_lst[f_name])
                if seg_idx == len(iris_seq):  # we have reached the end position
                    x_init_idx += d
                    continue
            num_boxes_current = len(next(iter(iris_seq[seg_idx].values())))
            x_init_idx += num_boxes_current * d
            seg_idx += 1

    # organize lower and upper state limits (include initial and final state bounds)
    x_init_idx = d       # initial point is assumed to be feasible
    for frame, ee_iris in iris_regions.items():
        for seg_idx in range(len(iris_seq)):
            for ir_seg_count in range(len(iris_seq[seg_idx][frame])):
                ir_seq_idx = iris_seq[seg_idx][frame][ir_seg_count]
                # look for intersections
                if ir_seg_count != (len(iris_seq[seg_idx][frame]) - 1):
                    ir_next_seq_idx = iris_seq[seg_idx][frame][ir_seg_count+1]
                    iris_current = ee_iris.iris_list[ir_seq_idx].iris_region
                    iris_next = ee_iris.iris_list[ir_next_seq_idx].iris_region
                    iris_intersect = iris_current.Intersection(iris_next, check_for_redundancy=True)
                    A = iris_intersect.A()
                    b = iris_intersect.b()
                else:       # last region, use as is
                    A = ee_iris.iris_list[ir_seq_idx].iris_region.A()
                    b = ee_iris.iris_list[ir_seq_idx].iris_region.b()
                constr.append(A @ x[x_init_idx:x_init_idx+d] <= b)
                x_init_idx += d
        x_init_idx += d     # initial point is assumed to be feasible

    # Construct end-effector reachability constraints (initial & final points specified)
    # H = np.zeros((N_planes * (num_iris_tot - 1), d * n_f * (num_iris_tot + 1)))
    # d_vec = np.zeros((N_planes * (num_iris_tot - 1) * (n_ee - 1),))
    # for sp_lst in safe_points_list:
    #     for frame_name in iris_regions.keys():    # go in order
    #         # get corresponding index
    #         frame_idx = list(sp_lst.keys()).index(frame_name)
    #
    #         if frame_name == 'torso' or frame_name == stance_foot:
    #             pass
    #         else:
    #             coeffs = reach[frame_name]
    #             for idx_box in range(num_iris_tot - 1):
    #                 # torso translation terms
    #                 torso_prev_start_idx = idx_box * d * n_f
    #                 torso_prev_end_idx = idx_box * d * n_f + d
    #                 torso_post_start_idx = (idx_box + 1) * d * n_f
    #                 torso_post_end_idx = (idx_box + 1) * d * n_f + d
    #                 H[N_planes * (idx_box):N_planes * (idx_box + 1), torso_prev_start_idx:torso_prev_end_idx] = -coeffs['H']
    #                 H[N_planes * (idx_box):N_planes * (idx_box + 1), torso_post_start_idx:torso_post_end_idx] = coeffs['H']
    #
    #                 # term corresponding to current end effector
    #                 ee_start_idx = d * n_f * (idx_box + 1) + d * frame_idx
    #                 ee_end_idx = d * n_f * (idx_box + 1) + d * (frame_idx + 1)
    #                 H[N_planes * (idx_box):N_planes * (idx_box + 1), ee_start_idx:ee_end_idx] = coeffs['H']
    #
    #             # the 'd' term is the same for all planes since the shift is included in the torso term
    #             d_vec = np.tile(coeffs['d'], num_iris_tot - 1)

    # add feasibility of end curve point? Perhaps since it depends on torso

    frame_list = list(safe_points_list[0].keys())
    # add rigid link constraint
    cost_log_abs = 0.
    soc_constraint = []
    A_soc_debug, d_soc_debug, cost_log_abs_list = [], [], []  # for debug purposes
    if aux_frames is not None:
        # w_i = cp.Parameter(pos=True, value=1.)
        w_i = np.array([0.1621, 0.006, 0.0808])    # based on desired distance between foot-shin frames
        for aux_fr in aux_frames:
            # get corresponding indices of optimization variable
            prox_idx, dist_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot + 1)

            if not np.isnan(prox_idx):
                # add convex relaxation of norm constraint
                A_soc_aux, d_soc_aux = create_cvx_norm_eq_relaxation(
                    prox_idx, dist_idx, link_length, d, num_iris_tot + 1, x)

                # concatenate A inequality matrices for debugging
                A_soc_debug += copy.deepcopy(A_soc_aux)
                d_soc_debug += copy.deepcopy(d_soc_aux)

        for Ai, di in zip(A_soc_debug, d_soc_debug):
            soc_constraint.append(cp.SOC(di, Ai @ x))
            cost_log_abs_list.append(w_i[0] * cp.log(Ai[0] @ x))
            # cost_log_abs_list.append(w_i[1] * cp.log(Ai[1] @ x))
            cost_log_abs_list.append(w_i[2] * cp.log(Ai[2] @ x))

        cost_log_abs = -(cp.sum(cost_log_abs_list))

    # knee-to-foot fixed distance constraint (this should be trivially satisfied when 2 boxes)
    # for bs_lst in iris_seq:
    #     num_boxes_current = len(next(iter(bs_lst.values())))
    #     if (num_boxes_current != 2) and (aux_frames is not None):
    #         cost_log_abs, soc_constraint, A_soc = create_cvx_norm_eq_relaxation(
    #                                                         aux_frames, num_iris_tot, d*n_f, x)
    #     else:
    #         soc_constraint = []
    #         A_soc = []
    #         cost_log_abs = 0.

    # minimum distance cost (add distance between points of corresponding frame)
    cost = 0
    for fr in range(n_f):
        start_idx = fr * d * (num_iris_tot + 1)
        end_idx = start_idx + d * (num_iris_tot + 1)
        x_fr = x[start_idx: end_idx]
        p_fr_t = cp.reshape(x_fr, [d, num_iris_tot + 1])
        cost += cp.sum(cp.norm(p_fr_t[:, 1:] - p_fr_t[:, :-1], axis=1))

    # solve
    prob = cp.Problem(cp.Minimize(cost + cost_log_abs), constr + soc_constraint)
    prob.solve(solver='SCS')

    if prob.status == 'infeasible':
        print('Polygonal problem was infeasible. Retrying with relaxed tolerances.')
        prob.solve(solver='SCS', eps_rel=0.1, eps_abs=0.1)

    length = prob.value
    traj = x.value
    solver_time = prob.solver_stats.solve_time

    # check distance of knee and foot points at each curve point
    for Ai in A_soc_debug:
        opt_shin_len_err = np.linalg.norm(Ai @ traj) - link_length
        print(f"Shin length discrepancy: {opt_shin_len_err}")

    if b_debug:
        yaml = YAML()
        file_loc = cwd + '/test' + '/poly_min_distance_data.yaml'

        traj_reshape = np.reshape(traj, [7, 30])
        with open(file_loc, 'w') as f:
            for p in traj_reshape:
                yaml.dump(p.tolist(), f)

    return traj, length, solver_time


def solve_min_reach_distance(reach, safe_boxes, box_seq, safe_points_list, aux_frames=None):
    stance_foot = 'LF'

    # Make copy of ee reachability region with only end effectors (e.g., excluding torso)
    ee_reach = {}
    for frame in safe_boxes.keys():
        if frame != 'torso':
            ee_reach[frame] = safe_boxes[frame]

    # parameters needed for state dimensions
    first_box = next(iter(safe_boxes.values()))
    d = first_box.B.d
    if reach is not None:
        N_planes = len(next(iter(reach.values()))['H'])
    n_ee = len(ee_reach)
    n_f = len(safe_boxes)

    num_boxes_tot = 0
    for bs_lst in box_seq:
        num_boxes_tot += len(next(iter(bs_lst.values())))
    # num_boxes = np.max([len(bs) for bs in box_seq.values()])

    # we write the problem as
    # x = [p_torso^{(i)}, p_lfoot^{(i)}, p_rfoot^{(i)}, p_lknee^{(i)}, p_rknee^{(i)}, ... , t^{(i)}]
    # containing all "i" curve points + auxiliary variables t^(i) that assimilate constant shin
    # lengths in paths for each leg
    x = cp.Variable(d * n_f * (num_boxes_tot + 1))

    constr = []
    x_init_idx = 0
    # re-write multi-stage goal points (locations) in terms of optimization variables
    for f_name in safe_boxes.keys():  # go in order (hence, refer to an OrderedDict)
        seg_idx = 0
        for sp_lst in safe_points_list:
            # process initial positions when in new frame
            if seg_idx == 0:
                constr.append(x[x_init_idx:x_init_idx + d] == sp_lst[f_name])
            else:
                if f_name in sp_lst:
                    constr.append(x[x_init_idx:x_init_idx + d] == sp_lst[f_name])
                if seg_idx == len(box_seq):     # we have reached the end position
                    x_init_idx += d
                    continue
            num_boxes_current = len(next(iter(box_seq[seg_idx].values())))
            x_init_idx += num_boxes_current * d
            seg_idx += 1

    frame_list = list(safe_points_list[0].keys())
    # organize lower and upper state limits (include initial and final state bounds)
    l = np.zeros((d * n_f * (num_boxes_tot + 1),))
    u = np.zeros((d * n_f * (num_boxes_tot + 1),))
    x_init_idx = 0
    for frame, ee_box in safe_boxes.items():
        # initial ee state / position
        l[x_init_idx:x_init_idx + d] = ee_box.B.boxes[box_seq[0][frame][0]].l
        u[x_init_idx:x_init_idx + d] = ee_box.B.boxes[box_seq[0][frame][0]].u
        x_init_idx += d
        seg_idx = 1     # we can start from the first segment
        for bs_lst in box_seq:
            num_boxes_current = len(next(iter(bs_lst.values())))
            boxes = [ee_box.B.boxes[i] for i in bs_lst[frame]]

            for p in range(0, num_boxes_current-1):
                llim = np.array([np.maximum(boxes[p].l, boxes[p+1].l)])
                ulim = np.array([np.minimum(boxes[p].u, boxes[p+1].u)])
                l[x_init_idx:x_init_idx + d] = llim
                u[x_init_idx:x_init_idx + d] = ulim
                x_init_idx += d
            # for last box simply use the given bounds
            l[x_init_idx:x_init_idx + d] = boxes[num_boxes_current-1].l
            u[x_init_idx:x_init_idx + d] = boxes[num_boxes_current-1].u
            x_init_idx += d
            seg_idx += 1

    # Construct end-effector reachability constraints (initial & final points specified)
    # H = np.zeros((N_planes * (num_boxes_tot - 1), d * n_f * (num_boxes_tot + 1)))
    # d_vec = np.zeros((N_planes * (num_boxes_tot - 1) * (n_ee - 1),))
    # for sp_lst in safe_points_list:
    #     for frame_name in safe_boxes.keys():    # go in order
    #         # get corresponding index
    #         frame_idx = list(sp_lst.keys()).index(frame_name)
    #
    #         if frame_name == 'torso' or frame_name == stance_foot:
    #             pass
    #         else:
    #             coeffs = reach[frame_name]
    #             for idx_box in range(num_boxes_tot - 1):
    #                 # torso translation terms
    #                 torso_prev_start_idx = idx_box * d * n_f
    #                 torso_prev_end_idx = idx_box * d * n_f + d
    #                 torso_post_start_idx = (idx_box + 1) * d * n_f
    #                 torso_post_end_idx = (idx_box + 1) * d * n_f + d
    #                 H[N_planes * (idx_box):N_planes * (idx_box + 1), torso_prev_start_idx:torso_prev_end_idx] = -coeffs['H']
    #                 H[N_planes * (idx_box):N_planes * (idx_box + 1), torso_post_start_idx:torso_post_end_idx] = coeffs['H']
    #
    #                 # term corresponding to current end effector
    #                 ee_start_idx = d * n_f * (idx_box + 1) + d * frame_idx
    #                 ee_end_idx = d * n_f * (idx_box + 1) + d * (frame_idx + 1)
    #                 H[N_planes * (idx_box):N_planes * (idx_box + 1), ee_start_idx:ee_end_idx] = coeffs['H']
    #
    #             # the 'd' term is the same for all planes since the shift is included in the torso term
    #             d_vec = np.tile(coeffs['d'], num_boxes_tot - 1)

    # add feasibility of end curve point? Perhaps since it depends on torso

    # add rigid link constraint
    cost_log_abs = 0.
    soc_constraint = []
    A_soc_debug, d_soc_debug, cost_log_abs_list = [], [], []      # for debug purposes
    if aux_frames is not None:
        w_i = cp.Parameter(pos=True, value=1.)
        for aux_fr in aux_frames:
            # get corresponding indices of optimization variable
            prox_idx, dist_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_boxes_tot + 1)

            if not np.isnan(prox_idx):
                # add convex relaxation of norm constraint
                A_soc_aux, d_soc_aux = create_cvx_norm_eq_relaxation(
                    prox_idx, dist_idx, link_length, d, num_boxes_tot+1, x)

                # concatenate A inequality matrices for debugging
                A_soc_debug += copy.deepcopy(A_soc_aux)
                d_soc_debug += copy.deepcopy(d_soc_aux)

        for Ai, di in zip(A_soc_debug, d_soc_debug):
            soc_constraint.append(cp.SOC(di, Ai @ x))
            cost_log_abs_list.append(w_i * cp.log(Ai[0] @ x))
            cost_log_abs_list.append(0.*w_i * cp.log(Ai[1] @ x))
            cost_log_abs_list.append(w_i * cp.log(Ai[2] @ x))
        cost_log_abs = -(cp.sum(cost_log_abs_list))

    # knee-to-foot fixed distance constraint (this should be trivially satisfied when 2 boxes)
    # for bs_lst in box_seq:
    #     num_boxes_current = len(next(iter(bs_lst.values())))
    #     if (num_boxes_current != 2) and (aux_frames is not None):
    #         cost_log_abs, soc_constraint, A_soc = create_cvx_norm_eq_relaxation(
    #                                                         aux_frames, num_boxes_tot, d*n_f, x)
    #     else:
    #         soc_constraint = []
    #         A_soc = []
    #         cost_log_abs = 0.

    # minimum distance cost (add distance between points of corresponding frame)
    cost = 0
    for fr in range(n_f):
        start_idx = fr * d * (num_boxes_tot + 1)
        end_idx = start_idx + d * (num_boxes_tot + 1)
        x_fr = x[start_idx: end_idx]
        p_fr_t = cp.reshape(x_fr, [d, num_boxes_tot + 1])
        cost += cp.sum(cp.norm(p_fr_t[:, 1:] - p_fr_t[:, :-1], axis=1))

    # add limits if more than 2 boxes
    constr.append(l <= x)   # safe, collision-free boxes (lower limit)
    constr.append(x <= u)   # safe, collision-free boxes (upper limit)

    # solve
    prob = cp.Problem(cp.Minimize(cost + cost_log_abs), constr + soc_constraint)
    prob.solve(solver='CLARABEL')

    length = prob.value
    traj = x.value
    solver_time = prob.solver_stats.solve_time

    # check distance of knee and foot points at each curve point
    for Ai, di in zip(A_soc_debug, d_soc_debug):
        opt_shin_len_err = np.linalg.norm(Ai @ traj) - di.value
        print(f"Shin length discrepancy: {opt_shin_len_err}")

    if b_debug:
        yaml = YAML()
        file_loc = cwd + '/test' + '/poly_min_distance_data.yaml'

        traj_reshape = np.reshape(traj, [7, 30])
        with open(file_loc, 'w') as f:
            for p in traj_reshape:
                yaml.dump(p.tolist(), f)

    return traj, length, solver_time


def log(s1, size=10):
    s1 = str(s1)
    s0 = list('|' + ' ' * size + '|')
    s0[2:2 + len(s1)] = s1
    return ''.join(s0)


def init_log():
    print(log('Iter.') + log('Length') + log('N boxes'))
    print('-' * 36)


def term_log():
    print('-' * 36)


def update_log(i, length, n_boxes):
    print(log(i) + log('{:.2e}'.format(length)) + log(n_boxes))


def iterative_planner(B, start, goal, box_seq, verbose=True, tol=1e-5, **kwargs):
    if verbose:
        init_log()

    box_seq = np.array(box_seq)
    solver_time = 0
    n_iters = 0
    while True:
        n_iters += 1

        box_seq = jump_box_repetitions(box_seq)
        traj, length, solver_time_i = solve_min_distance(B, box_seq, start, goal, **kwargs)
        solver_time += solver_time_i

        if verbose:
            update_log(n_iters, length, len(box_seq))

        box_seq, traj = merge_overlaps(box_seq, traj, tol)

        kinks = find_kinks(traj, tol)

        insert_k = []
        insert_i = []
        for k in kinks:

            i1 = box_seq[k - 1]
            i2 = box_seq[k]
            B1 = B.boxes[i1]
            B2 = B.boxes[i2]
            cached_finf = 0

            subset = list(B.inters[i1] & B.inters[i2])
            for i in B.contain(traj[k], tol, subset):
                B3 = B.boxes[i]
                B13 = B1.intersect(B3)
                B23 = B2.intersect(B3)
                f = dual_box_insertion(*traj[k - 1:k + 2], B13, B23, tol)
                f2 = np.linalg.norm(f)
                finf = np.linalg.norm(f, ord=np.inf)

                if f2 > 1 + tol and finf > cached_finf + tol:
                    cached_i = i
                    cached_finf = finf

            if cached_finf > 0:
                insert_k.append(k)
                insert_i.append(cached_i)

        if len(insert_k) > 0:
            box_seq = np.insert(box_seq, insert_k, insert_i)
        else:
            if verbose:
                term_log()
                print(f'Polygonal phase terminated in {n_iters} iterations')
                print(f'Final length is ' + '{:.3e}'.format(length))
                print(f'Solver time was {np.round(solver_time, 5)}')
            return list(box_seq), traj, length, solver_time


def iterative_planner_multiple(safe_boxes, reach, safe_points_list, box_seq, verbose=True,
                               aux_frames=None, tol=1e-5, **kwargs):
    if verbose:
        init_log()

    # box_seq = np.array(box_seq)
    solver_time = 0
    n_iters = 0
    while True:
        n_iters += 1

        # for frame, bs in box_seq.items():
        #     box_seq[frame] = jump_box_repetitions(np.array(bs))
        traj, length, solver_time_i = solve_min_reach_distance(reach, safe_boxes, box_seq,
                                                               safe_points_list, aux_frames, **kwargs)
        solver_time += solver_time_i

        # TODO: from here on, deal with each end effector separately
        # for frame, b_seq in box_seq.items():
        #     B = safe_boxes[frame].B
        #
        #     if verbose:
        #         update_log(n_iters, length, len(box_seq[frame]))

        # box_seq[frame], traj = merge_overlaps_multiple(box_seq, traj, tol)

        # kinks = find_kinks(traj, tol)

        insert_k = []
        insert_i = []
        # for k in kinks:
        #
        #     i1 = box_seq[k - 1]
        #     i2 = box_seq[k]
        #     B1 = B.boxes[i1]
        #     B2 = B.boxes[i2]
        #     cached_finf = 0
        #
        #     subset = list(B.inters[i1] & B.inters[i2])
        #     for i in B.contain(traj[k], tol, subset):
        #         B3 = B.boxes[i]
        #         B13 = B1.intersect(B3)
        #         B23 = B2.intersect(B3)
        #         f = dual_box_insertion(*traj[k-1:k+2], B13, B23, tol)
        #         f2 = np.linalg.norm(f)
        #         finf = np.linalg.norm(f, ord=np.inf)
        #
        #         if f2 > 1 + tol and finf > cached_finf + tol:
        #             cached_i = i
        #             cached_finf = finf
        #
        #     if cached_finf > 0:
        #         insert_k.append(k)
        #         insert_i.append(cached_i)

        if len(insert_k) > 0:
            b_seq = np.insert(b_seq, insert_k, insert_i)
        else:
            if verbose:
                term_log()
                print(f'Polygonal phase terminated in {n_iters} iterations')
                print(f'Final length is ' + '{:.3e}'.format(length))
                print(f'Solver time was {np.round(solver_time, 5)}')
            return box_seq, traj, length, solver_time


def merge_overlaps(box_seq, traj, tol):
    keep = list(np.linalg.norm(traj[:-1] - traj[1:], axis=1) > tol)
    box_seq = box_seq[keep]
    traj = traj[keep + [True]]

    return box_seq, traj


def merge_overlaps_multiple(box_seq, traj, tol):
    keep = list(np.linalg.norm(traj[:-1] - traj[1:], axis=1) > tol)
    box_seq = box_seq[keep]
    traj = traj[keep + [True]]

    return box_seq, traj


def find_kinks(traj, tol):
    '''
    Detects the indices of the points where the trajectory bends. To do so, it
    uses the triangle inequality (division free):

        |traj[i] - traj[i-1]| + |traj[i+1] - traj[i]| > |traj[i+1] - traj[i-1]|

    implies that traj[i] is a kink.
    '''

    xy = np.linalg.norm(traj[1:-1] - traj[:-2], axis=1)
    yz = np.linalg.norm(traj[2:] - traj[1:-1], axis=1)
    xz = np.linalg.norm(traj[2:] - traj[:-2], axis=1)

    return np.where(xy + yz - xz > tol)[0] + 1


def dual_box_insertion(a, x, b, B13, B23, tol=1e-9):
    '''
    Given a point x that solves

        minimize     |x - a| + |x - b|
        subject to   x in B1 cap B2,

    checks if x1 = x2 = x is also optimal for the following problem:

        minimize     |x1 - a| + |x1 - x2| + |x2 - b|
        subject to   x1 in B1 cap B3
                     x2 in B2 cap B3.

    The arguments B13 and B23 denote B1 cap B3 and B2 cap B3, respectively.
    Assumes that neither a nor b is equal to x.

    Returns the dual variable f associated with the additional constraint
    x1 = x2 needed for the check. The multiplier f gives the rate of change of
    the cost as the points x1 and x2 are moved away from each other, i.e., 
    f = d length / d (x2 - x1). If the two-norm of f is greater than one then
    inserting the box B3 decreases the cost.
    '''

    lam1 = a - x
    lam2 = b - x
    lam1 /= np.linalg.norm(lam1)
    lam2 /= np.linalg.norm(lam2)

    L1, U1 = B13.active_set(x, tol)
    L2, U2 = B23.active_set(x, tol)

    fmin1 = - lam1
    fmax1 = - lam1
    fmin2 = lam2.copy()
    fmax2 = lam2.copy()

    fmin1[L1] = - np.inf
    fmax1[U1] = np.inf
    fmin2[U2] = - np.inf
    fmax2[L2] = np.inf

    fmin = np.maximum(fmin1, fmin2)
    fmax = np.minimum(fmax1, fmax2)
    f = np.clip(0, fmin, fmax)

    return f


def jump_box_repetitions(box_seq):
    i = 0
    keep = []
    while True:
        keep.append(i)
        b = box_seq[i]
        for j, c in enumerate(box_seq[i:][::-1]):
            if c == b:
                break
        i = len(box_seq) - j
        if i >= len(box_seq):
            break

    return box_seq[keep]
