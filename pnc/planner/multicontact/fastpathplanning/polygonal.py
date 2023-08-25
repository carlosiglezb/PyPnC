import numpy as np
import cvxpy as cp
from time import time
from itertools import accumulate
from bisect import bisect
from scipy.special import binom

from pnc.planner.multicontact.cvx_mfpp_tools import create_cvx_norm_eq_relaxation


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


def solve_min_reach_distance(reach, safe_boxes, box_seq, start, goal, aux_frames=None):
    stance_foot = 'LF'

    # Make copy of ee reachability region with only end effectors (e.g., excluding torso)
    ee_reach = {}
    for frame, coeffs in reach.items():
        if frame != 'torso':
            ee_reach[frame] = reach[frame]

    # parameters needed for state dimensions
    first_box = next(iter(safe_boxes.values()))
    d = first_box.B.d
    num_boxes = np.max([len(bs) for bs in box_seq.values()])
    N_planes = len(next(iter(reach.values()))['H'])
    n_ee = len(ee_reach)
    n_f = len(reach)

    # we write the problem as
    # x = [p_torso^{(i)}, p_lfoot^{(i)}, p_rfoot^{(i)}, p_lknee^{(i)}, p_rknee^{(i)}, ... , t^{(i)}]
    # containing all "i" curve points + auxiliary variables t^(i) that assimilate constant shin
    # lengths in paths for each leg
    x = cp.Variable(d * n_f * (num_boxes + 1))

    # frame start and goal locations in terms of optimization variables
    x_init, x_goal = [], []
    for x0 in start.values():
        x_init = np.concatenate((x_init, x0))
    for xf in goal.values():
        x_goal = np.concatenate((x_goal, xf))

    # organize lower and upper state limits
    l = np.zeros((d * n_f * (num_boxes - 1),))
    u = np.zeros((d * n_f * (num_boxes - 1),))
    ee_idx = 0
    for frame, ee_box in safe_boxes.items():
        boxes = [ee_box.B.boxes[i] for i in box_seq[frame]]
        llim_current_ee = np.array([np.maximum(b.l, c.l) for b, c in zip(boxes[:-1], boxes[1:])])
        ulim_current_ee = np.array([np.minimum(b.u, c.u) for b, c in zip(boxes[:-1], boxes[1:])])
        box_idx = 0
        for p in range(num_boxes - 1):
            l[p * d * n_f + ee_idx:p * d * n_f + d + ee_idx] = llim_current_ee[box_idx]
            u[p * d * n_f + ee_idx:p * d * n_f + d + ee_idx] = ulim_current_ee[box_idx]
            box_idx += 1
        ee_idx += d

    # Construct end-effector reachability constraints (initial & final points specified)
    H = np.zeros((N_planes * (num_boxes - 1), d * n_f * (num_boxes + 1)))
    d_vec = np.zeros((N_planes * (num_boxes - 1) * (n_ee - 1),))
    for frame_name, _ in start.items():
        # get corresponding index
        frame_idx = list(start.keys()).index(frame_name)

        if frame_name == 'torso' or frame_name == stance_foot:
            pass
        else:
            coeffs = reach[frame_name]
            for idx_box in range(num_boxes - 1):
                # torso translation terms
                torso_prev_start_idx = idx_box * d * n_f
                torso_prev_end_idx = idx_box * d * n_f + d
                torso_post_start_idx = (idx_box + 1) * d * n_f
                torso_post_end_idx = (idx_box + 1) * d * n_f + d
                H[N_planes * (idx_box):N_planes * (idx_box + 1), torso_prev_start_idx:torso_prev_end_idx] = -coeffs['H']
                H[N_planes * (idx_box):N_planes * (idx_box + 1), torso_post_start_idx:torso_post_end_idx] = coeffs['H']

                # term corresponding to current end effector
                ee_start_idx = d * n_f * (idx_box + 1) + d * frame_idx
                ee_end_idx = d * n_f * (idx_box + 1) + d * (frame_idx + 1)
                H[N_planes * (idx_box):N_planes * (idx_box + 1), ee_start_idx:ee_end_idx] = coeffs['H']

            # the 'd' term is the same for all planes since the shift is included in the torso term
            d_vec = np.tile(coeffs['d'], num_boxes - 1)

    # add feasibility of end curve point? Perhaps since it depends on torso

    # knee-to-foot fixed distance constraint
    if aux_frames is not None:
        cost_log_abs, soc_constraint, A_soc = create_cvx_norm_eq_relaxation(
                                                        aux_frames, num_boxes, d*n_f, x)
    else:
        soc_constraint = []
        cost_log_abs = 0.

    cost = cp.sum(cp.norm(x[d * n_f:] - x[:-d * n_f], 2)) + cost_log_abs

    constr = [x[:d * n_f] == x_init,
              x[d * n_f:-d * n_f] >= l,     # safe, collision-free boxes (lower limit)
              x[d * n_f:-d * n_f] <= u,     # safe, collision-free boxes (upper limit)
              # H @ x <= -d_vec,            # non-stance frame remains reachable
              x[-d * n_f:] == x_goal]

    prob = cp.Problem(cp.Minimize(cost), constr + soc_constraint)
    prob.solve(solver='SCS')

    length = prob.value
    traj = x.value
    solver_time = prob.solver_stats.solve_time

    # check distance of knee and foot points at each curve point
    for Ai in A_soc:
        opt_shin_len_err = np.linalg.norm(Ai @ traj) - 0.32428632635527505
        print(f"Shin length discrepancy: {opt_shin_len_err}")
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


def iterative_planner_multiple(safe_boxes, reach, start, goal, box_seq, verbose=True,
                               aux_frames=None, tol=1e-5, **kwargs):
    if verbose:
        init_log()

    # box_seq = np.array(box_seq)
    solver_time = 0
    n_iters = 0
    while True:
        n_iters += 1

        for frame, bs in box_seq.items():
            box_seq[frame] = jump_box_repetitions(np.array(bs))
        traj, length, solver_time_i = solve_min_reach_distance(reach, safe_boxes, box_seq, start, goal,
                                                               aux_frames, **kwargs)
        solver_time += solver_time_i

        # TODO: from here on, deal with each end effector separately
        for frame, b_seq in box_seq.items():
            B = safe_boxes[frame].B

        if verbose:
            update_log(n_iters, length, len(box_seq[frame]))

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
