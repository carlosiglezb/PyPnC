import copy
from typing import List

import numpy as np
import cvxpy as cp
from time import time
from itertools import accumulate
from bisect import bisect
from scipy.special import binom

from pnc.planner.multicontact.cvx_mfpp_tools import create_bezier_cvx_norm_eq_relaxation, get_aux_frame_idx

from enum import Enum

from vision.iris.iris_regions_manager import IrisRegionsManager


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2


class BezierParam(Enum):
    POS = 0
    VEL = 1
    ACC = 2


eps_vel_constr = 0.01


class BezierCurve:

    def __init__(self, points, a=0, b=1):

        assert b > a

        self.points = points
        self.h = points.shape[0] - 1
        self.d = points.shape[1]
        self.a = a
        self.b = b
        self.duration = b - a

    def __call__(self, t):

        c = np.array([self.berstein(t, n) for n in range(self.h + 1)])
        return c.T.dot(self.points)

    def berstein(self, t, n):

        c1 = binom(self.h, n)
        c2 = (t - self.a) / self.duration 
        c3 = (self.b - t) / self.duration
        value = c1 * c2 ** n * c3 ** (self.h - n) 

        return value

    def start_point(self):

        return self.points[0]

    def end_point(self):

        return self.points[-1]
        
    def derivative(self):

        points = (self.points[1:] - self.points[:-1]) * (self.h / self.duration)

        return BezierCurve(points, self.a, self.b)

    def l2_squared(self):

        A = np.zeros((self.h + 1, self.h + 1))
        for m in range(self.h + 1):
            for n in range(self.h + 1):
                A[m, n] = binom(self.h, m) * binom(self.h, n) / binom(2 * self.h, m + n)
        A *= self.duration / (2 * self.h + 1)
        A = np.kron(A, np.eye(self.d))

        p = self.points.flatten()

        return p.dot(A.dot(p))

    def plot2d(self, samples=51, **kwargs):

        import matplotlib.pyplot as plt

        options = {'c':'b'}
        options.update(kwargs)
        t = np.linspace(self.a, self.b, samples)
        plt.plot(*self(t).T, **options)

    def plot3d(self, ax, samples=51, **kwargs):

        import matplotlib.pyplot as plt

        options = {'c':'b'}
        options.update(kwargs)
        t = np.linspace(self.a, self.b, samples)
        ax.plot(*self(t).T, **options)

    def get_sample_points(self, samples=11):
        t = np.linspace(self.a, self.b, samples)
        points = self(t).astype(np.float32)
        return t, points

    def scatter2d(self, **kwargs):

        import matplotlib.pyplot as plt

        options = {'fc':'orange', 'ec':'k', 'zorder':3}
        options.update(kwargs)
        plt.scatter(*self.points.T, **options)

    def plot_2dpolygon(self, **kwargs):

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from scipy.spatial import ConvexHull

        options = {'fc':'lightcoral'}
        options.update(kwargs)
        hull = ConvexHull(self.points)
        ordered_points = hull.points[hull.vertices]
        poly = Polygon(ordered_points, **options)
        plt.gca().add_patch(poly)


class CompositeBezierCurve:

    def __init__(self, beziers):

        for bez1, bez2 in zip(beziers[:-1], beziers[1:]):
            assert bez1.b == bez2.a
            assert bez1.d == bez2.d

        self.beziers = beziers
        self.N = len(self.beziers)
        self.d = beziers[0].d
        self.a = beziers[0].a
        self.b = beziers[-1].b
        self.duration = self.b - self.a
        self.transition_times = [self.a] + [bez.b for bez in beziers]

    def find_segment(self, t):

        return min(bisect(self.transition_times, t) - 1, self.N - 1)

    def __call__(self, t):

        i = self.find_segment(t)

        return self.beziers[i](t)

    def start_point(self):

        return self.beziers[0].start_point()

    def end_point(self):

        return self.beziers[-1].end_point()

    def derivative(self):

        return CompositeBezierCurve([b.derivative() for b in self.beziers])

    def bound_on_integral(self, f):

        return sum(bez.bound_on_integral(f) for bez in self.beziers)

    def plot2d(self, **kwargs):

        for bez in self.beziers:
            bez.plot2d(**kwargs)

    def plot3d(self, ax, **kwargs):

        for bez in self.beziers:
            bez.plot3d(ax, **kwargs)

    def scatter2d(self, **kwargs):

        for bez in self.beziers:
            bez.scatter2d(**kwargs)

    def plot_2dpolygon(self, **kwargs):

        for bez in self.beziers:
            bez.plot_2dpolygon(**kwargs)

def optimize_bezier(L, U, durations, alpha, initial, final,
    n_points=None, **kwargs):

    # Problem size.
    n_boxes, d = L.shape
    D = max(alpha)
    assert max(initial) <= D
    assert max(final) <= D
    if n_points is None:
        n_points = (D + 1) * 2

    # Control points of the curves and their derivatives.
    points = {}
    for k in range(n_boxes):
        points[k] = {}
        for i in range(D + 1):
            size = (n_points - i, d)
            points[k][i] = cp.Variable(size)

    # Boundary conditions.
    constraints = []
    for i, value in initial.items():
        constraints.append(points[0][i][0] == value)
    for i, value in final.items():
        constraints.append(points[n_boxes - 1][i][-1] == value)

    # Loop through boxes.
    cost = 0
    continuity = {}
    for k in range(n_boxes):
        continuity[k] = {}

        # Box containment.
        Lk = np.array([L[k]] * n_points)
        Uk = np.array([U[k]] * n_points)
        constraints.append(points[k][0] >= Lk)
        constraints.append(points[k][0] <= Uk)

        # Bezier dynamics.
        for i in range(D):
            h = n_points - i - 1
            ci = durations[k] / h
            constraints.append(points[k][i][1:] - points[k][i][:-1] == ci * points[k][i + 1])

        # Continuity and differentiability.
        if k < n_boxes - 1:
            for i in range(D + 1):
                constraints.append(points[k][i][-1] == points[k + 1][i][0])
                if i > 0:
                    continuity[k][i] = constraints[-1]

        # Cost function.
        for i, ai in alpha.items():
            h = n_points - 1 - i
            A = np.zeros((h + 1, h + 1))
            for m in range(h + 1):
                for n in range(h + 1):
                    A[m, n] = binom(h, m) * binom(h, n) / binom(2 * h, m + n)
            A *= durations[k] / (2 * h + 1)
            A = np.kron(A, np.eye(d))
            p = cp.vec(points[k][i], order='C')
            cost += ai * cp.quad_form(p, A)

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver='CLARABEL')

    # Reconstruct trajectory.
    beziers = []
    a = 0
    for k in range(n_boxes):
        b = a + durations[k]
        beziers.append(BezierCurve(points[k][0].value, a, b))
        a = b
    path = CompositeBezierCurve(beziers)

    retiming_weights = {}
    for k in range(n_boxes - 1):
        retiming_weights[k] = {}
        for i in range(1, D + 1):
            primal = points[k][i][-1].value
            dual = continuity[k][i].dual_value
            retiming_weights[k][i] = primal.dot(dual)

    # Reconstruct costs.
    cost_breakdown = {}
    for k in range(n_boxes):
        cost_breakdown[k] = {}
        bez = beziers[k]
        for i in range(1, D + 1):
            bez = bez.derivative()
            if i in alpha:
                cost_breakdown[k][i] = alpha[i] * bez.l2_squared()

    # Solution statistics.
    sol_stats = {}
    sol_stats['cost'] = prob.value
    sol_stats['runtime'] = prob.solver_stats.solve_time
    sol_stats['cost_breakdown'] = cost_breakdown
    sol_stats['retiming_weights'] = retiming_weights

    return path, sol_stats


def optimize_multiple_bezier(reach_region, aux_frames, L, U, durations, alpha, safe_points_lst,
                             fixed_frames=None, surface_normals_lst=None,
                             n_points=None, **kwargs):
    stance_foot = 'LF'

    # number of frames
    # n_frames = len(reach_region)
    n_frames = len(safe_points_lst[0].keys())

    # Problem size. Assume for now same number of boxes for all frames
    _, d = L[0][next(iter(L[0]))].shape
    num_boxes_tot = 0
    for bs_lst in L:
        num_boxes_tot += len(next(iter(bs_lst.values())))
    D = max(alpha)

    # Uncomment below when/if method with derivative inputs is implemented
    # assert max(initial) <= D
    # assert max(final) <= D

    if n_points is None:
        n_points = (D + 1) * 2

    # Control points of the curves and their derivatives.
    points = {}
    for k in range(num_boxes_tot * n_frames):
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
    for k in range(num_boxes_tot * n_frames):
        continuity[k] = {}

        # Update frame name and number of boxes within segment/interval
        f_name = frame_list[frame_idx]
        num_boxes_current, _ = L[seg_idx][f_name].shape

        # Box constraints apply in all cases
        Lk = np.array([L[seg_idx][f_name][fr_seg_k_box]] * n_points)
        Uk = np.array([U[seg_idx][f_name][fr_seg_k_box]] * n_points)

        constraints.append(points[k][0] >= Lk)
        constraints.append(points[k][0] <= Uk)

        # Enforce given positions
        if k % num_boxes_tot == 0:          # initial position for each frame
            # if also a fixed frame, repeat for entire segment duration
            if f_name in fixed_frames[seg_idx]:
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points, axis=0)
                constraints.append(points[k][0] == fixed_frame_pos_mat)
            else:   # assign for just the first time instant
                constraints.append(points[k][0][0] == safe_points_lst[0][f_name])   # initial position
                # check if it has a final safe point assigned
                if fr_seg_k_box == (num_boxes_current-1) and f_name in safe_points_lst[seg_idx+1].keys():
                    constraints.append(points[k][0][-1] == safe_points_lst[seg_idx+1][f_name])
                    add_vel_acc_constr(f_name, surface_normals_lst[seg_idx], points[k], constraints)
        elif (k + 1) % num_boxes_tot == 0:  # final position for each frame
            if f_name in fixed_frames[seg_idx]:
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points-1, axis=0)
                constraints.append(points[k][0][1:] == fixed_frame_pos_mat)
            else:
                constraints.append(points[k][0][-1] == safe_points_lst[-1][f_name])
                add_vel_acc_constr(f_name, surface_normals_lst[-1], points[k], constraints)
        else:       # safe and fixed positions at other times
            if f_name in fixed_frames[seg_idx]:
                fixed_frame_pos_mat = np.repeat(np.array([safe_points_lst[seg_idx][f_name]]), n_points-1, axis=0)
                constraints.append(points[k][0][1:] == fixed_frame_pos_mat)
            # Check if safe_point is available for the current frame
            elif f_name in safe_points_lst[seg_idx].keys():
                # Enforce (pre-computed) safe points at the end of each desired motion
                # note: the initial point within a segment is defined by the continuity constraint below
                if fr_seg_k_box == (num_boxes_current-1):
                    constraints.append(points[k][0][-1] == safe_points_lst[seg_idx+1][f_name])  # pos
                    add_vel_acc_constr(f_name, surface_normals_lst[seg_idx], points[k], constraints)

        # Bezier dynamics.
        for i in range(D):
            h = n_points - i - 1
            ci = durations[seg_idx][f_name][fr_seg_k_box] / h
            constraints.append(points[k][i][1:] - points[k][i][:-1] == ci * points[k][i + 1])

        # if we are in the same frame, enforce dynamics, continuity, differentiability, and cost
        if (k+1) % num_boxes_tot != 0:
            # Continuity and differentiability.
            if fr_seg_k_box < num_boxes_current:
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
        if (k+1) % num_boxes_tot == 0:
            frame_idx += 1
            seg_idx = 0
            fr_seg_k_box = 0
        else:           # move to next segment if this is the last box
            if fr_seg_k_box == (num_boxes_current - 1):   # or (k % num_boxes_current == 0)
                fr_seg_k_box = 0        # reset the box count
                seg_idx += 1            # increase segment
            else:
                fr_seg_k_box += 1

    # Rigid links (e.g., shin link length) constraint relaxation
    soc_constraint, cost_log_abs = [], []
    cost_log_abs_sum = 0.
    if aux_frames is not None:
        link_based_weights = np.array([0.1621, 0.006, 0.2808])    # based on distance between foot-shin frames

        # apply auxiliary rigid link constraint throughout all safe regions
        for aux_fr in aux_frames:
            prox_fr_idx, dist_fr_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_boxes_tot)

            # loop through all safe boxes
            for nb in range(1, num_boxes_tot):
                # for pnt in range(n_points-1):
                for pnt in range(1):
                    link_proximal_point = points[prox_fr_idx+nb][0][pnt]
                    link_distal_point = points[dist_fr_idx+nb][0][pnt]
                    create_bezier_cvx_norm_eq_relaxation(link_length, link_proximal_point,
                                             link_distal_point, soc_constraint, cost_log_abs,
                                                         wi=link_based_weights)

                    # knee should mostly be above the foot
                    # soc_constraint.append(link_proximal_point[2] - link_distal_point[2] <= 0.02)

        cost_log_abs_sum = -(cp.sum(cost_log_abs))

    # Reachability constraints
    # if reach_region is not None:
    #     fr_idx, k_fr_box = 0, 0
    #     for frame_name in frame_list:
    #         if frame_name == 'torso':
    #             coeffs = reach_region[frame_name]
    #             H = coeffs['H']
    #             d_vec = np.reshape(coeffs['d'], (len(H), 1))
    #             d_mat = np.repeat(d_vec, n_points, axis=1)
    #             num_boxes_current, _ = L[k_fr_box][frame_name].shape
    #             for idx_box in range(1, num_boxes_current - 1):
    #                 # torso translation terms
    #                 z_t_prev = points[idx_box - 1][0]
    #                 z_t_post = points[idx_box][0]
    #
    #                 # torso w.r.t. corresponding standing effector frame
    #                 z_stance_post = points[0][0]    # was points[fr_idx*n_boxes + idx_box][0]
    #                 # z_stance_post = points[n_boxes + idx_box][0]    # was points[fr_idx*n_boxes + idx_box][0]
    #
    #                 # reachable constraint
    #                 constraints.append(H @ z_t_prev.T - 2 * H @ z_t_post.T + H @ z_stance_post.T <= -d_mat)
    #             fr_idx += 1
    #
    #         else:
    #             coeffs = reach_region[frame_name]
    #             H = coeffs['H']
    #             d_vec = np.reshape(coeffs['d'], (len(H), 1))
    #             d_mat = np.repeat(d_vec, n_points, axis=1)
    #             num_boxes_current, _ = L[k_fr_box][frame_name].shape
    #             for idx_box in range(1, num_boxes_current - 1):
    #                 # torso translation terms
    #                 z_t_prev = points[idx_box - 1][0]
    #                 z_t_post = points[idx_box][0]
    #
    #                 # corresponding end effector frame index
    #                 z_ee_post = points[fr_idx*num_boxes_current + idx_box][0]
    #
    #                 # reachable constraint
    #                 constraints.append(H @ z_t_prev.T - H @ z_t_post.T + H @ z_ee_post.T <= -d_mat)
    #             fr_idx += 1

    # Solve problem.
    prob = cp.Problem(cp.Minimize(cost + cost_log_abs_sum), constraints + soc_constraint)
    prob.solve(solver='SCS')

    if prob.status == 'infeasible':
        print('Problem was infeasible. Retrying with relaxed tolerances.')
        prob.solve(solver='SCS', eps_rel=5e-2, eps_abs=5e-2)

    # Reconstruct trajectory.
    beziers, path = [], []
    a = 0
    fr_seg_k_box, frame_idx, seg_idx = 0, 0, 0
    frame_name = frame_list[frame_idx]
    for k in range(num_boxes_tot * n_frames):
        num_boxes_current, _ = L[seg_idx][frame_name].shape
        # move on to next segment after the current number of safe boxes
        if (fr_seg_k_box != 0) and fr_seg_k_box % num_boxes_current == 0 and seg_idx != (len(L)-1):
            seg_idx += 1
            fr_seg_k_box = 0

        # move on to next frame after all boxes processed for each frame
        if k != 0 and (k % num_boxes_tot) == 0:
            frame_idx += 1
            frame_name = frame_list[frame_idx]
            fr_seg_k_box = 0

        b = a + durations[seg_idx][frame_name][fr_seg_k_box]
        beziers.append(BezierCurve(points[k][0].value, a, b))
        a = b
        fr_seg_k_box += 1
        # skip the final positions, those are assigned later
        if (k + 1) % num_boxes_tot == 0:
            fr_seg_k_box = 0  # might be redundant
            seg_idx = 0
            path.append(copy.deepcopy(CompositeBezierCurve(beziers)))
            beziers.clear()
            a = 0

    # path = CompositeBezierCurve(beziers)

    retiming_weights = {}
    # n_box_counter = 0
    # for k in range(num_boxes_tot * n_frames - 1):
    #     retiming_weights[k] = {}
    #     if k == 0 or k % ((n_boxes - 1) + n_box_counter * n_boxes) != 0:
    #         for i in range(1, D + 1):
    #             primal = points[k][i][-1].value
    #             dual = continuity[k][i].dual_value
    #             retiming_weights[k][i] = primal.dot(dual)
    #     else:
    #         n_box_counter += 1

    # Reconstruct costs.
    cost_breakdown = {}
    # for k in range((num_boxes_tot - 1) * n_frames):
    #     cost_breakdown[k] = {}
    #     bez = beziers[k]
    #     for i in range(1, D + 1):
    #         bez = bez.derivative()
    #         if i in alpha:
    #             cost_breakdown[k][i] = alpha[i] * bez.l2_squared()

    # Solution statistics.
    sol_stats = {}
    sol_stats['cost'] = prob.value
    sol_stats['runtime'] = prob.solver_stats.solve_time
    sol_stats['cost_breakdown'] = cost_breakdown
    sol_stats['retiming_weights'] = retiming_weights

    return path, sol_stats


def optimize_multiple_bezier_iris(reach_region: dict[str: np.array, str: np.array],
                                  aux_frames: List[dict],
                                  iris_regions: dict[str: IrisRegionsManager],
                                  durations: List[dict[str, np.array]],
                                  alpha: dict[int: float],
                                  safe_points_lst: List[dict[str, np.array]],
                                  fixed_frames=None,
                                  surface_normals_lst=None,
                                  n_points=None, **kwargs):
    stance_foot = 'RF'

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
        link_based_weights = np.array([0.1621, 0.006, 0.2808])    # based on distance between foot-shin frames
        wi = np.array([500., 0.5, 50.])

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
                                                         wi=wi)

                    # knee should mostly be above the foot
                    # soc_constraint.append(link_proximal_point[2] - link_distal_point[2] <= 0.02)

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

    # path = CompositeBezierCurve(beziers)

    retiming_weights = {}
    # n_box_counter = 0
    # for k in range(num_iris_tot * n_frames - 1):
    #     retiming_weights[k] = {}
    #     if k == 0 or k % ((n_boxes - 1) + n_box_counter * n_boxes) != 0:
    #         for i in range(1, D + 1):
    #             primal = points[k][i][-1].value
    #             dual = continuity[k][i].dual_value
    #             retiming_weights[k][i] = primal.dot(dual)
    #     else:
    #         n_box_counter += 1

    # Reconstruct costs.
    cost_breakdown = {}
    # for k in range((num_iris_tot - 1) * n_frames):
    #     cost_breakdown[k] = {}
    #     bez = beziers[k]
    #     for i in range(1, D + 1):
    #         bez = bez.derivative()
    #         if i in alpha:
    #             cost_breakdown[k][i] = alpha[i] * bez.l2_squared()

    # Solution statistics.
    sol_stats = {}
    sol_stats['cost'] = prob.value
    sol_stats['runtime'] = prob.solver_stats.solve_time
    sol_stats['cost_breakdown'] = cost_breakdown
    sol_stats['retiming_weights'] = retiming_weights

    return path, sol_stats, points


def add_vel_acc_constr(f_name, seg_surface_normal, point, constraints, b_constr_accel=True):

    # check if we have multiple contacts occurring in the same segment
    if type(seg_surface_normal) is list:
        # figure out which normal we are currently using based on its frame name
        for i, sn in enumerate(seg_surface_normal):
            if f_name == sn.contact_frame_name:
                seg_surface_normal = sn
                break

            # if we reach this point, the current frame does not have an assigned contact surface at this segment
            if i == len(seg_surface_normal) - 1:
                print(f'{f_name} motion frame not found in surface contact {seg_surface_normal.contact_frame_name}')
                return

    surf_normal = seg_surface_normal.surface_normal
    if seg_surface_normal is not None:
        # check that a normal vector has been specified for this frame and segment
        if f_name not in seg_surface_normal.contact_frame_name:
            surf_contact_name = seg_surface_normal.contact_frame_name
            print(f'{f_name} motion frame not found in {surf_contact_name} surface contact.'
                  f'No velocity/acceleration constraints added for this point.')
            return

        # apply epsilon motion constraint along specified direction
        if seg_surface_normal.b_initial_vel:
            frame_vel_ini = seg_surface_normal.get_contact_breaking_velocity()
            constraints.append(frame_vel_ini @ point[BezierParam.VEL.value][0] >= 0)

        # final velocity parallel to normal surface
        normal_mat = np.array([[0, -surf_normal[2], surf_normal[1]],
                               [surf_normal[2], 0, -surf_normal[0]],
                               [-surf_normal[1], surf_normal[0], 0]])
        constraints.append(normal_mat @ point[BezierParam.VEL.value][-1] == 0)
        # final velocity magnitude
        normal_tilde = (1. / eps_vel_constr) * surf_normal
        constraints.append(-normal_tilde @ point[BezierParam.VEL.value][-2] >= 0)
        # constraints.append(point[BezierParam.VEL.value][-1] == - eps_vel_constr * np.sign(surf_normal))
    if b_constr_accel:
        # apply only strictly positive and negative accelerations
        if surf_normal[Axis.X.value] > 0:
            constraints.append(point[BezierParam.ACC.value][-1][Axis.X.value] >= 0.)  # pos acc
        elif surf_normal[Axis.X.value] < 0:
            constraints.append(point[BezierParam.ACC.value][-1][Axis.X.value] <= 0.)  # neg acc
        else:
            constraints.append(point[BezierParam.ACC.value][-1][Axis.X.value] == 0.)  # zero acc

        if surf_normal[Axis.Y.value] > 0:
            constraints.append(point[BezierParam.ACC.value][-1][Axis.Y.value] >= 0.)  # pos acc
        elif surf_normal[Axis.Y.value] < 0:
            constraints.append(point[BezierParam.ACC.value][-1][Axis.Y.value] <= 0.)  # neg acc
        else:
            constraints.append(point[BezierParam.ACC.value][-1][Axis.Y.value] == 0.)  # zero acc

        if surf_normal[Axis.Z.value] > 0:
            constraints.append(point[BezierParam.ACC.value][-1][Axis.Z.value] >= 0.)  # pos acc
        elif surf_normal[Axis.Z.value] < 0:
            constraints.append(point[BezierParam.ACC.value][-1][Axis.Z.value] <= 0.)  # neg acc
        else:
            constraints.append(point[BezierParam.ACC.value][-1][Axis.Z.value] == 0.)  # zero acc


def retiming(kappa, costs, durations, retiming_weights, **kwargs):

    # Decision variables.
    n_boxes = max(costs) + 1
    eta = cp.Variable(n_boxes)
    eta.value = np.ones(n_boxes)
    constr = [durations @ eta == sum(durations)]

    # Scale costs from previous trajectory.
    cost = 0
    for i, ci in costs.items():
        for j, cij in ci.items():
            cost += cij * cp.power(eta[i], 1 - 2 * j)

    # Retiming weights.
    for k in range(n_boxes - 1):
        for i, w in retiming_weights[k].items():
            cost += i * retiming_weights[k][i] * (eta[k + 1] - eta[k])

    # Trust region.
    if not np.isinf(kappa):
        constr.append(eta[1:] - eta[:-1] <= kappa)
        constr.append(eta[:-1] - eta[1:] <= kappa)
        
    # Solve SOCP and get new durations.
    prob = cp.Problem(cp.Minimize(cost), constr)
    prob.solve(solver='CLARABEL')
    new_durations = np.multiply(eta.value, durations)

    # New candidate for kappa.
    kappa_max = max(np.abs(eta.value[1:] - eta.value[:-1]))

    return new_durations, prob.solver_stats.solve_time, kappa_max

def log(s1, size=10):
    s1 = str(s1)
    s0 = list('|' + ' ' * size + '|')
    s0[2:2 + len(s1)] = s1
    return ''.join(s0)

def init_log():
    print(log('Iter.') + log('Cost') + log('Decr.') + \
          log('Kappa') + log('Accept'))
    print('-' * 60)

def term_log():
    print('-' * 60)

def update_log(i, cost, cost_decrease, kappa, accept):
    print(log(i) + \
              log('{:.2e}'.format(cost)) + \
              log('{:.1e}'.format(cost_decrease)) + \
              log('{:.1e}'.format(kappa)) + \
              log(accept))

def optimize_bezier_with_retiming(L, U, durations, alpha, initial, final,
    omega=3, kappa_min=1e-2, verbose=False, **kwargs):

    # Solve initial Bezier problem.
    path, sol_stats = optimize_bezier(L, U, durations, alpha, initial, final, **kwargs)
    cost = sol_stats['cost']
    cost_breakdown = sol_stats['cost_breakdown']
    retiming_weights = sol_stats['retiming_weights']

    if verbose:
        init_log()
        update_log(0, cost, np.nan, np.inf, True)

    # Lists to populate.
    costs = [cost]
    paths = [path]
    durations_iter = [durations]
    bez_runtimes = [sol_stats['runtime']]
    retiming_runtimes = []

    # Iterate retiming and Bezier.
    kappa = 1
    n_iters = 0
    i = 1
    while True:
        n_iters += 1

        # Retime.
        new_durations, runtime, kappa_max = retiming(kappa, cost_breakdown,
            durations, retiming_weights, **kwargs)
        durations_iter.append(new_durations)
        retiming_runtimes.append(runtime)

        # Improve Bezier curves.
        path_new, sol_stats = optimize_bezier(L, U, new_durations,
            alpha, initial, final, **kwargs)
        cost_new = sol_stats['cost']
        costs.append(cost_new)
        paths.append(path_new)
        bez_runtimes.append(sol_stats['runtime'])

        decr = cost_new - cost
        accept = decr < 0
        if verbose:
            update_log(i, cost_new, decr, kappa, accept)

        # If retiming improved the trajectory.
        if accept:
            durations = new_durations
            path = path_new
            cost = cost_new
            cost_breakdown = sol_stats['cost_breakdown']
            retiming_weights = sol_stats['retiming_weights']

        if kappa < kappa_min:
            break
        kappa = kappa_max / omega
        i += 1

    runtime = sum(bez_runtimes) + sum(retiming_runtimes)
    if verbose:
        term_log()
        print(f'Smooth phase terminated in {i} iterations')
        print(f'Final cost is ' + '{:.3e}'.format(cost))
        print(f'Solver time was {np.round(runtime, 5)}')

    # Solution statistics.
    sol_stats = {}
    sol_stats['cost'] = cost
    sol_stats['n_iters'] = n_iters
    sol_stats['costs'] = costs
    sol_stats['paths'] = paths
    sol_stats['durations_iter'] = durations_iter
    sol_stats['bez_runtimes'] = bez_runtimes
    sol_stats['retiming_runtimes'] = retiming_runtimes
    sol_stats['runtime'] = runtime
    
    return path, sol_stats


def optimize_multiple_bezier_with_retiming(S, R, A, box_seq, durations, alpha, safe_points_lst,
                                           fixed_frames=None, surface_normals_lst=None,
                                           omega=3, kappa_min=1e-2, verbose=False, **kwargs):
    L, U = [], []
    b_i = 0
    for bs in box_seq:
        L.append({})
        U.append({})
        for frame, i in bs.items():
            L[b_i][frame] = np.array([S[frame].B.boxes[j].l for j in i])
            U[b_i][frame] = np.array([S[frame].B.boxes[j].u for j in i])
        b_i += 1

        # Boundary conditions.
        # initial = {0: initial[frame]}
        # final = {0: final[frame]}

    # Solve initial Bezier problem.
    # path, sol_stats = optimize_bezier(L[frame], U[frame], durations[frame], alpha, initial[frame], final[frame], **kwargs)
    path, sol_stats = optimize_multiple_bezier(R, A, L, U, durations, alpha, safe_points_lst,
                                               fixed_frames, surface_normals_lst, **kwargs)
    cost = sol_stats['cost']
    cost_breakdown = sol_stats['cost_breakdown']
    retiming_weights = sol_stats['retiming_weights']

    if verbose:
        init_log()
        update_log(0, cost, np.nan, np.inf, True)

    # Lists to populate.
    costs = [cost]
    paths = [path]
    durations_iter = [durations]
    bez_runtimes = [sol_stats['runtime']]
    retiming_runtimes = []

    # Iterate retiming and Bezier.
    kappa = 1
    n_iters = 0
    i = 1
    # while True:
    #     n_iters += 1
    #
    #     # Retime.
    #     new_durations, runtime, kappa_max = retiming(kappa, cost_breakdown,
    #         durations, retiming_weights, **kwargs)
    #     durations_iter.append(new_durations)
    #     retiming_runtimes.append(runtime)
    #
    #     # Improve Bezier curves.
    #     path_new, sol_stats = optimize_multiple_bezier(R, A, L, U, new_durations,
    #         alpha, initial, final, **kwargs)
    #     cost_new = sol_stats['cost']
    #     costs.append(cost_new)
    #     paths.append(path_new)
    #     bez_runtimes.append(sol_stats['runtime'])
    #
    #     decr = cost_new - cost
    #     accept = decr < 0
    #     if verbose:
    #         update_log(i, cost_new, decr, kappa, accept)
    #
    #     # If retiming improved the trajectory.
    #     if accept:
    #         durations = new_durations
    #         path = path_new
    #         cost = cost_new
    #         cost_breakdown = sol_stats['cost_breakdown']
    #         retiming_weights = sol_stats['retiming_weights']
    #
    #     if kappa < kappa_min:
    #         break
    #     kappa = kappa_max / omega
    #     i += 1

    runtime = sum(bez_runtimes) + sum(retiming_runtimes)
    if verbose:
        term_log()
        print(f'Smooth phase terminated in {i} iterations')
        print(f'Final cost is ' + '{:.3e}'.format(cost))
        print(f'Solver time was {np.round(runtime, 5)}')

    # Solution statistics.
    sol_stats = {}
    sol_stats['cost'] = cost
    sol_stats['n_iters'] = n_iters
    sol_stats['costs'] = costs
    sol_stats['paths'] = paths
    sol_stats['durations_iter'] = durations_iter
    sol_stats['bez_runtimes'] = bez_runtimes
    sol_stats['retiming_runtimes'] = retiming_runtimes
    sol_stats['runtime'] = runtime

    return path, sol_stats
