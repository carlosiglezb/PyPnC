from collections import OrderedDict

import numpy as np
from pnc.planner.multicontact.fastpathplanning.boxes import Box, BoxCollection
from pnc.planner.multicontact.fastpathplanning.polygonal import iterative_planner, iterative_planner_multiple
from pnc.planner.multicontact.fastpathplanning.smooth import optimize_bezier_with_retiming, optimize_multiple_bezier_with_retiming

class SafeSet:

    def __init__(self, L, U, verbose=True):

        if verbose:
            print(f'Preprocessing phase:')

        assert L.shape == U.shape
        boxes = [Box(l, u) for l, u in zip(L, U)]
        self.B = BoxCollection(boxes, verbose)
        self.G = self.B.line_graph(verbose)

    def plot2d(self, **kwargs):

        self.B.plot2d(**kwargs)

    def plot3d(self, ax):

        self.B.plot3d(ax)

def plan(S, p_init, p_term, T, alpha, der_init={}, der_term={}, verbose=True):

    if verbose:
        print('Polygonal phase:')

    discrete_planner, runtime = S.G.shortest_path(p_term)
    box_seq, length, runtime = discrete_planner(p_init)
    if box_seq is None:
        print('Infeasible problem, initial and terminal points are disconnected.')
        return
    box_seq, traj, length, solver_time = iterative_planner(S.B, p_init, p_term, box_seq, verbose)

    if verbose:
        print('\nSmooth phase:')

    # Fix box sequence.
    L = np.array([S.B.boxes[i].l for i in box_seq])
    U = np.array([S.B.boxes[i].u for i in box_seq])

    # Cost coefficients.
    alpha = {i + 1: ai for i, ai in enumerate(alpha)}

    # Boundary conditions.
    initial = {0: p_init} | der_init
    final = {0: p_term} | der_term

    # Initialize transition times.
    durations = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
    durations *= T / sum(durations)

    path, sol_stats = optimize_bezier_with_retiming(L, U, durations, alpha, initial, final, verbose=True)

    return path


def plan_multiple(S, R, p_init, p_term, T, alpha, der_init={}, der_term={},
                  verbose=True, A=None, fixed_frames=None, motion_frames=None):

    if verbose:
        print('Polygonal phase:')

    box_seq = OrderedDict()
    for frame, boxes in S.items():
        # discrete_planner, runtime = boxes.G.shortest_path(p_term[frame])
        # box_seq[frame], length, runtime = discrete_planner(p_init[frame])
        # if box_seq[frame] is None:
        #     print('Infeasible safe problem, initial and terminal points are disconnected.')
        #     return

        # If no fixed frames are provided, find the shortest paths for all frames
        # Also find the shortest path for all non-fixed frames
        if (fixed_frames is None) or (fixed_frames is not None and frame not in fixed_frames):
            discrete_planner, runtime = boxes.G.shortest_path(p_term[frame])
            box_seq[frame], length, runtime = discrete_planner(p_init[frame])
            if box_seq[frame] is None:
                print('Infeasible safe problem, initial and terminal points are disconnected.')
                return
        else:
            if frame in fixed_frames:
                box_seq[frame] = [0] * 2   # assume point is stationary on first box TODO generalize *2

    box_seq, traj, length, solver_time = iterative_planner_multiple(S, R, p_init, p_term,
                                                        box_seq, verbose, A)

    if verbose:
        print('\nSmooth phase:')

    # Fix box sequence.
    durations = {}
    for frame, i in box_seq.items():
        # Cost coefficients.
        alpha = {i + 1: ai for i, ai in enumerate(alpha)}

        # Initialize transition times.
        num_boxes = len(i)
        frame_idx = list(p_init.keys()).index(frame)
        n_f = len(p_init)
        d = S[frame].B.boxes[i[0]].d
        # get indices of current frame for all curve points
        for b in range(num_boxes+1):
            if b == 0:
                ee_traj_idx = np.linspace(d*frame_idx+b*d*n_f, d*frame_idx+d*(b*n_f+1)-1, d).astype(int)
            else:
                ee_traj_idx = np.vstack((ee_traj_idx, (np.linspace(d*frame_idx+b*d*n_f, d*frame_idx+d*(b*n_f+1)-1, d)).astype(int)))

        ee_traj_change = traj[ee_traj_idx[1:]]-traj[ee_traj_idx[:-1]]
        durations[frame] = np.linalg.norm(ee_traj_change, axis=1)
        #TODO deal with case where any of durations[frame] == 0
        durations[frame] *= T / sum(durations[frame])

    path, sol_stats = optimize_multiple_bezier_with_retiming(S, R, A, box_seq, durations,
                                                             alpha, p_init, p_term,
                                                             fixed_frames, motion_frames,
                                                             verbose=verbose)

    return path