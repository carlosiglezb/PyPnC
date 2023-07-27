from collections import OrderedDict

import numpy as np
from pnc.planner.multicontact.fastpathplanning.boxes import Box, BoxCollection
from pnc.planner.multicontact.fastpathplanning.polygonal import iterative_planner, iterative_planner_multiple
from pnc.planner.multicontact.fastpathplanning.smooth import optimize_bezier_with_retiming

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

def plan_multiple(S, R, p_init, p_term, T, alpha, der_init={}, der_term={}, verbose=True):

    if verbose:
        print('Polygonal phase:')

    box_seq = OrderedDict()
    for frame, boxes in S.items():
        discrete_planner, runtime = boxes.G.shortest_path(p_term[frame])
        box_seq[frame], length, runtime = discrete_planner(p_init[frame])
        if box_seq[frame] is None:
            print('Infeasible problem, initial and terminal points are disconnected.')
            return

    box_seq, traj, length, solver_time = iterative_planner_multiple(S, R, p_init, p_term, box_seq, verbose)

    if verbose:
        print('\nSmooth phase:')

    # Fix box sequence.
    ee_idx = 0
    path, sol_stats = OrderedDict(), OrderedDict()
    for frame, i in box_seq.items():
        L = np.array([S[frame].B.boxes[i[j]].l for j in i])
        U = np.array([S[frame].B.boxes[i[j]].u for j in i])

        # Cost coefficients.
        alpha = {i + 1: ai for i, ai in enumerate(alpha)}

        # Boundary conditions.
        initial = {0: p_init[frame]} | der_init
        final = {0: p_term[frame]} | der_term

        # Initialize transition times. TODO use dimension "d" instead of indices below
        durations = np.linalg.norm(traj[ee_idx+1:ee_idx+4] - traj[ee_idx:ee_idx+3], axis=1)
        durations *= T / sum(durations)

        path[frame], sol_stats[frame] = optimize_bezier_with_retiming(L, U, durations, alpha, initial, final, verbose=True)

    return path