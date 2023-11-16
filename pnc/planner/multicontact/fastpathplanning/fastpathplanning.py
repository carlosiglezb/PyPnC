from collections import OrderedDict

import numpy as np
from pnc.planner.multicontact.fastpathplanning.boxes import Box, BoxCollection
from pnc.planner.multicontact.fastpathplanning.polygonal import iterative_planner, iterative_planner_multiple
from pnc.planner.multicontact.fastpathplanning.smooth import optimize_bezier_with_retiming, optimize_multiple_bezier_with_retiming
import copy


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


def find_shortest_box_path(boxes, p_init, p_term):
    # find the shortest path for all non-fixed frames
    discrete_planner, runtime = boxes.G.shortest_path(p_term)
    box_seq, length, runtime = discrete_planner(p_init)
    if box_seq is None:
        print('Infeasible safe problem, initial and terminal points are disconnected.')
        return

    return box_seq


def distribute_box_seq(box_seq_all_frames, b_max):
    for frame, seq in box_seq_all_frames.items():
        if len(seq) != b_max:
            # if all values are the same, simply copy up to b_max
            if np.mean(seq) == seq[-1]:
                box_seq_all_frames[frame] = [seq[-1]] * b_max
            else:
                raise NotImplementedError


def plan_mulistage_box_seq(safe_boxes, fixed_frames, motion_frames,
                           p_init):
    box_seq_lst = []
    safe_points_lst = [p_init]
    d = safe_boxes[next(iter(safe_boxes))].B.boxes[0].d

    # create dictionary with all frames and positions initialized to zero
    frames_pos_dict, box_seq_dict = {}, {}
    for fname in p_init.keys():
        frames_pos_dict[fname] = np.zeros(d, )
        box_seq_dict[fname] = [np.nan]
    box_seq_nan = copy.deepcopy(box_seq_dict)

    # find box sequence for frames in the fixed_frame list
    k_transition = 0
    for f_frames in fixed_frames:
        # skip first set of fixed frames (initial stance)
        if k_transition == 0:
            k_transition += 1
            continue

        safe_points_lst.append({})
        # first process the motion frames to determine the length of box sequences
        for fm, pm_next in motion_frames[k_transition-1].items():
            if fm in f_frames:
                safe_points_lst[k_transition][fm] = pm_next
                pm_init = safe_points_lst[k_transition - 1][fm]
                box_seq_dict[fm] = find_shortest_box_path(safe_boxes[fm], pm_init, pm_next)
        # maximum number of boxes in a box sequence in the motion frames
        b_max = np.max([len(bs) for bs in box_seq_dict.values()])

        # then, go through the fixed frames from previous state
        for ff in fixed_frames[k_transition-1]:
            # for a fixed frame, the shortest path is the box that contains the point
            pf_prev = safe_points_lst[k_transition - 1][ff]
            safe_points_lst[k_transition][ff] = pf_prev
            box_pf_prev = next(iter((safe_boxes[ff].B.contain(pf_prev))))
            box_seq_dict[ff] = [box_pf_prev] * b_max

        box_seq_lst.append(copy.deepcopy(box_seq_dict))
        box_seq_dict.clear()
        box_seq_dict = copy.deepcopy(box_seq_nan)
        k_transition += 1

    # append terminal motion frames
    # k_transition -= 1
    safe_points_lst.append({})
    for fm, pm_next in motion_frames[k_transition-1].items():
        safe_points_lst[k_transition][fm] = pm_next
        pm_init = safe_points_lst[k_transition - 2][fm]     # TODO find last nan box
        box_seq_dict[fm] = find_shortest_box_path(safe_boxes[fm], pm_init, pm_next)

    # fill out last safe points based on fixed frames from last sequence
    b_max = np.max([len(bs) for bs in box_seq_lst[-1].values()])
    for ff in fixed_frames[-1]:
        pf_prev = safe_points_lst[-2][ff]
        safe_points_lst[-1][ff] = pf_prev
        box_pf_prev = next(iter((safe_boxes[ff].B.contain(pf_prev))))
        box_seq_dict[ff] = [box_pf_prev] * b_max
    box_seq_lst.append(copy.deepcopy(box_seq_dict))

    # re-assign block sequence (in case last motion frame changed b_max)
    b_max_new = np.max([len(bs) for bs in box_seq_lst[-1].values()])
    if b_max_new != b_max:
        distribute_box_seq(box_seq_lst[-1], b_max_new)

    # fix box sequence list frames that had unassigned blocks (e.g., with nan entries)
    for f_list in box_seq_lst:
        b_max = np.max([len(bs) for bs in f_list.values()])
        for fname, bs in f_list.items():
            if any(np.isnan(bs)):
                # look for the nearest non-nan boxes and fill the nan boxes with that information
                if f_list != box_seq_lst[-1]:
                    f_list[fname] = [box_seq_lst[-1][fname][0]] * b_max     # TODO find next non-nan box
                else:                   # last list corresponds to the motion frame, so we repeat the last box sequence
                    f_list[fname] = [box_seq_lst[-2][fname][-1]] * b_max

    return box_seq_lst, safe_points_lst


def plan_multiple(S, R, p_init, p_term, T, alpha, der_init={}, der_term={},
                  verbose=True, A=None, fixed_frames=None, motion_frames=None):

    if verbose:
        print('Polygonal phase:')

    box_seq, safe_pnt_lst = plan_mulistage_box_seq(S, fixed_frames, motion_frames, p_init)
    box_seq, traj, length, solver_time = iterative_planner_multiple(S, R, safe_pnt_lst,
                                                        box_seq, verbose, A)

    if verbose:
        print('\nSmooth phase:')

    # Fix box sequence.
    durations = []
    bs_i = 0
    for bs in box_seq:
        durations.append({})
        for frame, i in bs.items():
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
            durations[bs_i][frame] = np.linalg.norm(ee_traj_change, axis=1)
            #TODO deal with case where any of durations[frame] == 0
            durations[bs_i][frame] *= T / sum(durations[bs_i][frame])
        bs_i += 1

    paths, sol_stats = optimize_multiple_bezier_with_retiming(S, R, A, box_seq, durations,
                                                             alpha, safe_pnt_lst,
                                                             fixed_frames, motion_frames,
                                                             verbose=verbose)

    return paths, box_seq