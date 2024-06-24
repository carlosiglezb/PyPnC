import numpy as np
from pnc.planner.multicontact.fastpathplanning.boxes import Box, BoxCollection
from pnc.planner.multicontact.fastpathplanning.polygonal import iterative_planner, iterative_planner_multiple, \
    solve_min_reach_iris_distance
from pnc.planner.multicontact.fastpathplanning.smooth import optimize_bezier_with_retiming, \
    optimize_multiple_bezier_with_retiming, optimize_multiple_bezier_iris
import copy

from vision.iris.iris_regions_manager import IrisRegionsManager


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
    box_p_init = boxes.B.contain(p_init)
    if box_p_init == boxes.B.contain(p_term):
        box_seq = list(box_p_init)
    else:
        box_seq, length, runtime = discrete_planner(p_init)

    if box_seq is None:
        print('Infeasible safe problem, initial and terminal points are disconnected.')
        return

    return box_seq


def distribute_box_seq(box_seq_all_frames, b_max):
    for frame, seq in box_seq_all_frames.items():
        if not np.isnan(seq[0]) and len(seq) != b_max:
            # if all values are the same, simply copy up to b_max
            if np.mean(seq) == seq[-1]:
                box_seq_all_frames[frame] = [seq[-1]] * b_max
            else:
                # approach 1: repeat the last entry until we match the size b_max
                last_box = seq[-1]
                missing_entries = b_max - len(seq)
                box_seq_all_frames[frame] = seq + missing_entries * [last_box]


def get_last_defined_point(safe_points_list, frame_name):
    for sp in reversed(safe_points_list):
        if frame_name in sp.keys():
            return sp[frame_name]

    # if we reach this point, the corresponding frame is never assigned
    return 0


def get_last_defined_box(box_seq_list, frame_name):
    for bs in reversed(box_seq_list):
        if frame_name in bs.keys() and not any(np.isnan(bs[frame_name])):
            return bs[frame_name]

    # if we reach this point, the corresponding frame is never assigned
    print(f'Frame {frame_name} is never assigned in the box sequence list')
    return np.nan


def get_num_unassigned_boxes(box_seq_list, frame_name):
    num_unassigned_boxes = 1
    for bs in reversed(box_seq_list):
        if any(np.isnan(bs[frame_name])):
            num_unassigned_boxes += 1   # update location of last nan box
        else:
            return num_unassigned_boxes


def unassigned_box_seq_interpolator(box_seq_list, last_box_seq, frame_name):
    num_boxes_unassigned = get_num_unassigned_boxes(box_seq_list, frame_name)

    last_defined_box_seq = get_last_defined_box(box_seq_list, frame_name)

    # if there are no unassigned boxes, and the box seq list contains nan, over-write it
    if num_boxes_unassigned is None:
        if np.isnan(last_defined_box_seq):
            if last_box_seq[frame_name][0] is not None:
                last_defined_box_seq = [last_box_seq[frame_name][0]]
                num_boxes_unassigned = 1
                print('[warning] Double-check the box sequence list')

    # check last defined box sequence is consistent
    if last_defined_box_seq[0] != last_box_seq[frame_name][0]:
        print(f"[warning] Box sequence in {frame_name} free frame is inconsistent. "
              f"Check that seeds for {frame_name} frame are contained in both IRIS regions")

    interval_boxes = last_box_seq[frame_name][-1] - last_box_seq[frame_name][0]
    fract_box = interval_boxes / num_boxes_unassigned

    # distribute boxes equally
    k_box = 0
    for bs in box_seq_list:
        new_box_seq_val = round(last_defined_box_seq[0] + k_box * fract_box)
        b_max = np.max([len(boxes) for boxes in bs.values()])
        bs[frame_name] = [new_box_seq_val] * b_max
        k_box += 1

    # clear boxes from last box_seq
    last_box_seq[frame_name] = [last_box_seq[frame_name][-1]]


def distribute_free_frames(last_box_seq, box_seq_list, frame_name):

    # distribute according to the number of segments allocated
    unassigned_box_seq_interpolator(box_seq_list, last_box_seq, frame_name)


def plan_mulistage_box_seq(safe_boxes, fixed_frames, motion_frames, p_init):
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
            # get free frames and assign box containing initial position
            for fr, p0 in p_init.items():
                if fr not in motion_frames[0].keys() and fr not in fixed_frames[0]:
                    box_seq_dict[fr] = list(safe_boxes[fr].B.contain(p0))
            k_transition += 1
            continue

        safe_points_lst.append({})
        # first process the motion frames to determine the length of box sequences
        for fm, pm_next in motion_frames[k_transition-1].items():
            if fm in f_frames:
                pm_init = get_last_defined_point(safe_points_lst, fm)
                safe_points_lst[k_transition][fm] = pm_next
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

        if k_transition > 1:
            # if one of the old un-assigned frames got a new assignment, fill the gap
            for fname in p_init.keys():
                is_old_seg_unassigned = np.isnan(box_seq_lst[-1][fname][0])
                is_current_seg_assigned = not np.isnan(box_seq_dict[fname][0])
                if is_old_seg_unassigned and is_current_seg_assigned:
                    distribute_free_frames(box_seq_dict, box_seq_lst, fname)

        box_seq_lst.append(copy.deepcopy(box_seq_dict))
        box_seq_dict.clear()
        box_seq_dict = copy.deepcopy(box_seq_nan)
        k_transition += 1

    # append terminal motion frames
    safe_points_lst.append({})
    for fm, pm_next in motion_frames[k_transition-1].items():
        pm_init = get_last_defined_point(safe_points_lst, fm)
        safe_points_lst[k_transition][fm] = pm_next
        # if pm_init and pm_next are on the same box, no need to find the shortest path
        if safe_boxes[fm].B.contain(pm_next) == safe_boxes[fm].B.contain(pm_init):
            box_seq_dict[fm] = list(safe_boxes[fm].B.contain(pm_next))
        else:
            box_seq_dict[fm] = find_shortest_box_path(safe_boxes[fm], pm_init, pm_next)

    # check distribution of free motion frames over all segments/intervals
    for fname in p_init.keys():
        if np.isnan(box_seq_lst[-1][fname][0]):
            distribute_free_frames(box_seq_dict, box_seq_lst, fname)

    # re-assign block sequence (in case last motion frame changed b_max)
    b_max_new = np.max([len(bs) for bs in box_seq_lst[-1].values()])
    if b_max_new != b_max:
        distribute_box_seq(box_seq_lst[-1], b_max_new)
    b_min_new = np.min([len(bs) for bs in box_seq_lst[-1].values()])
    if b_max_new != b_min_new:
        distribute_box_seq(box_seq_lst[-1], b_max_new)

    # fill out last safe points based on fixed frames from last sequence
    b_max = np.max([len(bs) for bs in box_seq_dict.values()])
    for ff in fixed_frames[-1]:
        pf_prev = get_last_defined_point(safe_points_lst, ff)
        safe_points_lst[-1][ff] = pf_prev
        box_pf_prev = next(iter((safe_boxes[ff].B.contain(pf_prev))))
        box_seq_dict[ff] = [box_pf_prev] * b_max
    box_seq_lst.append(copy.deepcopy(box_seq_dict))

    # re-assign block sequence (in case last motion frame changed b_max)
    b_max_new = np.max([len(bs) for bs in box_seq_lst[-1].values()])
    if b_max_new != b_max:
        distribute_box_seq(box_seq_lst[-1], b_max_new)
    b_min_new = np.min([len(bs) for bs in box_seq_lst[-1].values()])
    if b_max_new != b_min_new:
        distribute_box_seq(box_seq_lst[-1], b_max_new)

    # throw exception if any frames have un-assigned safe regions
    for f_list in box_seq_lst:
        for fname, bs in f_list.items():
            if any(np.isnan(bs)):
                raise Exception(f"{fname} frame has un-assigned safe regions")

    return box_seq_lst, safe_points_lst


def plan_multistage_iris_seq(iris_regions: dict[str: IrisRegionsManager],
                             fixed_frames,
                             motion_frames,
                             p_init: dict[str: np.array]):
    # safe_regions_mgr should be of type IrisSafeSet
    safe_regions = iris_regions[next(iter(iris_regions))].getIrisRegions()
    box_seq_lst = []
    safe_points_lst = [p_init]
    d = safe_regions[0].domain_mut.ambient_dimension()

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
            # get free frames and assign box containing initial position
            for fr, p0 in p_init.items():
                if fr not in motion_frames[0].keys() and fr not in fixed_frames[0]:
                    box_seq_dict[fr] = list(iris_regions[fr].regionsContainingPoint(p0))
            k_transition += 1
            continue

        safe_points_lst.append({})
        # first process the motion frames to determine the length of box sequences
        for fm, pm_next in motion_frames[k_transition-1].items():
            if fm in f_frames:
                pm_init = get_last_defined_point(safe_points_lst, fm)
                safe_points_lst[k_transition][fm] = pm_next
                box_seq_dict[fm] = iris_regions[fm].findShortestPath(pm_init, pm_next)
                # box_seq_dict[fm] = find_shortest_iris_path(safe_regions[fm], pm_init, pm_next)
        # maximum number of boxes in a box sequence in the motion frames
        b_max = np.max([len(bs) for bs in box_seq_dict.values()])

        # check that all motion frames so far have the same length
        for fn, fs in box_seq_dict.items():
            # ignore if entry is still nan (i.e., hasn't been assigned, yet)
            if not np.isnan(fs[0]) and len(fs) != b_max:
                distribute_box_seq(box_seq_dict, b_max)

        # then, go through the fixed frames from previous state
        for ff in fixed_frames[k_transition-1]:
            # for a fixed frame, the shortest path is the box that contains the point
            pf_prev = safe_points_lst[k_transition - 1][ff]
            safe_points_lst[k_transition][ff] = pf_prev
            box_pf_prev = next(iter((iris_regions[ff].regionsContainingPoint(pf_prev))))
            box_seq_dict[ff] = [box_pf_prev] * b_max

        if k_transition > 1:
            # if one of the old un-assigned frames got a new assignment, fill the gap
            for fname in p_init.keys():
                is_old_seg_unassigned = np.isnan(box_seq_lst[-1][fname][0])
                is_current_seg_assigned = not np.isnan(box_seq_dict[fname][0])
                if is_old_seg_unassigned and is_current_seg_assigned:
                    distribute_free_frames(box_seq_dict, box_seq_lst, fname)

        box_seq_lst.append(copy.deepcopy(box_seq_dict))
        box_seq_dict.clear()
        box_seq_dict = copy.deepcopy(box_seq_nan)
        k_transition += 1

    # append terminal motion frames
    safe_points_lst.append({})
    for fm, pm_next in motion_frames[k_transition-1].items():
        pm_init = get_last_defined_point(safe_points_lst, fm)
        safe_points_lst[k_transition][fm] = pm_next
        # if pm_init and pm_next are on the same box, no need to find the shortest path
        if iris_regions[fm].regionsContainingPoint(pm_next) == iris_regions[fm].regionsContainingPoint(pm_init):
            box_seq_dict[fm] = list(iris_regions[fm].regionsContainingPoint(pm_next))
        else:
            box_seq_dict[fm] = iris_regions[fm].findShortestPath(pm_init, pm_next)
            # box_seq_dict[fm] = find_shortest_iris_path(safe_regions[fm], pm_init, pm_next)

    # check motion frames for consistency
    for mf_idx, mf in enumerate(motion_frames):
        for mframe, mpos in mf.items():
            # add it to the safe points list if it's not already there
            if mframe not in safe_points_lst[mf_idx+1].keys():
                safe_points_lst[mf_idx+1][mframe] = mpos

    # if there were no fixed frames (only motion frames), return solution
    # TODO: this might still fail if there were any free frames, fix later
    if len(box_seq_lst) == 0:
        for fname, ir in iris_regions.items():
            ir.iris_idx_seq = [box_seq_dict[fname]]
        return [box_seq_dict], safe_points_lst

    # check distribution of free motion frames over all segments/intervals
    for fname in p_init.keys():
        if np.isnan(box_seq_lst[-1][fname][0]):
            distribute_free_frames(box_seq_dict, box_seq_lst, fname)

    # re-assign block sequence (in case last motion frame changed b_max)
    b_max_new = np.max([len(bs) for bs in box_seq_lst[-1].values()])
    if b_max_new != b_max:
        distribute_box_seq(box_seq_lst[-1], b_max_new)
    b_min_new = np.min([len(bs) for bs in box_seq_lst[-1].values()])
    if b_max_new != b_min_new:
        distribute_box_seq(box_seq_lst[-1], b_max_new)

    # fill out last safe points based on fixed frames from last sequence
    b_max = np.max([len(bs) for bs in box_seq_dict.values()])
    for ff in fixed_frames[-1]:
        pf_prev = get_last_defined_point(safe_points_lst, ff)
        safe_points_lst[-1][ff] = pf_prev
        box_pf_prev = next(iter((iris_regions[ff].regionsContainingPoint(pf_prev))))
        box_seq_dict[ff] = [box_pf_prev] * b_max
    box_seq_lst.append(copy.deepcopy(box_seq_dict))

    # re-assign block sequence (in case last motion frame changed b_max)
    b_max_new = np.max([len(bs) for bs in box_seq_lst[-1].values()])
    if b_max_new != b_max:
        distribute_box_seq(box_seq_lst[-1], b_max_new)
    b_min_new = np.min([len(bs) for bs in box_seq_lst[-1].values()])
    if b_max_new != b_min_new:
        distribute_box_seq(box_seq_lst[-1], b_max_new)

    # throw exception if any frames have un-assigned safe regions
    for f_list in box_seq_lst:
        for fname, bs in f_list.items():
            if any(np.isnan(bs)):
                raise Exception(f"{fname} frame has un-assigned safe regions or goal")

    # save iris sequence to IrisRegionsManager
    for fname, ir in iris_regions.items():
        ir.iris_idx_seq.clear()
        for seg in range(len(box_seq_lst)):
            ir.iris_idx_seq.append(box_seq_lst[seg][fname])

    return box_seq_lst, safe_points_lst


def plan_multiple(S, R, p_init, T, alpha,
                  verbose=True, A=None, fixed_frames=None, motion_frames_seq=None):

    if verbose:
        print('Polygonal phase:')

    motion_frames_lst = motion_frames_seq.get_motion_frames()
    box_seq, safe_pnt_lst = plan_mulistage_box_seq(S, fixed_frames, motion_frames_lst, p_init)
    box_seq, traj, length, solver_time = iterative_planner_multiple(S, R, safe_pnt_lst,
                                                        box_seq, verbose, A)

    if verbose:
        print('\nSmooth phase:')

    # Fix box sequence.
    n_f = len(p_init)
    n_poly_points = round(len(traj) / n_f)
    durations = []
    bs_i = 0
    seg_idx = 0
    for bs in box_seq:
        durations.append({})
        for frame, i in bs.items():
            # Cost coefficients.
            alpha = {i + 1: ai for i, ai in enumerate(alpha)}

            # Initialize transition times.
            num_boxes = len(i)
            frame_idx = list(p_init.keys()).index(frame)
            d = S[frame].B.boxes[i[0]].d
            # get indices of current frame for all curve points
            for b in range(num_boxes+1):
                first_idx = n_poly_points * frame_idx + (b + bs_i) * d
                last_idx = first_idx + d - 1
                if b == 0:
                   ee_traj_idx = np.linspace(first_idx, last_idx, d).astype(int)
                else:
                    ee_traj_idx = np.vstack((ee_traj_idx, (np.linspace(first_idx, last_idx, d)).astype(int)))

            ee_traj_change = traj[ee_traj_idx[1:]]-traj[ee_traj_idx[:-1]]
            durations[seg_idx][frame] = np.linalg.norm(ee_traj_change, axis=1)
            #TODO deal with case where any of durations[frame] == 0
            durations[seg_idx][frame] *= T / sum(durations[seg_idx][frame])
        bs_i += num_boxes
        seg_idx += 1

    surface_normals_lst = motion_frames_seq.get_contact_surfaces()
    paths, sol_stats = optimize_multiple_bezier_with_retiming(S, R, A, box_seq, durations,
                                                             alpha, safe_pnt_lst,
                                                             fixed_frames, surface_normals_lst,
                                                             verbose=verbose)

    return paths, box_seq


def plan_multiple_iris(S, R, p_init, T, alpha,
                  verbose=True, A=None, fixed_frames=None,
                  motion_frames_seq=None):
    # Find IRIS sequence and minimize length between safe points
    motion_frames_lst = motion_frames_seq.get_motion_frames()
    iris_seq, safe_pnt_lst = plan_multistage_iris_seq(S, fixed_frames, motion_frames_lst, p_init)
    traj, length, solver_time = solve_min_reach_iris_distance(R, S, iris_seq, safe_pnt_lst, A)

    if verbose:
        print(f"Min. distance solve time: {solver_time}")

    # Cost coefficients.
    alpha = {i + 1: ai for i, ai in enumerate(alpha)}

    # Fix box sequence.
    first_fr_iris = next(iter(S.values()))
    d = first_fr_iris.iris_list[0].iris_region.ambient_dimension()

    n_f = len(p_init)
    n_poly_points = round(len(traj) / n_f)
    durations = []
    ir_i = 0
    seg_idx = 0
    for ir_mgr in iris_seq:
        durations.append({})
        for frame, ir in ir_mgr.items():
            # Initialize transition times.
            num_iris = len(ir)
            frame_idx = list(p_init.keys()).index(frame)
            # get indices of current frame for all curve points
            for b in range(num_iris+1):
                first_idx = n_poly_points * frame_idx + (b + ir_i) * d
                last_idx = first_idx + d - 1
                if b == 0:
                   ee_traj_idx = np.linspace(first_idx, last_idx, d).astype(int)
                else:
                    ee_traj_idx = np.vstack((ee_traj_idx, (np.linspace(first_idx, last_idx, d)).astype(int)))

            ee_traj_change = traj[ee_traj_idx[1:]]-traj[ee_traj_idx[:-1]]
            durations[seg_idx][frame] = np.linalg.norm(ee_traj_change, axis=1)
            #TODO deal with case where any of durations[frame] == 0
            durations[seg_idx][frame] *= T / sum(durations[seg_idx][frame])
        ir_i += num_iris
        seg_idx += 1

    surface_normals_lst = motion_frames_seq.get_contact_surfaces()
    paths, sol_stats, points = optimize_multiple_bezier_iris(R, A, S, durations, alpha, safe_pnt_lst,
                                                     fixed_frames, surface_normals_lst, verbose=verbose)
    if verbose:
        print(f"Bezier solve time: {sol_stats['runtime']}")

    return paths, iris_seq, points
