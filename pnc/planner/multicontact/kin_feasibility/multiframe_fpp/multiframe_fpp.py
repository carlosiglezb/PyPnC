import copy

import numpy as np

from pnc.planner.multicontact.kin_feasibility.fastpathplanning.fastpathplanning import distribute_box_seq, distribute_free_frames
from pnc.planner.multicontact.kin_feasibility.multiframe_fpp.mfpp_polygonal import solve_min_reach_iris_distance
from pnc.planner.multicontact.kin_feasibility.multiframe_fpp.mfpp_smooth import optimize_multiple_bezier_iris
from pnc.planner.multicontact.kin_feasibility.fpp_sequencer_tools import get_last_defined_point
from vision.iris.iris_regions_manager import IrisRegionsManager


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


def plan_multiple_iris(S, R, p_init, T, alpha,
                  verbose=True, A=None, fixed_frames=None,
                  motion_frames_seq=None, w_rigid=None, w_rigid_poly=None):
    # Find IRIS sequence and minimize length between safe points
    motion_frames_lst = motion_frames_seq.get_motion_frames()
    iris_seq, safe_pnt_lst = plan_multistage_iris_seq(S, fixed_frames, motion_frames_lst, p_init)
    traj, length, solver_time = solve_min_reach_iris_distance(R, S, iris_seq, safe_pnt_lst, A, w_rigid_poly)

    if verbose:
        print(f"[Compute Time] Min. distance solve time: {solver_time}")

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
                if frame == 'torso':
                    print("first, last idx: ", first_idx, "      ", last_idx)
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
                                                             fixed_frames, surface_normals_lst,
                                                             weights_rigid_link=w_rigid,
                                                             verbose=verbose)
    if verbose:
        print(f"[Compute Time] Bezier solve time: {sol_stats['runtime']}")

    return paths, iris_seq, points
