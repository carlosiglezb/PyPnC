import copy
from typing import List

import numpy as np

from .mfpp_polygonal import solve_min_reach_iris_distance
from .mfpp_smooth import optimize_multiple_bezier_iris, \
    optimize_multiple_bezier_iris_casadi, pack_points_for_single_vector
from vision.iris.iris_regions_manager import IrisRegionsManager
from ..fpp_sequencer_tools import get_last_defined_point, distribute_box_seq, distribute_free_frames
from ..self_collision_avoidance.sca_robot_geometry import SCARobotGeometry


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
                    pack_box_seq_from_point(1, box_seq_dict, box_seq_lst, fr, iris_regions, p0)
            k_transition += 1
            continue

        safe_points_lst.append({})
        # first process the motion frames to determine the length of box sequences
        for fm, pm_next in motion_frames[k_transition-1].items():
            if fm in f_frames:
                pm_init = get_last_defined_point(safe_points_lst, fm)
                safe_points_lst[k_transition][fm] = pm_next
                if len(box_seq_lst) >= 1 and (box_seq_lst[-1][fm][-1] is not np.nan):
                    box_seq_dict[fm] = iris_regions[fm].findShortestPath(pm_init, pm_next, box_seq_lst[-1][fm][-1])
                else:
                    box_seq_dict[fm] = iris_regions[fm].findShortestPath(pm_init, pm_next)
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
            pack_box_seq_from_point(b_max, box_seq_dict, box_seq_lst, ff, iris_regions, pf_prev)

        if k_transition > 1:
            # if one of the old un-assigned frames got a new assignment, fill the gap
            for fname in p_init.keys():
                is_old_seg_unassigned = np.isnan(box_seq_lst[-1][fname][0])
                is_current_seg_assigned = not np.isnan(box_seq_dict[fname][0])
                if is_old_seg_unassigned and is_current_seg_assigned:
                    # if last known IRIS region remained unchanged, simply copy accordingly
                    prev_prev_iris = box_seq_lst[-2][fname][0]
                    current_iris = box_seq_dict[fname][0]
                    if not np.isnan(prev_prev_iris) and prev_prev_iris == current_iris:
                        b_max_prev = np.max([len(bs) for bs in box_seq_lst[-1].values()])
                        box_seq_lst[-1][fname] = [current_iris] * b_max_prev
                    else:
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
        pack_box_seq_from_point(b_max, box_seq_dict, box_seq_lst, ff, iris_regions, pf_prev)
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

    # last check that all dimensions are the same
    for bs in box_seq_lst:
        b_max = np.max([len(bval) for bval in bs.values()])
        for v in bs.values():
            if len(v) < b_max:
                distribute_box_seq(bs, b_max)

    # save iris sequence to IrisRegionsManager
    for fname, ir in iris_regions.items():
        ir.iris_idx_seq.clear()
        for seg in range(len(box_seq_lst)):
            ir.iris_idx_seq.append(box_seq_lst[seg][fname])

    return box_seq_lst, safe_points_lst


def pack_box_seq_from_point(b_max, box_seq_dict, box_seq_lst, ff, iris_regions, pf_prev):
    box_pf_prev = (iris_regions[ff].regionsContainingPoint(pf_prev))
    if len(box_pf_prev) > 1:
        # check that one of the previous feasible regions is the same as the last from previous contact phase
        for bpp in box_pf_prev:
            if (len(box_seq_lst) == 0) or (bpp == box_seq_lst[-1][ff][-1]):
                box_seq_dict[ff] = [bpp] * b_max
                break
    else:
        box_seq_dict[ff] = box_pf_prev * b_max


def plan_multiple_iris(S, R, p_init, T, alpha,
                  verbose=True, A=None, fixed_frames=None,
                  motion_frames_seq=None,
                  sca_robot_geometry: SCARobotGeometry=None,
                  w_rigid=None, w_rigid_poly=None):
    # Find IRIS sequence and minimize length between safe points
    motion_frames_lst = motion_frames_seq.get_motion_frames()
    iris_seq, safe_pnt_lst = plan_multistage_iris_seq(S, fixed_frames, motion_frames_lst, p_init)

    contact_seq_polygonal = []
    for cont_seq_idx, ir_seq in enumerate(iris_seq):
        num_segs = len(next(iter(ir_seq.values())))
        current_ff_lst = fixed_frames[cont_seq_idx]
        for j in range(num_segs):
            if current_ff_lst[0] != 'torso':
                contact_seq_polygonal.append([current_ff_lst[0]])
            else:
                contact_seq_polygonal.append([current_ff_lst[1]])
    traj, length, solver_time = solve_min_reach_iris_distance(R, S, iris_seq, safe_pnt_lst,
                                                              contact_seq=contact_seq_polygonal,
                                                              aux_frames=A,
                                                              weights_rigid=w_rigid_poly)

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
            if num_iris <= 2:
                # get indices of current frame for all curve points
                for b in range(num_iris):
                    first_idx = n_poly_points * frame_idx + (b + ir_i) * d
                    last_idx = first_idx + d - 1
                    if b == 0:
                       ee_traj_idx = np.linspace(first_idx, last_idx, d).astype(int)
                    else:
                        ee_traj_idx = np.vstack((ee_traj_idx, (np.linspace(first_idx, last_idx, d)).astype(int)))
            else:
                # we used intersection so we should take additional point into account
                for b in range(num_iris + 1):
                    first_idx = n_poly_points * frame_idx + (b + ir_i) * d - d
                    last_idx = first_idx + d - 1
                    if b == 0:
                       ee_traj_idx = np.linspace(first_idx, last_idx, d).astype(int)
                    else:
                        ee_traj_idx = np.vstack((ee_traj_idx, (np.linspace(first_idx, last_idx, d)).astype(int)))

            if seg_idx == len(iris_seq) - 1:
                # assumes the last sequence is a fixed frame to stabilize last motion
                durations[seg_idx][frame] = [float(T)]
            else:
                ee_traj_change = traj[ee_traj_idx[1:]]-traj[ee_traj_idx[:-1]]
                durations[seg_idx][frame] = np.linalg.norm(ee_traj_change, axis=1)
                #TODO deal with case where any of durations[frame] == 0
                durations[seg_idx][frame] *= T / sum(durations[seg_idx][frame])
        ir_i += num_iris
        seg_idx += 1

    # distribute contact sequence according to partitioned iris regions segments
    parsed_contact_seq = []
    for cont_seq_idx, ir_seq in enumerate(iris_seq):
        num_segs = len(next(iter(ir_seq.values())))
        current_ff_lst = fixed_frames[cont_seq_idx]
        if current_ff_lst[0] != 'torso':
            parsed_contact_seq.append([current_ff_lst[0]] * num_segs)
        else:
            parsed_contact_seq.append([current_ff_lst[1]] * num_segs)

    surface_normals_lst = motion_frames_seq.get_contact_surfaces()
    paths, sol_stats, points, dvars = optimize_multiple_bezier_iris(R, A, S, durations, alpha, safe_pnt_lst,
                                                             fixed_frames=fixed_frames,
                                                             contact_sequence=parsed_contact_seq,
                                                             surface_normals_lst=surface_normals_lst,
                                                             weights_rigid_link=w_rigid,
                                                             verbose=verbose)
    if verbose:
        print(f"[Compute Time] Bezier solve time: {sol_stats['runtime']}")

    initial_guess = {}
    initial_guess['x0'] = pack_points_for_single_vector(points, 'cvxpy')
    initial_guess['lam_g0'] = pack_points_for_single_vector(dvars['lam_g0'], 'cvxpy')
    initial_guess['lam_x0'] = dvars['lam_x0']
    paths, sol_stats, points, _ = optimize_multiple_bezier_iris_casadi(R, A, S, durations, alpha, safe_pnt_lst,
                                                             sca_robot_geometry,
                                                             fixed_frames=fixed_frames,
                                                             contact_sequence=parsed_contact_seq,
                                                             surface_normals_lst=surface_normals_lst,
                                                             weights_rigid_link=w_rigid,
                                                             initial_guess=initial_guess,
                                                             verbose=verbose)

    return paths, iris_seq, points, safe_pnt_lst
