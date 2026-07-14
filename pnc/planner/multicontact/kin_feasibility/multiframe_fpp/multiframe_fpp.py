import copy
from typing import List

import numpy as np

from .mfpp_polygonal import solve_min_reach_iris_distance
from .mfpp_smooth import optimize_multiple_bezier_iris, \
    optimize_multiple_bezier_iris_casadi, pack_points_for_single_vector
from vision.iris.iris_regions_manager import IrisRegionsManager
from ..fpp_sequencer_tools import get_last_defined_point, distribute_box_seq, distribute_free_frames
from ..planner_surface_contact import (get_contact_seq_from_fixed_frames_seq,
                                        get_contact_planes_from_motion_frames_seq)
from ..self_collision_avoidance.sca_robot_geometry import SCARobotGeometry
from ..stability_polytope_tools import StabilityPolytopeManager


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

    # check if iris region had global_iris. If so, assign it to contact sequence.
    # Only flatten to a single region when that region contains EVERY safe point
    # of the frame; otherwise keep the sequenced (per-segment) assignment.
    for fname in p_init.keys():
        frame_pts = [sp[fname] for sp in safe_points_lst if fname in sp]
        for gi in iris_regions[fname].global_iris:
            if all(iris_regions[fname].iris_list[gi[0]].isPointSafe(p) for p in frame_pts):
                for bs_i in range(len(box_seq_lst)):
                    for ir_region_num in range(len(box_seq_lst[bs_i][fname])):
                        box_seq_lst[bs_i][fname][ir_region_num] = gi[0]
                break

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


def _region_slack(mgr, ridx, p):
    """Normalized slack of point p inside region ridx (negative = outside)."""
    A = mgr.iris_list[ridx].iris_region.A()
    b = mgr.iris_list[ridx].iris_region.b()
    return float(np.min((b - A @ p) / np.linalg.norm(A, axis=1)))


def _junction_depth(mgr, ridx1, ridx2):
    """Largest t such that some point has >= t normalized slack in BOTH regions.

    Caps the erosion margins of two consecutive boxes with different regions so
    their eroded intersection stays nonempty (continuity remains feasible).
    """
    import cvxpy as cp
    A1 = mgr.iris_list[ridx1].iris_region.A()
    b1 = mgr.iris_list[ridx1].iris_region.b()
    A2 = mgr.iris_list[ridx2].iris_region.A()
    b2 = mgr.iris_list[ridx2].iris_region.b()
    n1 = np.linalg.norm(A1, axis=1)
    n2 = np.linalg.norm(A2, axis=1)
    x = cp.Variable(3)
    t = cp.Variable()
    prob = cp.Problem(cp.Maximize(t),
                      [A1 @ x + t * n1 <= b1, A2 @ x + t * n2 <= b2])
    try:
        prob.solve(solver='CLARABEL')
        return float(t.value) if t.value is not None else 0.
    except Exception:
        return 0.


def compute_sphere_containment_margins(iris_regions, safe_points_lst,
                                       fixed_frames, sca_robot_geometry,
                                       n_points=8, eps=5e-3):
    # eps: slack buffer kept at capped (pinned/junction) control points. Too
    # small (~1e-3) leaves the pins only ~1 mm inside the eroded regions and
    # makes the warm-start QP numerically degenerate for CLARABEL.
    """Per-frame, PER-CONTROL-POINT IRIS containment margins for sphere links.

    Eroding the containment of a control point by the frame's collision-sphere
    radius extends the Bezier convex-hull guarantee to the swept sphere: the
    IRIS regions exclude all obstacles, and the normalized region slack is
    concave, so the curve's slack is lower-bounded by the Bernstein-weighted
    combination of its control points' margins.

    margins[frame] is an (n_boxes, n_points) array, starting at the sphere
    radius everywhere and capped LOCALLY only where boundary conditions
    require it (so protection is not lost along whole swing segments):
      - control points that pin contact/boundary positions (initial/final
        positions, segment-boundary safe points, all points of fixed
        segments) are capped by that point's slack in the box's region;
      - the continuity-tied endpoint pair of consecutive boxes with different
        regions is capped by the junction depth (largest common slack), so
        the eroded regions still intersect there.
    Curve portions near a contact therefore approach the surface as required,
    while interior control points keep the full sphere-radius margin.
    """
    margins = {}
    if sca_robot_geometry is None:
        return margins
    n_seg = len(fixed_frames)
    for fr, mgr in iris_regions.items():
        if fr == 'torso' or not sca_robot_geometry.is_link_in_sca_list(fr):
            continue
        if sca_robot_geometry.get_primitive_shape_type(fr) != 'sphere':
            continue
        U = sca_robot_geometry.get_sphere_representation(fr)['U']
        radius = 1.0 / U[0, 0]      # sphere radius (U = I / r)

        # global box list: (segment, region index) in traversal order
        boxes = [(s, int(r)) for s in range(len(mgr.iris_idx_seq))
                 for r in mgr.iris_idx_seq[s]]
        n_boxes = len(boxes)
        m = np.full((n_boxes, n_points), radius)
        seg_first = {}      # segment -> first global box index
        seg_last = {}       # segment -> last global box index
        for k, (s, _) in enumerate(boxes):
            seg_first.setdefault(s, k)
            seg_last[s] = k

        def cap(k, j, p):
            """Cap margin of control point j (or all points, j=None) of box k."""
            slack = max(_region_slack(mgr, boxes[k][1], np.asarray(p)) - eps, 0.)
            if j is None:
                m[k, :] = np.minimum(m[k, :], slack)
            else:
                m[k, j] = min(m[k, j], slack)

        # initial / final pinned positions
        if fr in safe_points_lst[0]:
            cap(0, 0, safe_points_lst[0][fr])
        if fr in safe_points_lst[-1]:
            cap(n_boxes - 1, n_points - 1, safe_points_lst[-1][fr])
        for s in range(n_seg):
            b_fixed = fixed_frames[s] is not None and fr in fixed_frames[s]
            if b_fixed and fr in safe_points_lst[s]:
                # fixed segments pin the point at every control point
                for k in range(seg_first[s], seg_last[s] + 1):
                    cap(k, None, safe_points_lst[s][fr])
            if s >= 1 and fr in safe_points_lst[s]:
                # segment-boundary safe point: pinned at the first point of
                # segment s and, via continuity, the last point of segment s-1
                cap(seg_first[s], 0, safe_points_lst[s][fr])
                cap(seg_last[s - 1], n_points - 1, safe_points_lst[s][fr])

        # junction feasibility: the continuity-tied endpoint pair between
        # consecutive boxes with different regions must fit in both eroded
        # regions simultaneously
        for k in range(n_boxes - 1):
            if boxes[k][1] == boxes[k + 1][1]:
                continue
            depth = max(_junction_depth(mgr, boxes[k][1], boxes[k + 1][1]) - eps, 0.)
            m[k, n_points - 1] = min(m[k, n_points - 1], depth)
            m[k + 1, 0] = min(m[k + 1, 0], depth)

        # smoothness propagation: derivative continuity (up to order D) into or
        # out of a FIXED (constant-position) box forces the first/last D+1
        # control points of the neighboring box to coincide with the shared
        # pinned point (zero velocity/acc/jerk at the junction), so the
        # endpoint cap must extend to those control points too
        D = n_points // 2 - 1
        fixed_box = np.zeros(n_boxes, dtype=bool)
        for s in range(n_seg):
            if fixed_frames[s] is not None and fr in fixed_frames[s] and s in seg_first:
                fixed_box[seg_first[s]:seg_last[s] + 1] = True
        for k in range(n_boxes):
            if k > 0 and fixed_box[k - 1]:
                m[k, 1:D + 1] = np.minimum(m[k, 1:D + 1], m[k, 0])
            if k + 1 < n_boxes and fixed_box[k + 1]:
                m[k, n_points - 1 - D:n_points - 1] = np.minimum(
                    m[k, n_points - 1 - D:n_points - 1], m[k, n_points - 1])

        margins[fr] = m
    return margins


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
                  env_geometry=None,
                  w_rigid=None, w_rigid_poly=None,
                  b_use_knees_in_smooth_plan=False,
                  b_final_vel_constr=False,
                  b_use_stability_polytope=False,
                  robot_mass=None,
                  w_stability_polytope=0.0,
                  foot_force_lim=1.5,
                  hand_force_lim=0.25,
                  stab_poly_callback=None,
                  b_use_sphere_margins=True):
    solver_stats = {}
    # Find IRIS sequence and minimize length between safe points
    motion_frames_lst = motion_frames_seq.get_motion_frames()
    iris_seq, safe_pnt_lst = plan_multistage_iris_seq(S, fixed_frames, motion_frames_lst, p_init)

    # Build stability-polytope manager (computed once, reused by both optimisers)
    stab_poly_manager = None
    if b_use_stability_polytope and robot_mass is not None:
        stab_poly_manager = StabilityPolytopeManager.from_fixed_frames(
            fixed_frames, motion_frames_seq, robot_mass,
            n_phases_out=len(iris_seq),
            foot_force_lim=foot_force_lim,
            hand_force_lim=hand_force_lim)
        stab_poly_manager.compute(safe_pnt_lst)
        if verbose:
            print(f"[StabilityPolytope] computed {len(stab_poly_manager)} polytopes "
                  f"(robot_mass={robot_mass:.1f} kg)")
        if stab_poly_callback is not None:
            stab_poly_callback(stab_poly_manager)

    traj, length, solver_time = solve_min_reach_iris_distance(R, S, iris_seq, safe_pnt_lst,
                                                              aux_frames=A,
                                                              weights_rigid=w_rigid_poly)
    solver_stats['min_reach_iris_distance_cvxpy_time'] = solver_time
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

    # distribute contact sequence according to partitioned iris regions segments
    parsed_contact_seq = get_contact_seq_from_fixed_frames_seq(fixed_frames)

    surface_normals_lst = motion_frames_seq.get_contact_surfaces()

    # Robot-environment collision avoidance: sphere-radius containment margins
    # (swept-sphere avoidance via eroded IRIS containment). Convex, zero solve
    # cost, and guarantees the whole curve's sphere stays collision-free by the
    # Bezier convex-hull property. Self-collisions are handled separately by
    # the DCOL callback constraints in the casadi solve.
    containment_margins = {}
    if b_use_sphere_margins:
        containment_margins = compute_sphere_containment_margins(
            S, safe_pnt_lst, fixed_frames, sca_robot_geometry,
            n_points=(max(alpha) + 1) * 2)
    if containment_margins:
        print("[Smooth] Sphere containment margins (per box, min/max over ctrl pts): "
              + str({k: [(round(float(r.min()), 3), round(float(r.max()), 3)) for r in v]
                     for k, v in containment_margins.items()}))

    paths, sol_stats, points, dvars = optimize_multiple_bezier_iris(R, A, S, durations, alpha, safe_pnt_lst,
                                                             fixed_frames=fixed_frames,
                                                             contact_sequence=parsed_contact_seq,
                                                             surface_normals_lst=surface_normals_lst,
                                                             weights_rigid_link=w_rigid,
                                                             verbose=verbose,
                                                             b_use_knees_in_smooth_plan=b_use_knees_in_smooth_plan,
                                                             b_final_vel_constr=b_final_vel_constr,
                                                             stab_poly_manager=stab_poly_manager,
                                                             w_stability_polytope=w_stability_polytope,
                                                             containment_margins=containment_margins)
    solver_stats['multiple_bezier_iris_cvxpy_time'] = sol_stats['runtime']
    for key in ('stab_poly_violation', 'stab_poly_violation_time'):
        if key in sol_stats:
            solver_stats[key] = sol_stats[key]
    solver_stats['stab_poly_manager'] = stab_poly_manager  # None when not active
    if verbose:
        print(f"[Compute Time] Bezier solve time: {sol_stats['runtime']}")

    if sca_robot_geometry is not None:
        b_skip_sca = False   # TODO: automate by checking if there are collisions using current solution
        initial_guess = {}
        initial_guess['x0'] = pack_points_for_single_vector(points, 'cvxpy')
        initial_guess['lam_g0'] = pack_points_for_single_vector(dvars['lam_g0'], 'cvxpy')
        initial_guess['lam_x0'] = dvars['lam_x0']
        initial_guess['constraints_idx'] = dvars['constraints_idx']
        initial_guess['reach_constr_idx'] = dvars['reach_constr_idx']
        initial_guess['soc_constr_idx'] = dvars['soc_constr_idx']
        paths, sol_stats, points, _ = optimize_multiple_bezier_iris_casadi(R, A, S, durations, alpha, safe_pnt_lst,
                                                                 sca_robot_geometry,
                                                                 env_geometry=env_geometry,
                                                                 fixed_frames=fixed_frames,
                                                                 contact_sequence=parsed_contact_seq,
                                                                 surface_normals_lst=surface_normals_lst,
                                                                 weights_rigid_link=w_rigid,
                                                                 initial_guess=initial_guess,
                                                                 verbose=verbose,
                                                                 b_use_knees_in_smooth_plan=b_use_knees_in_smooth_plan,
                                                                 b_final_vel_constr=b_final_vel_constr,
                                                                 b_skip_sca=b_skip_sca,
                                                                 stab_poly_manager=stab_poly_manager,
                                                                 w_stability_polytope=w_stability_polytope,
                                                                 containment_margins=containment_margins)
        solver_stats['multiple_bezier_iris_sca_casadi_time'] = sol_stats['runtime']
        if not b_skip_sca:
            solver_stats['multiple_bezier_iris_sca_build_time'] = sol_stats['sca_build_time']
            solver_stats['multiple_bezier_iris_sca_construct_time'] = sol_stats['prob_construct_time']
        # Overwrite with casadi violation (more refined than cvxpy warm-start)
        for key in ('stab_poly_violation', 'stab_poly_violation_time'):
            if key in sol_stats:
                solver_stats[key] = sol_stats[key]
    return paths, iris_seq, points, safe_pnt_lst, solver_stats
