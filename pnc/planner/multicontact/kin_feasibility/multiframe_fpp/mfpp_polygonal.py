import copy
from typing import List

import cvxpy as cp
import numpy as np

from pnc.planner.multicontact.kin_feasibility.cvx_mfpp_tools import get_aux_frame_idx, create_cvx_norm_eq_relaxation
from vision.iris.iris_regions_manager import IrisRegionsManager

b_debug = False

if b_debug:
    from ruamel.yaml import YAML
    import os
    import sys
    cwd = os.getcwd()
    sys.path.append(cwd)


def solve_min_reach_iris_distance(reach: dict[str: np.array, str: np.array],
                                  iris_regions: dict[str: IrisRegionsManager],
                                  iris_seq: List[dict[str: int]],
                                  safe_points_list: List[dict[str: np.array]],
                                  contact_seq=None,
                                  aux_frames=None,
                                  weights_rigid: np.array = None) -> [np.array, np.float64, np.float64]:
    if weights_rigid is None:
        weights_rigid = 10*np.array([0.1621, 0., 0.0808])
        # weights_rigid = np.array([0.0808, 0., 0.1621])

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
    if reach is not None:
        for sp_idx, sp_lst in enumerate(safe_points_list):
            # assume ee are reachable at the beginning and end
            if sp_idx == 0 or sp_idx == len(safe_points_list) - 1:
                continue

            # get corresponding indices of torso optimization variable
            t_curr_idx = 0 * num_iris_tot * d + d * sp_idx
            t_next_idx = t_curr_idx + 3
            z_t = x[t_curr_idx: t_next_idx]

            for frame_name in sp_lst.keys():
                # get corresponding frame index
                frame_idx = list(iris_regions.keys()).index(frame_name)

                # torso must be reachable from contact foot
                # Note: not including knee reachability eases infeasibility
                if frame_name == 'torso' or frame_name == 'L_knee' or frame_name == 'R_knee':
                    continue
                else:
                    ee_curr_idx = frame_idx * num_iris_tot * d + d * sp_idx
                    ee_next_idx = ee_curr_idx + 3
                z_ee = x[ee_curr_idx: ee_next_idx]
                coeffs = reach[frame_name]

                H = coeffs['H']
                d_vec = np.reshape(coeffs['d'], (len(H), ))
                constr.append(H @ (z_ee - z_t) <= -d_vec)

    # add feasibility of end curve point? Perhaps since it depends on torso

    frame_list = list(safe_points_list[0].keys())
    # add rigid link constraint
    cost_log_abs = 0.
    soc_constraint = []
    A_soc_debug, d_soc_debug, cost_log_abs_list = [], [], []  # for debug purposes
    if aux_frames is not None:
        link_threshold = 0.05
        # w_i = cp.Parameter(pos=True, value=1.)
        # w_i = np.array([0.1621, 0.006, 0.0808])    # based on desired distance between foot-shin frames
        for aux_fr in aux_frames:
            # get corresponding indices of optimization variable
            prox_idx, dist_idx, link_length = get_aux_frame_idx(
                aux_fr, frame_list, num_iris_tot + 1)

            if not np.isnan(prox_idx):
                link_length += link_threshold  # threshold for relaxation
                # add convex relaxation of norm constraint
                A_soc_aux, d_soc_aux = create_cvx_norm_eq_relaxation(
                    prox_idx, dist_idx, link_length, d, num_iris_tot + 1, x)

                # concatenate A inequality matrices for debugging
                A_soc_debug += copy.deepcopy(A_soc_aux)
                d_soc_debug += copy.deepcopy(d_soc_aux)

        for Ai, di in zip(A_soc_debug, d_soc_debug):
            soc_constraint.append(cp.SOC(di, Ai @ x))
            for i in range(3):
                if weights_rigid[i] != 0.:
                    cost_log_abs_list.append(weights_rigid[i] * cp.log(Ai[i] @ x))

        cost_log_abs = -(cp.sum(cost_log_abs_list))

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
        prob.solve(solver='SCS', eps_rel=0.05, eps_abs=0.01)

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
