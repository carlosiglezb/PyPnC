import copy
from typing import List
import scipy

import cvxpy as cp
import numpy as np

from ..cvx_mfpp_tools import get_aux_frame_idx, create_cvx_norm_eq_relaxation
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
    x = cp.Variable(d * n_f * (num_iris_tot + 1 ))

    contact_constr = []
    x_init_idx = 0
    # re-write multi-stage goal points (locations) in terms of optimization variables
    for f_name in iris_regions.keys():  # go in order (hence, refer to an OrderedDict)
        for seg_idx in range(len(safe_points_list)):
            # for the other segments, assign next desired location to end of respective segment
            if f_name in safe_points_list[seg_idx].keys():
                contact_constr.append(x[x_init_idx:x_init_idx + d] == safe_points_list[seg_idx][f_name])
            if seg_idx != (len(safe_points_list) - 1):
                next_seg_len = len(iris_seq[seg_idx][f_name])
            else:
                next_seg_len = 1
            x_init_idx += d * next_seg_len

    # organize lower and upper state limits (include initial and final state bounds)
    iris_constr_torso, iris_constr_lf, iris_constr_rf, iris_constr_lk, iris_constr_rk, iris_constr_lh, iris_constr_rh = [], [], [], [], [], [], []
    x_init_idx = 0      # reset index counter
    for frame, ee_iris in iris_regions.items():
        # initial condition is given
        ir_seq_idx = iris_seq[0][frame][0]
        iris_current = ee_iris.iris_list[ir_seq_idx].iris_region
        A = iris_current.A()
        b = iris_current.b()
        # debug
        if frame == 'torso':
            iris_constr_torso.append(A @ x[x_init_idx:x_init_idx + d] <= b)
        elif frame == 'LF':
            iris_constr_lf.append(A @ x[x_init_idx:x_init_idx + d] <= b)
        elif frame == 'RF':
            iris_constr_rf.append(A @ x[x_init_idx:x_init_idx + d] <= b)
        elif frame == 'L_knee':
            iris_constr_lk.append(A @ x[x_init_idx:x_init_idx + d] <= b)
        elif frame == 'R_knee':
            iris_constr_rk.append(A @ x[x_init_idx:x_init_idx + d] <= b)
        elif frame == 'LH':
            iris_constr_lh.append(A @ x[x_init_idx:x_init_idx + d] <= b)
        elif frame == 'RH':
            iris_constr_rh.append(A @ x[x_init_idx:x_init_idx + d] <= b)
        x_init_idx += d
        for seg_idx in range(len(iris_seq)):
            curr_seg_len = len(iris_seq[seg_idx][frame])
            for ir_seg_count in range(curr_seg_len):
                ir_seq_idx = iris_seq[seg_idx][frame][ir_seg_count]
                iris_current = ee_iris.iris_list[ir_seq_idx].iris_region
                if curr_seg_len == 1:
                    A = iris_current.A()
                    b = iris_current.b()
                else:
                    # if there's another IRIS region in this contact sequence, intersect with it
                    if ir_seg_count != (curr_seg_len - 1):
                        ir_next_seq_idx = iris_seq[seg_idx][frame][ir_seg_count + 1]
                        iris_next = ee_iris.iris_list[ir_next_seq_idx].iris_region
                        # check if the two IRIS regions intersect
                        if iris_current.IntersectsWith(iris_next):
                            iris_intersect = iris_current.Intersection(iris_next, check_for_redundancy=True)
                        else:
                            raise ValueError(f"IRIS regions of {frame} in segment {seg_idx} do not intersect.")
                        A = iris_intersect.A()
                        b = iris_intersect.b()
                    else:
                        # last IRIS region in the sequence
                        A = iris_current.A()
                        b = iris_current.b()
                # debug
                if frame == 'torso':
                    iris_constr_torso.append(A @ x[x_init_idx:x_init_idx+d] <= b)
                elif frame == 'LF':
                    iris_constr_lf.append(A @ x[x_init_idx:x_init_idx+d] <= b)
                elif frame == 'RF':
                    iris_constr_rf.append(A @ x[x_init_idx:x_init_idx+d] <= b)
                elif frame == 'L_knee':
                    iris_constr_lk.append(A @ x[x_init_idx:x_init_idx+d] <= b)
                elif frame == 'R_knee':
                    iris_constr_rk.append(A @ x[x_init_idx:x_init_idx+d] <= b)
                elif frame == 'LH':
                    iris_constr_lh.append(A @ x[x_init_idx:x_init_idx+d] <= b)
                elif frame == 'RH':
                    iris_constr_rh.append(A @ x[x_init_idx:x_init_idx+d] <= b)
                x_init_idx += d

    # Construct end-effector reachability constraints (initial & final points specified)
    l_arm_reach_constr, r_arm_reach_constr, l_leg_reach_constr, r_leg_reach_constr, l_knee_reach_constr, r_knee_reach_constr = [], [], [], [], [], []
    x_curr_idx = 0      # reset index counter
    if reach is not None:
        for frame_idx, frame in enumerate(iris_regions.keys()):
            for ti in range(num_iris_tot + 1):
                # torso reachability is redundant
                if frame == 'torso':
                    x_curr_idx += d
                    continue

                # get corresponding torso indices
                t_curr_idx = 0 * (num_iris_tot + 1) * d + d * ti
                t_next_idx = t_curr_idx + d
                z_t = x[t_curr_idx: t_next_idx]

                # torso must be reachable from contact foot
                # Note: not including knee reachability eases infeasibility
                if frame == 'NaN' or frame == 'MaM':
                    x_curr_idx += d
                    continue
                else:
                    ee_curr_idx = frame_idx * (num_iris_tot + 1) * d + d * ti
                    ee_next_idx = ee_curr_idx + d
                z_ee = x[ee_curr_idx: ee_next_idx]
                coeffs = reach[frame]

                H = coeffs['H']
                d_vec = np.reshape(coeffs['d'], (len(H), ))
                if frame == 'LH':
                    l_arm_reach_constr.append(H @ (z_ee - z_t) <= -d_vec)
                elif frame == 'RH':
                    r_arm_reach_constr.append(H @ (z_ee - z_t) <= -d_vec)
                elif frame == 'LF':
                    l_leg_reach_constr.append(H @ (z_ee - z_t) <= -d_vec)
                elif frame == 'RF':
                    r_leg_reach_constr.append(H @ (z_ee - z_t) <= -d_vec)
                elif frame == 'L_knee':
                    l_knee_reach_constr.append(H @ (z_ee - z_t) <= -d_vec)
                elif frame == 'R_knee':
                    r_knee_reach_constr.append(H @ (z_ee - z_t) <= -d_vec)
                x_curr_idx += d

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
                aux_fr, frame_list, num_iris_tot+1)

            if not np.isnan(prox_idx):
                link_length += link_threshold  # threshold for relaxation
                # add convex relaxation of norm constraint
                A_soc_aux, d_soc_aux = create_cvx_norm_eq_relaxation(
                    prox_idx, dist_idx, link_length, d, num_iris_tot+1, x)

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
        p_fr_t = cp.reshape(x_fr, [d, num_iris_tot + 1], order='F')
        cost += cp.sum(cp.norm(p_fr_t[:, 1:] - p_fr_t[:, :-1], axis=1))

    # solve
    prob = cp.Problem(cp.Minimize(cost + cost_log_abs),
                      l_arm_reach_constr + r_arm_reach_constr + l_leg_reach_constr + r_leg_reach_constr +
                      contact_constr +
                      iris_constr_torso + iris_constr_lf + iris_constr_rf + iris_constr_lk + iris_constr_rk + iris_constr_lh + iris_constr_rh +
                      soc_constraint)
    prob.solve(solver='SCS')

    if prob.status == 'infeasible':
        print('Polygonal problem was infeasible. Retrying with relaxed tolerances.')
        prob.solve(solver='SCS', eps_rel=0.05, eps_abs=0.05)

        # debug info returning which constraints are violated with this relaxed threshold
        # IRIS region containment check
        ic_t, ic_lf, ic_rf, ic_lh, ic_rh, ic_lk, ic_rk = [], [], [], [], [], [], []
        for ic in iris_constr_torso:
            ic_t.append(scipy.linalg.norm(ic.residual))
        for ic in iris_constr_lf:
            ic_lf.append(scipy.linalg.norm(ic.residual))
        for ic in iris_constr_rf:
            ic_rf.append(scipy.linalg.norm(ic.residual))
        for ic in iris_constr_lk:
            ic_lk.append(scipy.linalg.norm(ic.residual))
        for ic in iris_constr_rk:
            ic_rk.append(scipy.linalg.norm(ic.residual))
        for ic in iris_constr_lh:
            ic_lh.append(scipy.linalg.norm(ic.residual))
        for ic in iris_constr_rh:
            ic_rh.append(scipy.linalg.norm(ic.residual))
        # reachability check
        rc_lf, rc_rf, rc_lk, rc_rk, rc_lh, rc_rh = [], [], [], [], [], []
        for rc in l_arm_reach_constr:
            rc_lh.append(scipy.linalg.norm(rc.residual))
        for rc in r_arm_reach_constr:
            rc_rh.append(scipy.linalg.norm(rc.residual))
        for rc in l_leg_reach_constr:
            rc_lf.append(scipy.linalg.norm(rc.residual))
        for rc in r_leg_reach_constr:
            rc_rf.append(scipy.linalg.norm(rc.residual))
        for rc in l_knee_reach_constr:
            rc_lk.append(scipy.linalg.norm(rc.residual))
        for rc in r_knee_reach_constr:
            rc_rk.append(scipy.linalg.norm(rc.residual))

    length = prob.value
    traj = x.value
    solver_time = prob.solver_stats.solve_time

    # check distance of knee and foot points at each curve point
    for Ai in A_soc_debug:
        opt_shin_len_err = np.linalg.norm(Ai @ traj) - link_length
        print(f"[Polygonal] Shin length discrepancy: {opt_shin_len_err}")

    if b_debug:
        yaml = YAML()
        file_loc = cwd + '/test' + '/poly_min_distance_data.yaml'

        traj_reshape = np.reshape(traj, [7, 30])
        with open(file_loc, 'w') as f:
            for p in traj_reshape:
                yaml.dump(p.tolist(), f)

    return traj, length, solver_time
