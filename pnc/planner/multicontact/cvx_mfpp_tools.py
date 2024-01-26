import numpy as np
import cvxpy as cp


def create_cvx_norm_eq_relaxation(rigid_pnt_1_idx, rigid_pnt_2_idx, length,
                                  x_deg, num_boxes, x, soc_constraints):
    w_i = cp.Parameter(pos=True, value=1.)
    A_soc = []
    d_soc = []

    x_dim = x.size
    # for fr in aux_frames:
    populate_rigid_link_constraint(rigid_pnt_1_idx, rigid_pnt_2_idx, length,
                                   x_deg, num_boxes, x_dim, A_soc, d_soc)

    # write this as a cost term, instead? otherwise we just have inequality
    epsi = float(0.005)
    zero = cp.Parameter(value=0.)
    cost_log_abs_list = []
    for Ai, di in zip(A_soc, d_soc):
        soc_constraints.append(cp.SOC(di, Ai @ x))
        ci = Ai[0]
        # soc_constraints.append(cp.SOC(ci @ x + epsi, zero))     # distal slightly behind proximal
        # non_zero_idx_x = np.where(Ai != 0)[1][0:2]
        cost_log_abs_list.append(w_i * cp.log(di.value + Ai[0] @ x))

    cost_log_abs = -(cp.sum(cost_log_abs_list))

    return cost_log_abs, A_soc


def create_bezier_cvx_norm_eq_relaxation(link_length, x_var_first_point, x_var_second_point,
                                         soc_constraint, cost_log_abs_list, wi=None):
    if wi is None:
        wi = [1., 1., 1.]

    d_i = cp.Parameter(pos=True, value=link_length)

    # construct inequality constraints and log cost term to approximate norm equality
    # soc_constraint.append(cp.SOC(d_i, x_var_first_point - x_var_second_point))
    soc_constraint.append(cp.norm(x_var_first_point - x_var_second_point, 2) <= d_i)

    # add log det cost to encourage/apply relaxed rigid link constraint
    # cost_log_abs_list.append(cp.log_det(cp.diag(wi @ (x_var_first_point - x_var_second_point))))
    cost_log_abs_list.append(cp.log(1000. * (x_var_first_point[0] - x_var_second_point[0] + d_i)))

    return


def get_aux_frame_idx(aux_fr, frame_list, num_boxes_tot):
    prox_fr_idx, dist_fr_idx = np.nan, np.nan
    if aux_fr['parent_frame'] == 'l_knee_fe_ld' and 'L_knee' in frame_list:
        L_kn_frame_idx = np.where(np.array(frame_list) == 'L_knee')[0][0]
        LF_frame_idx = np.where(np.array(frame_list) == 'LF')[0][0]
        prox_fr_idx = int(L_kn_frame_idx * num_boxes_tot)
        dist_fr_idx = int(LF_frame_idx * num_boxes_tot)
    elif aux_fr['parent_frame'] == 'r_knee_fe_ld' and 'R_knee' in frame_list:
        R_kn_frame_idx = np.where(np.array(frame_list) == 'R_knee')[0][0]
        RF_frame_idx = np.where(np.array(frame_list) == 'RF')[0][0]
        prox_fr_idx = int(R_kn_frame_idx * num_boxes_tot)
        dist_fr_idx = int(RF_frame_idx * num_boxes_tot)
    else:
        child_str = aux_fr['child_frame']
        print(f'Aux child frame {child_str} is not past of the frame list')
    link_length = aux_fr['length']
    return prox_fr_idx, dist_fr_idx, link_length


def populate_rigid_link_constraint(rigid_pnt_1_idx, rigid_pnt_2_idx, link_length,
                                   x_deg, num_boxes, x_dim, A_soc, d_soc):
    r"""
    Parameters
    -----------------

    aux_frame : Dictionary
        Must contain entries {  parent_frame: String,
                                child_frame: String,
                                length: float }
    A_soc : List
        Second Order Cone constraints
        .. math::
            \| A_{soc} x \| \leq d_soc
        defining the rigid link connection between
        parent_frame and child_frame
    d_soc : List
        Offset in Second Order Cone constraint
            .. math::
        \| A_{soc} x \| \leq d_soc
    x_dim : int
        Dimension of the optimization vector x at any one time instance
    num_boxes : int
        Number of safe, collision-free  boxes
    """

    # initialize
    d_j = cp.Parameter(pos=True)
    d_j.value = link_length

    # create constraint for all interior curve points
    for p in range(1, num_boxes):
        A_j = np.zeros((3, x_dim))  # (x,y,z) of current curve point
        start_idx_1 = x_deg * p + rigid_pnt_1_idx * num_boxes
        start_idx_2 = x_deg * p + rigid_pnt_2_idx * num_boxes
        A_j[:, start_idx_1:start_idx_1 + x_deg] = np.eye(3)
        A_j[:, start_idx_2:start_idx_2 + x_deg] = -np.eye(3)
        A_soc.append(A_j)
        d_soc.append(d_j)
