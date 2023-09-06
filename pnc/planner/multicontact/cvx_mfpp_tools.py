import numpy as np
import cvxpy as cp


def create_cvx_norm_eq_relaxation(aux_frames, num_boxes, x_dim, x):
    d_i = cp.Parameter(pos=True)
    w_i = cp.Parameter(pos=True, value=1.)
    A_soc = []
    for fr in aux_frames:
        populate_rigid_link_constraint(fr, A_soc, d_i, x_dim, num_boxes)

    # write this as a cost term, instead? otherwise we just have inequality
    soc_constraint = []
    cost_log_abs_list = []
    for Ai in A_soc:
        soc_constraint.append(cp.SOC(d_i, Ai @ x))
        cost_log_abs_list.append(w_i * cp.log_det(cp.diag(Ai @ x)))

    cost_log_abs = -(cp.sum(cost_log_abs_list))

    return cost_log_abs, soc_constraint, A_soc


def create_bezier_cvx_norm_eq_relaxation(link_length, x_var_first_point, x_var_second_point,
                                         soc_constraint, cost_log_abs_list, wi=None):
    if wi is None:
        wi = [1., 1., 1.]

    d_i = cp.Parameter(pos=True, value=link_length)

    # construct inequality constraints and log cost term to approximate norm equality
    # soc_constraint.append(cp.SOC(d_i, x_var_first_point - x_var_second_point))
    soc_constraint.append(cp.norm(x_var_first_point - x_var_second_point, 2) <= d_i)

    # add log det cost to encourage/apply relaxed rigid link constraint
    cost_log_abs_list.append(cp.log_det(cp.diag(wi @ (x_var_first_point - x_var_second_point))))

    return


def populate_rigid_link_constraint(aux_frame, A_soc, d_soc, x_dim, num_boxes):
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

    # get indices for parent-child pair in state vector TODO pass parent/child link names
    if aux_frame['parent_frame'] == 'l_knee_fe_ld' and aux_frame['child_frame'] == 'l_foot_contact':
        foot_idx = np.array([3, 4, 5])
        knee_idx = np.array([9, 10, 11])
    elif aux_frame['parent_frame'] == 'r_knee_fe_ld' and aux_frame['child_frame'] == 'r_foot_contact':
        foot_idx = np.array([6, 7, 8])
        knee_idx = np.array([12, 13, 14])
    else:
        print(f"Error: length from {aux_frame['parent_frame']} to {aux_frame['child_frame']} does not exist")
    d_soc.value = aux_frame['length']

    # create constraint for all interior curve points
    for p in range(1, num_boxes):
        A_j = np.zeros((3, x_dim * (num_boxes + 1)))  # (x,y,z) of current curve point
        start_idx_ft = x_dim * p + foot_idx
        start_idx_kn = x_dim * p + knee_idx
        A_j[:, start_idx_ft] = np.eye(3)
        A_j[:, start_idx_kn] = -np.eye(3)
        A_soc.append(A_j)