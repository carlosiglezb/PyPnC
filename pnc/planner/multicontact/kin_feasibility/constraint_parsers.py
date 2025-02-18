import casadi as ca

def parse_mat_leq_constr(A, b, points, constraints, lbg, ubg):
    """
    Constraints are of the form:
    lbg <= constraints(x) <= ubg
    """
    num_ineq = A.shape[0]
    num_points = points[0].shape[0]
    if num_points == 8:
        for k in range(num_ineq):
            for np in range(num_points):
                constraints.append(points[0][np,:] @ A[k])
                lbg.append(-ca.inf)
                ubg.append(b[k])
    #FIXME hot fix for reachability constraints
    elif num_points == 1:
        for k in range(num_ineq):
            for np in range(8):
                constraints.append(points[np,:] @ A[k].reshape(-1, 1))
                lbg.append(-ca.inf)
                ubg.append(b[k])



def parse_repvec_eq_constr(b_vec, points, constraints, lbg, ubg):
    num_eq = points.shape[0]
    for k in range(num_eq):
        for j in range(3):
            constraints.append(points[k,j] - b_vec[0,j])
            lbg.append(0.)
            ubg.append(0.)


def parse_vec_eq_constr(b_vec, point, constraints, lbg, ubg):
    for k in range(3):
        constraints.append(point[k] - b_vec[k])
        lbg.append(0.)
        ubg.append(0.)


def parse_mat_eq_constr(b_mat, points, constraints, lbg, ubg):
    num_eq = points.shape[0]
    for k in range(num_eq):
        for j in range(3):
            constraints.append(points[k,j] - b_mat[k,j])
            lbg.append(0.)
            ubg.append(0.)
