from casadi import *
import numpy as np
import cvxpy as cp


"""
Superclasses to derive separate collision primitives from
"""
class PrimitiveHessFun(Callback):
    def __init__(self, name, opts={}):
        Callback.__init__(self)
        self.construct(name, opts)

    def get_n_in(self):
        return 3

    def get_n_out(self):
        return 2

    def get_sparsity_in(self, i):
        if i == 0:      # nominal input, x
            return Sparsity.dense(6, 1)
        elif i == 1:    # nominal output, f(x)
            return Sparsity.dense(1, 1)
        elif i == 2:    # nominal jac, J(x)
            return Sparsity.dense(1, 6)

    def get_sparsity_out(self, i):
        if i == 0:      # hessian
            return Sparsity.dense(6, 6)
        elif i == 1:    # jacobian
            return Sparsity.dense(6, 1)

    def eval(self, arg):
        x = np.array(arg[0])
        alpha = np.array(arg[1])
        jac = np.array(arg[2])
        return DM(6, 6), DM(6,1)


class PrimitiveJacFun(Callback):
    def __init__(self, name, geom_data, z, opts={}):
        self.A1 = geom_data['A1']
        self.b1 = geom_data['b1']
        self.Q = geom_data['Q']
        self.z = z
        self.hess_callback = None

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        if i == 0:  # nominal input
            return Sparsity.dense(6, 1)
        elif i == 1:  # nominal output
            return Sparsity(1, 1)

    def get_sparsity_out(self, i):
        if i == 0:
            return Sparsity.dense(1, 6)

    def update_dual_vars(self, z):
        self.z = z

    def has_jacobian(self, *_args):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        # It is required to keep a reference alive to the returned Callback object
        return self.hess_callback


class SinglePrimitivesDistanceCallback(Callback):
    def __init__(self, name, geom_data, opts={}):
        self.A1 = geom_data['A1']      # torso halfspace
        self.b1 = geom_data['b1']      # torso halfspace offset
        self.Q = geom_data['Q']        # torso rotation

        # torso cone representation matrices
        self.G1 = np.zeros((6,4))
        self.G1[:, :3] = self.A1 @ self.Q.T
        self.G1[:, 3:] = -self.b1

        # initialize object construction
        self.jac_callback = None

    def init(self):
        print('Initializing SinglePrimitivesDistanceCallback')

    def get_n_in(self):
        """
        Number of input arguments: r1, r2.
        These correspond to the center of the torso (r1) and the end effector (r2).
        """
        return 1

    def get_sparsity_in(self, i):
        """
        Both input arguments r1 and r2 are in 3-D coordinates.
        """
        return Sparsity.dense(6, 1)

    def get_sparsity_out(self, i):
        return Sparsity.dense(1, 1)

    def has_jacobian(self, *_args):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        # It is required to keep a reference alive to the returned Callback object
        return self.jac_callback


class IndexedHessFun(Callback):
    def __init__(self, name, dim_optim_var, pos1_curr_ir, pos2_curr_ir, opts):
        Callback.__init__(self)
        self.dim_optim_var = dim_optim_var

        self.jac_dm = np.zeros((dim_optim_var, 1), dtype=bool)
        self.hess_dm = np.zeros((dim_optim_var, dim_optim_var), dtype=bool)

        # Sparse entries in Jacobian
        self.jac_dm[pos1_curr_ir:pos1_curr_ir + 3, 0] = 1
        self.jac_dm[pos2_curr_ir:pos2_curr_ir + 3, 0] = 1
        self.jac_dm_in = self.jac_dm.reshape(1, -1)

        self.jac_np = np.zeros((self.dim_optim_var, 1))
        self.hess_np = np.zeros((self.dim_optim_var, self.dim_optim_var))

        self.construct(name, opts)

    def get_n_in(self):
        return 3

    def get_n_out(self):
        return 2

    def get_sparsity_in(self, i):
        if i == 0:      # nominal input, x
            return Sparsity.dense(self.dim_optim_var, 1)
        elif i == 1:    # nominal output, f(x)
            return Sparsity.dense(1, 1)
        elif i == 2:    # nominal jac, J(x)
            return sparsify(DM(self.jac_dm_in.tolist())).sparsity()
            # return Sparsity.dense(1, self.dim_optim_var)

    def get_sparsity_out(self, i):
        if i == 0:      # hessian
            return sparsify(DM(self.hess_dm.tolist())).sparsity()
            # return Sparsity.dense(self.dim_optim_var, self.dim_optim_var)
        elif i == 1:    # jacobian
            return sparsify(DM(self.jac_dm.tolist())).sparsity()
            # return Sparsity.dense(self.dim_optim_var, 1)

    def eval(self, arg):
        x = np.array(arg[0])
        alpha = np.array(arg[1])
        jac = np.array(arg[2])

        return self.hess_np, jac


class IndexedPrimitiveGeometryJacFun(Callback):
    def __init__(self, name, geometry_data, mfpp_bezier_data, z, opts={}):
        self.A1 = geometry_data['A1']
        self.b1 = geometry_data['b1']
        self.Q = geometry_data['Q']
        self.z = z

        self.current_frames = mfpp_bezier_data['current_frames']
        self.current_point = mfpp_bezier_data['current_point']
        self.n_frames = mfpp_bezier_data['num_frames']
        self.D = mfpp_bezier_data['num_derivatives']
        self.num_iris_regions_per_frame = mfpp_bezier_data['num_iris_per_frame']
        self.n_points = (self.D + 1) * 2
        self.bezier_higher_derivatives = 3 * ((self.D + 1) * self.n_points - (self.D + 3))
        self.state_dim_per_frame = self.bezier_higher_derivatives * self.num_iris_regions_per_frame
        self.dim_optim_var = self.state_dim_per_frame * self.n_frames

        current_frames = mfpp_bezier_data['current_frames']
        num_iris_regions = mfpp_bezier_data['num_iris_per_frame']
        # ------- sparse entries in Jacobian
        self.jac_dm = np.zeros((1, self.dim_optim_var), dtype=bool)
        # get current point and solve min distance for each pair of points
        curr_pnt_in_iris = self.current_point % self.n_points
        curr_ir = self.current_point // self.n_points
        pos1_curr_ir = (current_frames[0] * self.bezier_higher_derivatives * num_iris_regions +
                        self.bezier_higher_derivatives * curr_ir +
                        curr_pnt_in_iris)
        pos2_curr_ir = (current_frames[1] * self.bezier_higher_derivatives * num_iris_regions +
                        self.bezier_higher_derivatives * curr_ir +
                        curr_pnt_in_iris)
        pos1_idxs = np.arange(pos1_curr_ir, pos1_curr_ir + 3*self.n_points, step=self.n_points, dtype=int)
        pos2_idxs = np.arange(pos2_curr_ir, pos2_curr_ir + 3*self.n_points, step=self.n_points, dtype=int)
        self.jac_dm[0, pos1_idxs] = 1
        self.jac_dm[0, pos2_idxs] = 1

        self.hess_callback = IndexedHessFun(name, self.dim_optim_var, pos1_curr_ir, pos2_curr_ir, opts)

        self.pos1_idxs = pos1_idxs
        self.pos2_idxs = pos2_idxs

    def get_n_in(self):
        return 2

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        if i == 0:  # nominal input
            return Sparsity.dense(self.dim_optim_var, 1)
        elif i == 1:  # nominal output
            return Sparsity(1, 1)

    def get_sparsity_out(self, i):
        if i == 0:
            return sparsify(DM(self.jac_dm.tolist())).sparsity()
            # return Sparsity.dense(1, self.dim_optim_var)

    def update_dual_vars(self, z):
        self.z = z

    def has_jacobian(self, *_args):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        # It is required to keep a reference alive to the returned Callback object
        return self.hess_callback


class IndexedPrimitiveGeometryDistanceCallback(Callback):
    def __init__(self, name, geom_data, mfpp_bezier_data, opts={}):
        self.A1 = geom_data['A1']      # torso halfspace
        self.b1 = geom_data['b1']      # torso halfspace offset
        self.Q = geom_data['Q']      # torso rotation

        # dimensions of robot geometry
        self.num_halfplanes_A1 = self.A1.shape[0]
        # self.num_halfplanes_A2 = self.A2.shape[0]

        # dimensions to characterize the optimization variable
        self.current_frames = mfpp_bezier_data['current_frames']
        self.current_point = mfpp_bezier_data['current_point']
        self.D = mfpp_bezier_data['num_derivatives']
        self.num_iris_regions_per_frame = mfpp_bezier_data['num_iris_per_frame']
        self.n_frames = mfpp_bezier_data['num_frames']
        self.n_points = (self.D + 1) * 2
        self.bezier_higher_derivatives = 3 * ((self.D + 1) * self.n_points - (self.D + 3))
        self.state_dim_per_frame = self.bezier_higher_derivatives * self.num_iris_regions_per_frame
        self.dim_optim_var = self.state_dim_per_frame * self.n_frames

        # torso cone representation matrices
        # self.G1 = np.zeros((self.num_halfplanes_A1, 4))
        # self.G1[:, :3] = self.A1 @ self.Q.T
        # self.G1[:, 3:] = -self.b1
        # self.G2 = np.zeros((self.num_halfplanes_A2, 4))
        # self.G2[:, :3] = self.A2 @ self.Q.T
        # self.G2[:, 3:] = -self.b2

        # self.z = np.zeros((self.num_halfplanes_A1 + self.num_halfplanes_A2 + 1, 1))     # DCOL dual variable

        # initialize object construction
        self.jac_callback = None

    def get_n_in(self):
        """
        Number of input arguments: r1, r2.
        These correspond to the center of the torso (r1) and the end effector (r2).
        """
        return 1

    def get_sparsity_in(self, i):
        """
        Both input arguments r1 and r2 are in 3-D coordinates.
        """
        return Sparsity.dense(self.dim_optim_var, 1)

    def get_sparsity_out(self, i):
        """
        Alpha value corresponding to scaling of polytopes to reach collision
        """
        return Sparsity.dense(1, 1)

    def has_jacobian(self, *_args):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        # It is required to keep a reference alive to the returned Callback object
        # self.jac_callback = IndexedJacFun(name, self.geom_data, self.mfpp_bezier_data, self.z)
        return self.jac_callback



"""
Derived classes
"""
class SinglePolytopePolytopeJacFun(PrimitiveJacFun):
    def __init__(self, name, geom_data, z, opts={}):
        Callback.__init__(self)
        super().__init__(name, geom_data, z, opts)
        self.A2 = geom_data['A2']
        self.b2 = geom_data['b2']
        self.z = z
        self.hess_callback = PrimitiveHessFun(name, opts)
        self.construct(name, opts)

    def eval(self, arg):
        p = np.array(arg[0])
        r1 = p[:3]
        r2 = p[3:]

        ret1 = self.z[:-1].T @ (np.concatenate((-self.A1 @ self.Q.T, np.zeros((6, 3)))))
        ret2 = self.z[:-1].T @ (np.concatenate((np.zeros((6, 3)), -self.A2 @ self.Q.T)))
        grad_g = np.hstack((ret1, ret2))

        return [grad_g]


class SinglePolytopePolytopeDistanceCallback(SinglePrimitivesDistanceCallback):
    def __init__(self, name, geom_data, opts={}):
        Callback.__init__(self)
        super().__init__(name, geom_data, opts)

        # pending properties
        self.A2 = geom_data['A2']
        self.b2 = geom_data['b2']
        self.z = np.zeros((13,1))
        self.jac_callback = SinglePolytopePolytopeJacFun(name, geom_data, self.z, opts)

        # initialize object construction
        self.construct(name, opts)

    def eval(self, arg):
        p = np.array(arg[0])
        r1 = p[:3]   # torso center
        r2 = p[3:]   # end effector (sphere) center

        A1 = self.A1
        b1 = self.b1
        A2 = self.A2
        b2 = self.b2
        Q = self.Q
        x_val, alpha_val, dual_val = solve_two_polytope_min_prox(A1, b1, A2, b2, Q, r1, r2, False)

        self.jac_callback.update_dual_vars(dual_val)

        return [alpha_val]


class SinglePolytopeEllipsoidJacFun(PrimitiveJacFun):
    def __init__(self, name, geom_data, z, opts={}):
        Callback.__init__(self)
        super().__init__(name, geom_data, z)
        self.U = geom_data['U']
        self.z = z
        self.hess_callback = PrimitiveHessFun(name, opts)
        self.construct(name, opts)

    def get_jacobian(self, name, inames, onames, opts):
        # It is required to keep a reference alive to the returned Callback object
        return self.hess_callback

    # Evaluate numerically
    def eval(self, arg):
        p = np.array(arg[0])
        r1 = p[:3]
        r2 = p[3:]

        grad_polytope = -self.A1 @ self.Q.T
        grad_ellipse = np.vstack((np.zeros((1,3)), self.U @ self.Q.T))
        ret1 = self.z[:-1].T @ (np.concatenate((grad_polytope, np.zeros((4, 3)))))
        ret2 = self.z[:-1].T @ (np.concatenate((np.zeros((6, 3)), grad_ellipse)))
        grad_g = np.hstack((ret1, ret2))

        return [grad_g]

class SinglePolytopeEllipsoidDistanceCallback(SinglePrimitivesDistanceCallback):
    def __init__(self, name, geom_data, opts={}):
        Callback.__init__(self)
        super().__init__(name, geom_data, opts)

        # pending properties
        self.U = geom_data['U']
        self.G2 = np.zeros((4,4))
        self.G2[1:, :3] = -self.U @ self.Q.T
        self.G2[0, -1] = -1
        self.h2 = np.zeros((4,1))    # needs to be updated in eval
        self.z = np.zeros((11,1))
        self.jac_callback = SinglePolytopeEllipsoidJacFun(name, geom_data, self.z, opts)

        # initialize object construction
        self.construct(name, opts)

    def eval(self, arg):
        p = np.array(arg[0])
        r1 = p[:3]   # torso center
        r2 = p[3:]   # end effector (sphere) center

        A1 = self.A1
        b1 = self.b1
        U = self.U
        Q = self.Q
        x_val, alpha_val, dual_val = solve_polytope_ellipsoid_min_prox(A1, b1, Q, U, r1, r2)

        self.jac_callback.update_dual_vars(dual_val)

        return [alpha_val]


class IndexedPolytopeEllipsoidJacFun(IndexedPrimitiveGeometryJacFun):
    def __init__(self, name, geometry_data, mfpp_bezier_data, z, opts={}):
        Callback.__init__(self)
        super().__init__(name, geometry_data, mfpp_bezier_data, z, opts)
        self.U = geometry_data['U']
        self.construct(name, opts)

    # Evaluate numerically
    def eval(self, arg):
        z = np.array(arg[0])

        # reconstruct cone matrices with implicit parameters
        grad_polytope = -self.A1 @ self.Q.T
        grad_ellipse = np.vstack((np.zeros((1,3)), self.U @ self.Q.T))
        ret1 = self.z[:-1].T @ (np.concatenate((grad_polytope, np.zeros((4, 3)))))
        ret2 = self.z[:-1].T @ (np.concatenate((np.zeros((6, 3)), grad_ellipse)))

        # ---- distribute to corresponding indices in Jacobian
        # aesthetics
        pos1_idxs = self.pos1_idxs
        pos2_idxs = self.pos2_idxs

        jac_z = np.zeros((1, self.dim_optim_var))
        jac_z[0, pos1_idxs] = ret1
        jac_z[0, pos2_idxs] = ret2

        return [jac_z]


class IndexedPolytopeEllipsoidConstraint(IndexedPrimitiveGeometryDistanceCallback):
    def __init__(self, name, geom_data, mfpp_bezier_data, opts={}):
        Callback.__init__(self)
        super().__init__(name, geom_data, mfpp_bezier_data, opts)
        self.U = geom_data['U']      # EE ellipsoid

        # dimensions of robot geometry
        self.num_halfplanes_U = self.U.shape[0]

        self.z = np.zeros((self.num_halfplanes_A1 + self.num_halfplanes_U + 1, 1))     # DCOL dual variable

        # initialize object construction
        self.jac_callback = IndexedPolytopeEllipsoidJacFun(name, geom_data, mfpp_bezier_data, self.z)
        self.construct(name, opts)

    def eval(self, arg):
        z = np.array(arg[0])

        # aesthetics
        A1 = self.A1
        b1 = self.b1
        U = self.U
        Q = self.Q
        num_iris_regions = self.num_iris_regions_per_frame
        current_frames = self.current_frames
        current_point = self.current_point
        bezier_higher_derivatives = self.bezier_higher_derivatives

        # get current point and solve min distance for each pair of points
        curr_pnt_in_iris = current_point % self.n_points
        curr_ir = current_point // self.n_points
        pos1_curr_ir = (current_frames[0] * bezier_higher_derivatives * num_iris_regions +
                        bezier_higher_derivatives * curr_ir +
                        curr_pnt_in_iris)
        pos2_curr_ir = (current_frames[1] * bezier_higher_derivatives * num_iris_regions +
                        bezier_higher_derivatives * curr_ir +
                        curr_pnt_in_iris)
        pos1_idxs = np.arange(pos1_curr_ir, pos1_curr_ir + 3*self.n_points, step=self.n_points, dtype=int)
        pos2_idxs = np.arange(pos2_curr_ir, pos2_curr_ir + 3*self.n_points, step=self.n_points, dtype=int)
        r1_cp = z[pos1_idxs]
        r2_cp = z[pos2_idxs]
        x_val, alpha_val, dual_val = solve_polytope_ellipsoid_min_prox(A1, b1, Q, U, r1_cp, r2_cp)

        # update dual variables to use in Jacobian
        self.jac_callback.update_dual_vars(dual_val)

        return [alpha_val]


class IndexedPolytopePolytopeJacFun(IndexedPrimitiveGeometryJacFun):
    def __init__(self, name, geometry_data, mfpp_bezier_data, z, opts={}):
        Callback.__init__(self)
        super().__init__(name, geometry_data, mfpp_bezier_data, z, opts)
        self.A2 = geometry_data['A2']
        self.b2 = geometry_data['b2']
        self.construct(name, opts)

    # Evaluate numerically
    def eval(self, arg):
        z = np.array(arg[0])

        # reconstruct cone matrices with implicit parameters
        grad_torso_polytope = -self.A1 @ self.Q.T
        grad_ee_polytope = -self.A2 @ self.Q.T
        ret1 = self.z[:-1].T @ (np.concatenate((grad_torso_polytope, np.zeros((6, 3)))))
        ret2 = self.z[:-1].T @ (np.concatenate((np.zeros((6, 3)), grad_ee_polytope)))

        # ---- distribute to corresponding indices in Jacobian
        # aesthetics
        pos1_idxs = self.pos1_idxs
        pos2_idxs = self.pos2_idxs

        jac_z = np.zeros((1, self.dim_optim_var))
        jac_z[0, pos1_idxs] = ret1
        jac_z[0, pos2_idxs] = ret2

        return [jac_z]

class IndexedPolytopePolytopeConstraint(IndexedPrimitiveGeometryDistanceCallback):
    def __init__(self, name, geom_data, mfpp_bezier_data, opts={}):
        Callback.__init__(self)
        super().__init__(name, geom_data, mfpp_bezier_data, opts)
        self.A2 = geom_data['A2']      # EE halfspace
        self.b2 = geom_data['b2']      # EE halfspace offset

        # dimensions of robot geometry
        self.num_halfplanes_A2 = self.A2.shape[0]

        self.z = np.zeros((self.num_halfplanes_A1 + self.num_halfplanes_A2 + 1, 1))     # DCOL dual variable

        # initialize object construction
        self.jac_callback = IndexedPolytopePolytopeJacFun(name, geom_data, mfpp_bezier_data, self.z)
        self.construct(name, opts)

    def eval(self, arg):
        z = np.array(arg[0])

        # aesthetics
        A1 = self.A1
        b1 = self.b1
        A2 = self.A2
        b2 = self.b2
        Q = self.Q
        num_iris_regions = self.num_iris_regions_per_frame
        current_frames = self.current_frames
        current_point = self.current_point
        bezier_higher_derivatives = self.bezier_higher_derivatives

        # get current point and solve min distance for each pair of points
        curr_pnt_in_iris = current_point % self.n_points
        curr_ir = current_point // self.n_points
        pos1_curr_ir = (current_frames[0] * bezier_higher_derivatives * num_iris_regions +
                        bezier_higher_derivatives * curr_ir +
                        curr_pnt_in_iris)
        pos2_curr_ir = (current_frames[1] * bezier_higher_derivatives * num_iris_regions +
                        bezier_higher_derivatives * curr_ir +
                        curr_pnt_in_iris)
        pos1_idxs = np.arange(pos1_curr_ir, pos1_curr_ir + 3*self.n_points, step=self.n_points, dtype=int)
        pos2_idxs = np.arange(pos2_curr_ir, pos2_curr_ir + 3*self.n_points, step=self.n_points, dtype=int)
        r1_cp = z[pos1_idxs]
        r2_cp = z[pos2_idxs]
        x_val, alpha_val, dual_val = solve_two_polytope_min_prox(A1, b1, A2, b2, Q, r1_cp, r2_cp)

        # update dual variables to use in Jacobian
        self.jac_callback.update_dual_vars(dual_val)

        return [alpha_val]


"""
Helper functions solving differentiable collisions for polytope-polytope
and polytope-ellipsoid pairs 
"""
def solve_two_polytope_min_prox(A1, b1, A2, b2, Q, r1, r2, b_print_info=False):
    # find distance via optimization (dcol-style)
    alpha = cp.Variable(1)
    x = cp.Variable((3, 1))

    constraints = [
        A1 @ Q.T @ (x - r1) <= alpha * b1,
        A2 @ Q.T @ (x - r2) <= alpha * b2,
        alpha >= 0
    ]
    prob = cp.Problem(cp.Minimize(alpha), constraints)
    prob.solve(solver='CLARABEL')

    # z = np.concatenate(list((prob.solution.dual_vars).values()))
    z_lst = list((prob.solution.dual_vars).values())
    z_lst[-1] = np.reshape(z_lst[-1], (1, 1))
    z_lst = np.concatenate(z_lst)

    solver_time = prob.solver_stats.solve_time

    # print solution information
    if b_print_info:
        min_distance = np.linalg.norm(r1 - r2 + (r2 - r1) / alpha.value)
        cp_torso = r1 + (x.value - r1) / alpha.value
        cp_foot = r2 + (x.value - r2) / alpha.value
        print(f"alpha: {alpha.value}")
        print(f"contact in torso: {cp_torso.T}, contact in foot: {cp_foot.T}")
        print(f"min distance: {min_distance}")

    return x.value, alpha.value, z_lst



def solve_polytope_ellipsoid_min_prox(A, b, Q, U, r1, r2, verbose=False):
    # find distance via optimization (dcol-style)
    alpha = cp.Variable(1)
    x = cp.Variable((3, 1))

    constraints = []
    constraints.append(A @ Q.T @ (x - r1) <= alpha * b)
    constraints.append(cp.SOC(alpha, U @ Q.T @ (x - r2)))
    constraints.append(alpha >= 0)
    prob = cp.Problem(cp.Minimize(alpha), constraints)
    prob.solve(solver='CLARABEL')

    dual_v = list((prob.solution.dual_vars).values())
    dual_v[-1] = np.reshape(dual_v[-1], (-1, 1))
    dual_v[-2] = np.reshape(dual_v[-2], (-1, 1))
    dual_v = np.concatenate(dual_v)
    solver_time = prob.solver_stats.solve_time

    return x.value, alpha.value, dual_v
