from casadi import *
import numpy as np
import cvxpy as cp

class DColMinDistancePairsCallback(Callback):
    def __init__(self, name, geom_data, opts={}):
        Callback.__init__(self)
        self.A = geom_data['A']      # torso halfspace
        self.b = geom_data['b']      # torso halfspace offset
        self.Q = geom_data['Q']      # torso rotation
        self.U = geom_data['U']      # torso center

        # torso cone representation matrices
        self.G1 = np.zeros((6,4))
        self.G1[:, :3] = self.A @ self.Q.T
        self.G1[:, 3:] = -self.b
        self.h1 = np.zeros((6,6))    # needs to be updated in eval
        self.z = np.zeros((10,1))     # DCOL dual variable

        # end effector sphere representation matrices
        self.G2 = np.zeros((4,4))
        self.G2[1:, :3] = -self.U @ self.Q.T
        self.G2[0, -1] = -1
        self.h2 = np.zeros((4,1))    # needs to be updated in eval

        # initialize object construction
        self.construct(name, opts)

    def init(self):
        print('Initializing DColMinDistancePairsCallback')

    def get_n_in(self):
        """
        Number of input arguments: r1, r2.
        These correspond to the center of the torso (r1) and the end effector (r2).
        """
        return 2

    def get_sparsity_in(self, i):
        """
        Sparsity pattern of input arguments: 3-D coordinates.
        """
        return Sparsity.dense(3, 1)

    def get_sparsity_out(self, i):
        return Sparsity.dense(1, 1)

    def has_jacobian(self, *_args):
        return True

    def get_jacobian(self, *args):
        # FIXME: need to finish implementing and test this method
        G1 = self.G1
        G2 = self.G2

        # provide first order Jacobians
        r1 = MX.sym('r1', 3, 1)
        r2 = MX.sym('r2', 3, 1)
        x = MX.sym('x', 3, 1)
        alpha = MX.sym('alpha', 1)
        xsol = vertcat(x, alpha)

        # reconstruct cone matrices with implicit parameters
        h1 = self.A @ self.Q.T @ r1
        h2 = vertcat(0, -self.U @ self.Q.T @ r2)
        h = vertcat(h1, h2)
        G = vertcat(G1, G2)

        # optimality condition
        g_impl = G @ xsol - h
        grad_g = jacobian(g_impl, r1)

        # z = self.z
        # f = MX(1, 1)
        # grad_r1 = jacobian(solve_polytope_min_prox, r1)
        # grad_r2 = jacobian(solve_polytope_min_prox, r2)
        # Df =  Function('Df', [x, f], [z.T @ (G1 @ x - h1)])

        return grad_g


    def eval(self, arg):
        r1 = np.array(arg[0])   # torso center
        r2 = np.array(arg[1])   # end effector (sphere) center

        # find distance via optimization (dcol)
        alpha = cp.Variable(1)
        x = cp.Variable((3,1))

        A = self.A
        b = self.b
        Q = self.Q
        U = self.U
        constraints = []
        constraints.append(A @ Q.T @ (x - r1) <= alpha * b)
        constraints.append(cp.norm(U @ Q.T @ (x - r2)) <= alpha)
        constraints.append(alpha >= 0)
        prob = cp.Problem(cp.Minimize(alpha), constraints)
        prob.solve(solver='CLARABEL')

        alpha_val = alpha.value
        x_val = x.value
        solver_time = prob.solver_stats.solve_time

        min_distance = np.linalg.norm(r1 - r2 + (r2 - r1) / alpha_val)
        cp_torso = r1 + (x_val - r1) / alpha_val
        cp_sphere = r2 + (x_val - r2) / alpha_val
        print(f"alpha: {alpha_val}")
        print(f"contact in torso: {cp_torso.T}, contact in sphere: {cp_sphere.T}")
        print(f"min distance: {min_distance}")

        self.h1 = self.A @ self.Q.T @ r1
        self.h2[1:] = -self.U @ self.Q.T @ r2
        dual_v = list((prob.solution.dual_vars).values())
        dual_v[-1] = np.reshape(dual_v[-1], (1, 1))
        dual_v[-2] = np.reshape(dual_v[-2], (1, 1))
        self.z = np.concatenate(dual_v)

        return [min_distance]


class SingleHessFun(Callback):
    def __init__(self, name, opts):
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

class SingleJacFun(Callback):
    def __init__(self, name, A1, b1, A2, b2, Q, z, opts={}):
        Callback.__init__(self)
        self.A1 = A1
        self.b1 = b1
        self.A2 = A2
        self.b2 = b2
        self.Q = Q
        self.z = z
        self.hess_callback = SingleHessFun(name, opts)
        self.construct(name, opts)

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

    # Evaluate numerically
    def eval(self, arg):
        p = np.array(arg[0])
        r1 = p[:3]
        r2 = p[3:]

        # reconstruct cone matrices with implicit parameters
        # h1 = self.A1 @ self.Q.T @ r1
        # h2 = self.A2 @ self.Q.T @ r2
        # h = np.concatenate((h1, h2))
        # G = np.concatenate((self.G1, self.G2))

        ret1 = self.z[:-1].T @ (np.concatenate((-self.A1 @ self.Q.T, np.zeros((6, 3)))))
        ret2 = self.z[:-1].T @ (np.concatenate((np.zeros((6, 3)), -self.A2 @ self.Q.T)))

        return [np.hstack((ret1, ret2))]

class DColMinSinglePolytopesDistanceCallback(Callback):
    def __init__(self, name, geom_data, opts={}):
        Callback.__init__(self)
        self.A1 = geom_data['A1']      # torso halfspace
        self.b1 = geom_data['b1']      # torso halfspace offset
        self.A2 = geom_data['A2']      # torso halfspace
        self.b2 = geom_data['b2']      # torso halfspace offset
        self.Q = geom_data['Q']      # torso rotation

        # torso cone representation matrices
        self.G1 = np.zeros((6,4))
        self.G1[:, :3] = self.A1 @ self.Q.T
        self.G1[:, 3:] = -self.b1
        self.G2 = np.zeros((6,4))
        self.G2[:, :3] = self.A2 @ self.Q.T
        self.G2[:, 3:] = -self.b2

        self.z = np.zeros((13,1))     # DCOL dual variable
        self.xsol = np.zeros((4,1))

        # initialize object construction
        self.jac_callback = SingleJacFun(name, self.A1, self.b1, self.A2, self.b2, self.Q, self.z)
        self.construct(name, opts)

    def init(self):
        print('Initializing DColMinSinglePolytopesDistanceCallback')

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

class JacFun(Callback):
    def __init__(self, name, A1, b1, A2, b2, Q, num_iris_regions, z, opts={}):
        Callback.__init__(self)
        self.A1 = A1
        self.b1 = b1
        self.A2 = A2
        self.b2 = b2
        self.Q = Q
        self.z = z

        # dimensions of robot geometry
        self.num_halfplanes_A1 = self.A1.shape[0]
        self.num_halfplanes_A2 = self.A2.shape[0]

        # dimensions to characterize the optimization variable
        space_dim = self.A1.shape[1]
        self.D = space_dim
        self.n_points = (space_dim + 1) * 2
        self.n_frames = 2
        self.state_dim_per_frame = 3 * ((self.D + 1) * self.n_points - (self.D + 3)) * num_iris_regions
        self.num_iris_regions_per_frame = num_iris_regions

        # sparse Jacobian entries
        bezier_higher_derivatives = 3 * ((self.D + 1) * self.n_points - (self.D + 3))
        self.p1_start_idx = np.arange(0, self.state_dim_per_frame, step=bezier_higher_derivatives, dtype=int)
        self.p2_start_idx = np.arange(self.state_dim_per_frame, self.n_frames*self.state_dim_per_frame + 3*self.n_points, step=bezier_higher_derivatives, dtype=int)
        self.jac_dm = np.zeros((self.n_points * self.num_iris_regions_per_frame, self.n_frames * self.state_dim_per_frame), dtype=bool)
        n_points = self.n_points
        p1_start_idx = self.p1_start_idx
        p2_start_idx = self.p2_start_idx
        for ir in range(self.num_iris_regions_per_frame):
            last_ir1_pos_idx = p1_start_idx[ir] + self.n_points * 3
            last_ir2_pos_idx = p2_start_idx[ir] + self.n_points * 3
            pos1_curr_ir = np.arange(p1_start_idx[ir], last_ir1_pos_idx, step=3, dtype=int)
            pos2_curr_ir = np.arange(p2_start_idx[ir], last_ir2_pos_idx, step=3, dtype=int)
            assert (len(pos1_curr_ir) == n_points)
            assert (len(pos2_curr_ir) == n_points)
            for pnt in range(self.n_points):
                self.jac_dm[ir * n_points + pnt, pos1_curr_ir[pnt]: pos1_curr_ir[pnt] + 3] = 1
                self.jac_dm[ir * n_points + pnt, pos2_curr_ir[pnt]: pos2_curr_ir[pnt] + 3] = 1

        self.construct(name, opts)

    def get_n_in(self):
        """
        Jacobian input arguments are 2: x, f(x)
        """
        return 2

    def get_n_out(self):
        """
        Jacobian returns 1 output: gradient of the function.
        """
        return 1

    def get_sparsity_in(self, i):
        if i == 0:  # nominal input
            return Sparsity.dense(self.n_frames * self.state_dim_per_frame, 1)
        elif i == 1:  # nominal output
            return Sparsity.dense(self.n_points * self.num_iris_regions_per_frame, 1)

    def get_sparsity_out(self, i):
        # build selection matrix of non-zero Jacobian elements
        if i == 0:
            return sparsify(DM(self.jac_dm.tolist())).sparsity()
            # return Sparsity.dense(self.n_points * self.num_iris_regions_per_frame, self.n_frames * self.state_dim_per_frame)

    def update_dual_vars(self, z):
        self.z = z

    # Evaluate numerically
    def eval(self, arg):
        x = np.array(arg[0])
        n_hp_A1 = self.num_halfplanes_A1
        n_hp_A2 = self.num_halfplanes_A2

        # extract position indices
        p1_start_idx = self.p1_start_idx
        p2_start_idx = self.p2_start_idx
        n_hp_A1 = self.num_halfplanes_A1
        n_hp_A2 = self.num_halfplanes_A2
        num_constr_per_point = n_hp_A1 + n_hp_A2 + 1
        # jac_alpha_x = np.zeros((self.num_iris_regions_per_frame*self.n_points, self.n_frames * self.state_dim_per_frame))
        jac_alpha_x = DM(self.num_iris_regions_per_frame*self.n_points, self.n_frames * self.state_dim_per_frame)
        for ir in range(self.num_iris_regions_per_frame):
            last_ir1_pos_idx = p1_start_idx[ir] + self.n_points * 3
            last_ir2_pos_idx = p2_start_idx[ir] + self.n_points * 3
            pos1_curr_ir = np.arange(p1_start_idx[ir], last_ir1_pos_idx, step=3, dtype=int)
            pos2_curr_ir = np.arange(p2_start_idx[ir], last_ir2_pos_idx, step=3, dtype=int)
            for pnt in range(self.n_points):
                # reconstruct cone matrices with implicit parameters
                # h1 = self.A1 @ self.Q.T @ r1
                # h2 = self.A2 @ self.Q.T @ r2
                # h = np.concatenate((h1, h2))
                # G = np.concatenate((self.G1, self.G2))
                current_idx = pnt * num_constr_per_point
                next_idx = current_idx + num_constr_per_point
                curr_overall_idx = ir*self.n_points + pnt

                ret1 = self.z[:-1, curr_overall_idx].T @ (np.concatenate((-self.A1 @ self.Q.T, np.zeros((n_hp_A2, 3)))))
                ret2 = self.z[:-1, curr_overall_idx].T @ (np.concatenate((np.zeros((n_hp_A1, 3)), -self.A2 @ self.Q.T)))

                jac_alpha_x[curr_overall_idx, pos1_curr_ir[pnt]: pos1_curr_ir[pnt] + 3] = ret1
                jac_alpha_x[curr_overall_idx, pos2_curr_ir[pnt]: pos2_curr_ir[pnt] + 3] = ret2

        return [jac_alpha_x]

class DColMinPolytopesDistanceCallback(Callback):
    def __init__(self, name, geom_data, space_dim, num_iris_regions, opts={}):
        Callback.__init__(self)
        self.A1 = geom_data['A1']      # torso halfspace
        self.b1 = geom_data['b1']      # torso halfspace offset
        self.A2 = geom_data['A2']      # torso halfspace
        self.b2 = geom_data['b2']      # torso halfspace offset
        self.Q = geom_data['Q']      # torso rotation

        # dimensions of robot geometry
        self.num_halfplanes_A1 = self.A1.shape[0]
        self.num_halfplanes_A2 = self.A2.shape[0]

        # dimensions to characterize the optimization variable
        self.D = space_dim
        self.num_iris_regions_per_frame = num_iris_regions
        self.n_points = (space_dim + 1) * 2
        self.n_frames = 2   # TODO make general later
        self.state_dim_per_frame = 3 * ((self.D + 1) * self.n_points - (self.D + 3)) * num_iris_regions

        # torso cone representation matrices
        self.G1 = np.zeros((self.num_halfplanes_A1, 4))
        self.G1[:, :3] = self.A1 @ self.Q.T
        self.G1[:, 3:] = -self.b1
        self.G2 = np.zeros((self.num_halfplanes_A2, 4))
        self.G2[:, :3] = self.A2 @ self.Q.T
        self.G2[:, 3:] = -self.b2

        # DCOL dual variable
        self.z = np.zeros((self.num_halfplanes_A1 + self.num_halfplanes_A2 + 1, self.n_points * num_iris_regions))

        # initialize object construction
        self.construct(name, opts)

    def init(self):
        print('Initializing DColMinPolytopesDistanceCallback')

    def get_n_in(self):
        """
        Number of input arguments: x = [b1_pos; b1_vel; b1_acc; b1_jerk; ... ; b_np_jerk].
        These correspond to the bezier points (position, velocity, acceleration, jerk) center of
        the torso (r1) and the end effector (r2).
        """
        return 1

    def get_sparsity_in(self, i):
        """
        Input arguments is x \in \mathbb{R}^{n_f * D * n_p}
        """
        return Sparsity.dense(self.n_frames * self.state_dim_per_frame, 1)

    def get_sparsity_out(self, i):
        """
        Output argument is the minimum distance between the two polytopes.
        """
        return Sparsity.dense(self.n_points * self.num_iris_regions_per_frame, 1)

    def has_jacobian(self, *_args):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        # It is required to keep a reference alive to the returned Callback object
        self.jac_callback = JacFun(name, self.A1, self.b1, self.A2, self.b2, self.Q, self.num_iris_regions_per_frame, self.z)
        return self.jac_callback

    def eval(self, arg):
        z = np.array(arg[0])

        # extract position indices
        bezier_higher_derivatives = 3*((self.D + 1) * self.n_points - (self.D + 3))
        p1_start_idx = np.arange(0, self.state_dim_per_frame, step=bezier_higher_derivatives, dtype=int)
        p2_start_idx = np.arange(self.state_dim_per_frame, self.n_frames*self.state_dim_per_frame, step=bezier_higher_derivatives, dtype=int)
        r1 = z[p1_start_idx]      # get only position values for first frame (torso)
        r2 = z[p2_start_idx]      # get only position values for second frame (end-effector)

        A1 = self.A1
        b1 = self.b1
        A2 = self.A2
        b2 = self.b2
        Q = self.Q
        n_points = self.n_points
        n_frames = self.n_frames
        num_iris_regions = self.num_iris_regions_per_frame
        n_hp_A1 = self.num_halfplanes_A1
        n_hp_A2 = self.num_halfplanes_A2
        num_constr_per_point = n_hp_A1 + n_hp_A2 + 1
        dual_all = np.zeros((n_hp_A1 + n_hp_A2 + 1, n_points * num_iris_regions))
        alpha_vec = np.zeros((n_points * num_iris_regions, 1))

        # solve min distance for each pair of points
        for ir in range(num_iris_regions):
            last_ir1_pos_idx = p1_start_idx[ir] + n_points * 3
            last_ir2_pos_idx = p2_start_idx[ir] + n_points * 3
            pos1_curr_ir = np.arange(p1_start_idx[ir], last_ir1_pos_idx, step=3, dtype=int)
            pos2_curr_ir = np.arange(p2_start_idx[ir], last_ir2_pos_idx, step=3, dtype=int)
            assert(len(pos1_curr_ir) == n_points)
            assert(len(pos2_curr_ir) == n_points)
            for pnt in range(self.n_points):
                # get current point
                r1_cp = z[pos1_curr_ir[pnt]:pos1_curr_ir[pnt]+3]
                r2_cp = z[pos2_curr_ir[pnt]:pos2_curr_ir[pnt]+3]
                x_val, alpha_val, dual_val = solve_two_polytope_min_prox(A1, b1, A2, b2, Q, r1_cp, r2_cp)

                # self.xsol = np.concatenate((x_val, alpha_val.reshape((1,1))))
                # self.h1 = self.A1 @ self.Q.T @ r1
                # self.h2 = self.A2 @ self.Q.T @ r2
                alpha_vec[ir*n_points + pnt] = alpha_val
                dual_all[:, ir*n_points + pnt] = dual_val[:, 0]

        self.jac_callback.update_dual_vars(dual_all)

        return [alpha_vec]


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

class IndexedJacFun(Callback):
    def __init__(self, name, geometry_data, mfpp_bezier_data, z, opts={}):
        Callback.__init__(self)
        self.A1 = geometry_data['A1']
        self.b1 = geometry_data['b1']
        self.A2 = geometry_data['A2']
        self.b2 = geometry_data['b2']
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
        self.construct(name, opts)

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

    # Evaluate numerically
    def eval(self, arg):
        z = np.array(arg[0])

        # reconstruct cone matrices with implicit parameters
        # h1 = self.A1 @ self.Q.T @ r1
        # h2 = self.A2 @ self.Q.T @ r2
        # h = np.concatenate((h1, h2))
        # G = np.concatenate((self.G1, self.G2))

        ret1 = self.z[:-1].T @ (np.concatenate((-self.A1 @ self.Q.T, np.zeros((6, 3)))))
        ret2 = self.z[:-1].T @ (np.concatenate((np.zeros((6, 3)), -self.A2 @ self.Q.T)))

        # ---- distribute to corresponding indices in Jacobian
        # aesthetics
        pos1_idxs = self.pos1_idxs
        pos2_idxs = self.pos2_idxs

        jac_z = np.zeros((1, self.dim_optim_var))
        jac_z[0, pos1_idxs] = ret1
        jac_z[0, pos2_idxs] = ret2

        return [jac_z]

class DColIndexedPolytopesConstraint(Callback):
    def __init__(self, name, geom_data, mfpp_bezier_data, opts={}):
        Callback.__init__(self)
        self.A1 = geom_data['A1']      # torso halfspace
        self.b1 = geom_data['b1']      # torso halfspace offset
        self.A2 = geom_data['A2']      # torso halfspace
        self.b2 = geom_data['b2']      # torso halfspace offset
        self.Q = geom_data['Q']      # torso rotation

        # dimensions of robot geometry
        self.num_halfplanes_A1 = self.A1.shape[0]
        self.num_halfplanes_A2 = self.A2.shape[0]

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
        self.G1 = np.zeros((self.num_halfplanes_A1, 4))
        self.G1[:, :3] = self.A1 @ self.Q.T
        self.G1[:, 3:] = -self.b1
        self.G2 = np.zeros((self.num_halfplanes_A2, 4))
        self.G2[:, :3] = self.A2 @ self.Q.T
        self.G2[:, 3:] = -self.b2

        self.z = np.zeros((self.num_halfplanes_A1 + self.num_halfplanes_A2 + 1, 1))     # DCOL dual variable

        # initialize object construction
        self.jac_callback = IndexedJacFun(name, geom_data, mfpp_bezier_data, self.z)
        self.construct(name, opts)

    # def init(self):
        # print(f'Initializing {self.name()}')

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

def solve_polytope_min_prox(A, b, Q, U, r1, r2):
    # find distance via optimization (dcol-style)
    alpha = cp.Variable(1)
    x = cp.Variable((3, 1))

    constraints = []
    constraints.append(A @ Q.T @ (x - r1) <= alpha * b)
    constraints.append(cp.norm(U @ Q.T @ (x - r2)) <= alpha)
    constraints.append(alpha >= 0)
    prob = cp.Problem(cp.Minimize(alpha), constraints)
    prob.solve(solver='CLARABEL')

    z = np.concatenate(list((prob.solution.dual_vars).values()))
    solver_time = prob.solver_stats.solve_time

    return x.value, alpha.value, z
