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

    def eval(self, arg):
        r1 = np.array(arg[0]).reshape(3)   # torso center
        r2 = np.array(arg[1]).reshape(3)   # end effector (sphere) center

        # find distance via optimization (dcol)
        alpha = cp.Variable(1)
        x = cp.Variable(3)

        A = self.A
        b = self.b.reshape(-1)
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

class DColMinPolytopesDistanceCallback(Callback):
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
        self.construct(name, opts)

    def init(self):
        print('Initializing DColMinPolytopesDistanceCallback')

    def get_n_in(self):
        """
        Number of input arguments: r1, r2.
        These correspond to the center of the torso (r1) and the end effector (r2).
        """
        return 2

    def get_sparsity_in(self, i):
        """
        Both input arguments r1 and r2 are in 3-D coordinates.
        """
        return Sparsity.dense(3, 1)

    def get_sparsity_out(self, i):
        return Sparsity.dense(1, 1)

    def has_jacobian(self, *_args):
        return True

    # def get_jacobian(self, *args):
    #     G1 = self.G1
    #     G2 = self.G2
    #
    #     # provide first order Jacobians
    #     r1 = MX.sym('r1', 3, 1)
    #     r2 = MX.sym('r2', 3, 1)
    #     x = MX.sym('x', 3, 1)
    #     alpha = MX.sym('alpha', 1)
    #     f = MX(1, 1)
    #     # xsol = vertcat(x, alpha)
    #     xsol = self.xsol
    #
    #     # reconstruct cone matrices with implicit parameters
    #     h1 = self.A1 @ self.Q.T @ r1
    #     h2 = self.A2 @ self.Q.T @ r2
    #     h = vertcat(h1, h2)
    #     G = vertcat(G1, G2)
    #
    #     # optimality condition
    #     g_impl = G @ xsol - h
    #     # g_impl = Function('g_impl', [x, alpha, r1, r2, f], [G @ xsol - h])
    #     grad_g = jacobian(g_impl, r1)
    #     grad_eval = self.z[:-1].T @ grad_g
    #
    #     # g_impl = Function('g_impl', [x, alpha, r1, r2, f], [self.z[:-1].T @ jacobian(G @ xsol - h, r1)])
    #     # z = self.z
    #     # f = MX(1, 1)
    #     # grad_r1 = jacobian(solve_polytope_min_prox, r1)
    #     # grad_r2 = jacobian(solve_polytope_min_prox, r2)
    #     # Df =  Function('Df', [x, f], [z.T @ (G1 @ x - h1)])
    #
    #     return g_impl

    def get_jacobian(self, name, inames, onames, opts):
        class JacFun(Callback):
            def __init__(self, A1, b1, A2, b2, Q, z, opts={}):
                Callback.__init__(self)
                self.A1 = A1
                self.b1 = b1
                self.A2 = A2
                self.b2 = b2
                self.Q = Q
                self.z = z
                self.construct(name, opts)

            def get_n_in(self):
                return 3

            def get_n_out(self):
                return 2

            def get_sparsity_in(self, i):
                if (i == 0) or (i == 1):  # nominal input
                    return Sparsity.dense(3, 1)
                elif i == 2:  # nominal output
                    return Sparsity(1, 1)

            def get_sparsity_out(self, i):
                if (i == 0) or (i == 1):
                    return Sparsity.dense(1, 3)

            # Evaluate numerically
            def eval(self, arg):
                r1 = np.array(arg[0])
                r2 = np.array(arg[1])

                # reconstruct cone matrices with implicit parameters
                # h1 = self.A1 @ self.Q.T @ r1
                # h2 = self.A2 @ self.Q.T @ r2
                # h = np.concatenate((h1, h2))
                # G = np.concatenate((self.G1, self.G2))

                ret1 = self.z[:-1].T @ (np.concatenate((-self.A1 @ self.Q.T, np.zeros((6, 3)))))
                ret2 = self.z[:-1].T @ (np.concatenate((np.zeros((6, 3)), -self.A2 @ self.Q.T)))

                return [ret1, ret2]

        # It is required to keep a reference alive to the returned Callback object
        self.jac_callback = JacFun(self.A1, self.b1, self.A2, self.b2, self.Q, self.z)
        return self.jac_callback


    def eval(self, arg):
        r1 = np.array(arg[0])   # torso center
        r2 = np.array(arg[1])   # end effector (sphere) center

        # find distance via optimization (dcol)
        x = cp.Variable((3,1))
        alpha = cp.Variable(1)

        A1 = self.A1
        b1 = self.b1
        A2 = self.A2
        b2 = self.b2
        Q = self.Q
        constraints = []
        constraints.append(A1 @ Q.T @ (x - r1) <= alpha * b1)
        constraints.append(A2 @ Q.T @ (x - r2) <= alpha * b2)
        constraints.append(alpha >= 0)
        prob = cp.Problem(cp.Minimize(alpha), constraints)
        prob.solve(solver='CLARABEL')

        alpha_val = alpha.value
        x_val = x.value
        solver_time = prob.solver_stats.solve_time

        self.xsol = np.concatenate((x_val, alpha_val.reshape((1,1))))

        min_distance = np.linalg.norm(r1 - r2 + (r2 - r1) / alpha_val)
        cp_torso = r1 + (x_val - r1) / alpha_val
        cp_foot = r2 + (x_val - r2) / alpha_val
        print(f"alpha: {alpha_val}")
        print(f"contact in torso: {cp_torso.T}, contact in foot: {cp_foot.T}")
        print(f"min distance: {min_distance}")

        # self.h1 = self.A1 @ self.Q.T @ r1
        # self.h2 = self.A2 @ self.Q.T @ r2
        z_lst = list((prob.solution.dual_vars).values())
        z_lst[-1] = np.reshape(z_lst[-1], (1,1))
        self.z = np.concatenate(z_lst)

        return [min_distance]

        min_distance = np.linalg.norm(r1 - r2 + (r2 - r1) / alpha_val)
        cp_torso = r1 + (x_val - r1) / alpha_val
        cp_sphere = r2 + (x_val - r2) / alpha_val
        print(f"alpha: {alpha_val}")
        print(f"contact in torso: {cp_torso}, contact in sphere: {cp_sphere}")
        print(f"min distance: {min_distance}")

        return [min_distance]