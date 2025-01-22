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

        min_distance = np.linalg.norm(r1 - r2 + (r2 - r1) / alpha_val)
        cp_torso = r1 + (x_val - r1) / alpha_val
        cp_sphere = r2 + (x_val - r2) / alpha_val
        print(f"alpha: {alpha_val}")
        print(f"contact in torso: {cp_torso}, contact in sphere: {cp_sphere}")
        print(f"min distance: {min_distance}")

        return [min_distance]