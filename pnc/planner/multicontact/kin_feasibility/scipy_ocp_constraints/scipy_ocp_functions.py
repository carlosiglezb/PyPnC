import copy

import numpy as np
from scipy.optimize import LinearConstraint

class LinearBezierIneqConstraint:
    def __init__(self, x_dim, num_constraints):
        self.x_dim = x_dim
        self.A = np.zeros((num_constraints, x_dim))
        self.b = np.zeros((num_constraints, ))
        self.k = 0                  # current constraint index
        self.constraints = []

    def add_lin_ineq(self, Ak, bk, x_idx):
        k = self.k
        Ak_dim = Ak.shape[0]
        self.A[k: k+Ak_dim, x_idx:x_idx+3] = Ak
        self.b[k: k+Ak_dim] = bk
        self.k += Ak.shape[0]

    def get_constraints(self):
        A = self.A
        b = self.b
        # return {'type': 'ineq', 'fun': self.fun_iris_mat_ineq, 'args': (A, b, k)}
        return LinearConstraint(A, ub=b)


    # def fun_iris_mat_ineq(self, x, A, b, k):
    #     return b - A[k] @ x


class LinearBezierEqConstraint:
    def __init__(self, x_dim, num_constraints):
        self.x_dim = x_dim
        self.A = np.zeros((num_constraints, x_dim))
        self.c = np.zeros((num_constraints, ))
        self.k = 0                  # current constraint index
        self.constraints = []

    def add_lin_eq(self, Ak):
        k = self.k
        Ak_dim = Ak.shape[0]
        self.A[k: k+Ak_dim, :] = copy.deepcopy(Ak)     # copy?
        self.k += copy.deepcopy(Ak_dim)

    def get_constraints(self):
        A = self.A
        c = self.c
        return LinearConstraint(A, lb=-c, ub=c)
