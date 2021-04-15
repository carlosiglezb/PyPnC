import sys
import numpy as np
np.set_printoptions(precision=6, threshold=sys.maxsize)
from scipy.linalg import block_diag
from qpsolvers import solve_qp

from pnc.data_saver import DataSaver


class Draco3LBIHWBC(object):
    """
    Implicit Hierarchy Whole Body Control
    ------------------
    Usage:
        update_setting --> solve
    """
    def __init__(self, act_list, jac_int, data_save=False):
        self._n_q_dot = len(act_list)
        self._n_active = np.count_nonzero(np.array(act_list))
        self._n_passive = self._n_q_dot - self._n_active - 6

        # Selection matrix
        self._sa = np.zeros((self._n_active, self._n_q_dot))
        self._sv = np.zeros((self._n_passive, self._n_q_dot))
        j, k = 0, 0
        for i in range(self._n_q_dot):
            if i >= 6:
                if act_list[i]:
                    self._sa[j, i] = 1.
                    j += 1
                else:
                    self._sv[k, i] = 1.
                    k += 1

        # Assume first six is floating
        self._sf = np.zeros((6, self._n_q_dot))
        self._sf[0:6, 0:6] = np.eye(6)

        # Internal constraint
        self._n_int = jac_int.shape[0]
        self._j_int = np.copy(jac_int)
        self._lambda_int = np.linalg.pinv(
            np.dot(self._j_int, self._j_int.transpose()))
        self._j_bar_int = np.dot(self._j_int.transpose(), self._lambda_int)
        self._null_int = np.eye(self._n_q_dot) - np.dot(
            self._j_bar_int, self._j_int)
        sa_times_null_int = np.dot(self._sa, self._null_int)
        self._sa_times_null_int_transpose_inv = np.linalg.pinv(
            sa_times_null_int.transpose())

        self._trq_limit = None
        self._lambda_q_ddot = 0.
        self._lambda_if = 0.
        self._lambda_rf = 0.
        self._w_hierarchy = 0.

        self._b_data_save = data_save
        if self._b_data_save:
            self._data_saver = DataSaver()

    @property
    def trq_limit(self):
        return self._trq_limit

    @property
    def lambda_q_ddot(self):
        return self._lambda_q_ddot

    @property
    def lambda_rf(self):
        return self._lambda_rf

    @property
    def lambda_if(self):
        return self._lambda_if

    @property
    def w_hierarchy(self):
        return self._w_hierarchy

    @trq_limit.setter
    def trq_limit(self, val):
        assert val.shape[0] == self._n_active
        self._trq_limit = np.copy(val)

    @lambda_q_ddot.setter
    def lambda_q_ddot(self, val):
        self._lambda_q_ddot = val

    @lambda_rf.setter
    def lambda_rf(self, val):
        self._lambda_rf = val

    @lambda_if.setter
    def lambda_if(self, val):
        self._lambda_if = val

    @w_hierarchy.setter
    def w_hierarchy(self, val):
        self._w_hierarchy = val

    def update_setting(self, mass_matrix, mass_matrix_inv, coriolis, gravity):
        self._mass_matrix = np.copy(mass_matrix)
        self._mass_matrix_inv = np.copy(mass_matrix_inv)
        self._coriolis = np.copy(coriolis)
        self._gravity = np.copy(gravity)

    def solve(self, task_list, contact_list, verbose=False):
        """
        Parameters
        ----------
        task_list (list of Task):
            Task list
        contact_list (list of Contact):
            Contact list
        verbose (bool):
            Printing option

        Returns
        -------
        joint_trq_cmd (np.array):
            Joint trq cmd
        joint_acc_cmd (np.array):
            Joint acc cmd
        sol_rf (np.array):
            Reaction force
        sol_if (np.array):
            Internal force
        """

        # ======================================================================
        # Cost
        # ======================================================================
        cost_t_mat = np.zeros((self._n_q_dot, self._n_q_dot))
        cost_t_vec = np.zeros(self._n_q_dot)

        for i, task in enumerate(task_list):
            j = task.jacobian
            j_dot_q_dot = task.jacobian_dot_q_dot
            x_ddot = task.op_cmd
            if verbose:
                print(i, " th task")
                task.debug()

            cost_t_mat += self._w_hierarchy[i] * np.dot(j.transpose(), j)
            cost_t_vec += self._w_hierarchy[i] * np.dot(
                (j_dot_q_dot - x_ddot).transpose(), j)
        # cost_t_mat += self._lambda_q_ddot * np.eye(self._n_q_dot)
        cost_t_mat += self._lambda_q_ddot * self._mass_matrix

        if contact_list is not None:
            uf_mat = np.array(
                block_diag(
                    *[contact.cone_constraint_mat
                      for contact in contact_list]))
            uf_vec = np.concatenate(
                [contact.cone_constraint_vec for contact in contact_list])
            contact_jacobian = np.concatenate(
                [contact.jacobian for contact in contact_list], axis=0)

            assert uf_mat.shape[0] == uf_vec.shape[0]
            assert uf_mat.shape[1] == contact_jacobian.shape[0]
            dim_cone_constraint, dim_contacts = uf_mat.shape

            cost_rf_mat = self._lambda_rf * np.eye(dim_contacts)
            cost_rf_vec = np.zeros(dim_contacts)

            cost_mat = np.array(block_diag(
                cost_t_mat, cost_rf_mat))  # (nqdot+nc, nqdot+nc)
            cost_vec = np.concatenate([cost_t_vec, cost_rf_vec])  # (nqdot+nc,)

        else:
            dim_contacts = dim_cone_constraint = 0
            cost_mat = np.copy(cost_t_mat)
            cost_vec = np.copy(cost_t_vec)

        if verbose:
            print("cost_t_mat")
            print(cost_t_mat)
            print("cost_t_vec")
            print(cost_t_vec)
            print("cost_rf_mat")
            print(cost_rf_mat)
            print("cost_rf_vec")
            print(cost_rf_vec)

        # ======================================================================
        # Equality Constraint
        # ======================================================================

        if contact_list is not None:
            eq_floating_mat = np.concatenate((
                np.dot(self._sf, self._mass_matrix),
                -np.dot(self._sf,
                        np.dot(contact_jacobian, self._null_int).transpose())),
                                             axis=1)  # (6, nqdot+nc)
            eq_int_mat = np.concatenate(
                (self._j_int, np.zeros(
                    (2, dim_contacts))), axis=1)  # (2, nqdot+nc)
        else:
            eq_floating_mat = np.dot(self._sf, self._mass_matrix)
            eq_int_mat = np.copy(self._j_int)
        eq_floating_vec = -np.dot(
            self._sf, np.dot(self._null_int, (self._coriolis + self._gravity)))
        eq_int_vec = np.zeros(2)

        eq_mat = np.concatenate((eq_floating_mat, eq_int_mat), axis=0)
        eq_vec = np.concatenate((eq_floating_vec, eq_int_vec), axis=0)

        # ======================================================================
        # Inequality Constraint
        # ======================================================================

        if self._trq_limit is None:
            if contact_list is not None:
                ineq_mat = np.concatenate((np.zeros(
                    (dim_cone_constraint, self._n_q_dot)), -uf_mat),
                                          axis=1)
                ineq_vec = -uf_vec
            else:
                ineq_mat = None
                ineq_vec = None

        else:
            ## TODO
            if contact_list is not None:
                ineq_mat = np.concatenate(
                    (np.concatenate(
                        (np.zeros((dim_cone_constraint, self._n_q_dot)),
                         -np.dot(self._sa_times_null_int_transpose_inv,
                                 self._mass_matrix),
                         np.dot(self._sa_times_null_int_transpose_inv,
                                self._mass_matrix)),
                        axis=0),
                     np.concatenate(
                         (-uf_mat,
                          np.dot(
                              self._sa_times_null_int_transpose_inv,
                              np.dot(contact_jacobian,
                                     self._null_int).transpose()), -np.dot(
                                         self._sa_times_null_int_transpose_inv,
                                         np.dot(contact_jacobian,
                                                self._null_int).transpose())),
                         axis=0)),
                    axis=1)
                ineq_vec = np.concatenate(
                    (-uf_vec,
                     np.dot(
                         self._sa_times_null_int_transpose_inv,
                         np.dot(self._null_int,
                                (self._coriolis + self._gravity))) -
                     self._trq_limit[:, 0], -np.dot(
                         self._sa_times_null_int_transpose_inv,
                         np.dot(self._null_int,
                                (self._coriolis + self._gravity))) +
                     self._trq_limit[:, 1]))
            else:
                ineq_mat = np.concatenate(
                    (-np.dot(self._sa_times_null_int_transpose_inv,
                             self._mass_matrix),
                     np.dot(self._sa_times_null_int_transpose_inv,
                            self._mass_matrix)),
                    axis=0)
                ineq_vec = np.concatenate(
                    (np.dot(
                        self._sa_times_null_int_transpose_inv,
                        np.dot(self._null_int,
                               (self._coriolis + self._gravity))) -
                     self._trq_limit[:, 0], -np.dot(
                         self._sa_times_null_int_transpose_inv,
                         np.dot(self._null_int,
                                (self._coriolis + self._gravity))) +
                     self._trq_limit[:, 1]))

        if verbose:
            print("eq_mat")
            print(eq_mat)
            print("eq_vec")
            print(eq_vec)

            print("ineq_mat")
            print(ineq_mat)
            print("ineq_vec")
            print(ineq_vec)

        sol = solve_qp(cost_mat,
                       cost_vec,
                       ineq_mat,
                       ineq_vec,
                       eq_mat,
                       eq_vec,
                       solver="quadprog",
                       verbose=True)

        if contact_list is not None:
            sol_q_ddot, sol_rf = sol[:self._n_q_dot], sol[self._n_q_dot:]
        else:
            sol_q_ddot, sol_rf = sol, None

        if contact_list is not None:
            joint_trq_cmd = np.dot(
                self._sa_times_null_int_transpose_inv,
                np.dot(self._mass_matrix, sol_q_ddot) +
                np.dot(self._null_int,
                       (self._coriolis + self._gravity)) - np.dot(
                           np.dot(contact_jacobian,
                                  self._null_int).transpose(), sol_rf))
        else:
            joint_trq_cmd = np.dot(
                self._sa_times_null_int_transpose_inv,
                np.dot(self._mass_matrix, sol_q_ddot) +
                np.dot(self._null_int, (self._coriolis + self._gravity)))

        joint_acc_cmd = np.dot(self._sa, sol_q_ddot)

        if verbose:
            print("joint_trq_cmd: ", joint_trq_cmd)
            print("sol_q_ddot: ", sol_q_ddot)
            print("sol_rf: ", sol_rf)
            for i, task in enumerate(task_list):
                j = task.jacobian
                j_dot_q_dot = task.jacobian_dot_q_dot
                x_ddot = task.op_cmd
                print(i, " th task")
                print("des x ddot: ", x_ddot)
                print("J*qddot_sol + Jdot*qdot: ",
                      np.dot(j, sol_q_ddot) + j_dot_q_dot)

        # print("joint_trq_cmd: ", joint_trq_cmd)
        # print("sol_q_ddot: ", sol_q_ddot)
        # print("sol_rf: ", sol_rf)

        # for i, task in enumerate(task_list):
        # j = task.jacobian
        # j_dot_q_dot = task.jacobian_dot_q_dot
        # x_ddot = task.op_cmd
        # print(i, " th task")
        # print("des x ddot: ", x_ddot)
        # print("J*qddot_sol + Jdot*qdot: ",
        # np.dot(j, sol_q_ddot) + j_dot_q_dot)
        # __import__('ipdb').set_trace()

        if self._b_data_save:
            self._data_saver.add('joint_trq_cmd', joint_trq_cmd)
            self._data_saver.add('joint_acc_cmd', joint_acc_cmd)
            self._data_saver.add('rf_cmd', sol_rf)

        return joint_trq_cmd, joint_acc_cmd, sol_rf
