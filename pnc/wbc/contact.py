import abc
import numpy as np


class Contact(abc.ABC):
    def __init__(self, robot, dim):
        self._robot = robot
        self._dim_contact = dim
        self._rf_z_idx = self._dim_contact - 1
        self._jacobian = np.zeros((self._dim_contact, self._robot.n_q))
        self._jacobian_dot_q_dot = np.zeros(self._dim_contact)
        self._rf_z_max = 0.
        self._cone_constraint_mat = None
        self._cone_constraint_vec = None

    @property
    def jacobian(self):
        return self._jacobian

    @property
    def jacobian_dot_q_dot(self):
        return self._jacobian_dot_q_dot

    @property
    def dim_contact(self):
        return self._dim_contact

    @rf_z_max.setter
    def rf_z_max(self, value):
        self._rf_z_max = value

    @property
    def cone_constraint_mat(self):
        return self._cone_constraint_mat

    @property
    def cone_constraint_vec(self):
        return self._cone_constraint_vec

    @property
    def rf_z_idx(self):
        return self._rf_z_idx

    def update_contact_spec(self):
        self._update_jacobian()
        self._update_cone_constraint()
        return True

    @abc.abstractmethod
    def _update_jacobian(self):
        pass

    @abc.abstractmethod
    def _update_cone_constraint(self):
        pass