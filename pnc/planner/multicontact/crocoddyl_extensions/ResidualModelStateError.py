import numpy as np

from crocoddyl.libcrocoddyl_pywrap import *
from pnc.planner.multicontact.crocoddyl_extensions.ResidualDataStateError import ResidualDataStateError


class ResidualModelStateError(ResidualModelAbstract):
    def calc(self, data, *args, **kwargs):
        qdot = args[0]
        tau = args[1]

        # implement tau(id1) - tau(id2) = 0
        id1 = self.constr_ids[0]
        id2 = self.constr_ids[1]
        # data.h[1] = tau[self.constr_ids_u[0]] - tau[self.constr_ids_u[1]]
        # data.h[0] = tau[self.constr_ids[0]] - tau[self.constr_ids[1]]
        data.r[0] = qdot[id1] - qdot[id2]
        # data.residual.r[1] = tau[self.constr_ids_u[0]] - tau[self.constr_ids_u[1]]

    def calcDiff(self, data, *args, **kwargs):
        lx = np.zeros(self.state.nq - 1 + self.state.nv)
        # lu = np.zeros(self.nu)
        # lx = np.zeros(self.nu)
        lx[self.constr_ids[0] - 1] = 1
        lx[self.constr_ids[1] - 1] = -1
        # lu[self.constr_ids_u[0]] = 1
        # lu[self.constr_ids_u[1]] = -1
        # data.Hx = lx
        data.Rx = lx
        # data.residual.Hu = lx
        # data.Hu[1] = lu
        # data.residual.Rx = lx
        # data.residual.Ru[1] = lu

    def copy(self, ResidualModelStateError, *args, **kwargs):
        return self.copy(*args, **kwargs)

    def createData(self, ResidualModelStateError_x, *args, **kwargs):
        return ResidualDataStateError(self, ResidualModelStateError_x, *args, **kwargs)

    def __init__(self, state, *args, **kwargs):
        super().__init__(state, *args, **kwargs)

    def __copy__(self, ResidualModelStateError, *args, **kwargs):
        raise NotImplementedError("[ResidualModelStateError] Copy is not implemented in Python")

    def __deepcopy__(self, ResidualModelStateError, *args, **kwargs):
        raise NotImplementedError("[ResidualModelStateError] Deep Copy is not implemented in Python")

    """IDs in Pinocchio model for which the RCJ constraint is applied"""
    @property
    def constr_ids(self) -> list[int]:
        return self._constr_ids
    @constr_ids.setter
    def constr_ids(self, constr_ids):
        self._constr_ids = constr_ids

    @property
    def constr_ids_u(self) -> list[int]:
        return self._constr_ids_u
    @constr_ids_u.setter
    def constr_ids_u(self, constr_ids):
        self._constr_ids_u = constr_ids
