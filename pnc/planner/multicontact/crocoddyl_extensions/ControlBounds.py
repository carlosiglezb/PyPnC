import crocoddyl
import numpy as np


class ControlBounds(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu=None):
        """This residual model defines control bounds as a cost.

        :param state: state of the controlled system
        :param nu: dimension of the control input. If None it is set to state.nv / 2
        """
        if nu is None:
            nu = np.floor(state.ndx / 2)
        crocoddyl.ResidualModelAbstract.__init__(self, state, nu, nu, False, False, True)
        self.dfdx = np.zeros((nu, self.state.ndx))
        self.dfdu = np.eye(nu)

    def calc(self, data, x, u=None):
        """Compute the control bounds residual.

        :param data: control bounds residual data
        :param x: state point (dim. state.nx)
        :param u: control input (dim. nu)
        """
        if u is None:
            raise ValueError("[ControlBounds] Control input is required for calc.")
        data.r[: self.nu] = u

    def calcDiff(self, data, x, u=None):
        """Compute the derivatives of the control bounds residual.

        :param data: control bounds residual data
        :param x: state point (dim. state.nx)
        :param u: control input (dim. nu)
        """
        data.Rx[:] = self.dfdx
        data.Ru[:] = self.dfdu