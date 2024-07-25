import copy
from crocoddyl.libcrocoddyl_pywrap import *

from pnc.planner.multicontact.crocoddyl_extensions.ConstraintDataRCJ import ConstraintDataRCJ


class ConstraintModelRCJ(ConstraintModelAbstract):
    """ This defines equality Rolling Contact Joint constraint between two joint IDs. """

    def calc(self, ConstraintModelRCJ, *args, **kwargs):
        self.residual.calc(ConstraintModelRCJ, *args, **kwargs)

    def calcDiff(self, ConstraintModelRCJ, *args, **kwargs):
        self.residual.calcDiff(ConstraintModelRCJ, *args, **kwargs)

    def copy(self, ConstraintModelRCJ, *args, **kwargs):
        return self.copy(*args, **kwargs)

    def createData(self, ConstraintModelRCJ_x, *args, **kwargs):
        return ConstraintDataRCJ(self, ConstraintModelRCJ_x, *args, **kwargs)

    def __copy__(self, ConstraintModelRCJ, *args, **kwargs):
        raise NotImplementedError("[ConstraintModelRCJ] Copy is not implemented in Python")

    def __deepcopy__(self, ConstraintModelRCJ, *args, **kwargs):
        raise NotImplementedError("[ConstraintModelRCJ] Deep Copy is not implemented in Python")

    def __init__(self, p_object, *args, **kwargs):
        super().__init__(p_object, *args, **kwargs)

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        raise NotImplementedError("[ConstraintModelRCJ] Reduce is not implemented in Python")

