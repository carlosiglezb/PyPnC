from crocoddyl.libcrocoddyl_pywrap import *


class ConstraintDataRCJ(ConstraintDataAbstract):
    """ This defines equality Rolling Contact Joint constraint between two joint IDs. """
    def copy(self, ConstraintDataRCJ, *args, **kwargs):
        raise NotImplementedError("[ConstraintDataRCJ] Copy is not implemented in Python")

    def __copy__(self, ConstraintDataRCJ, *args, **kwargs):
        raise NotImplementedError("[ConstraintDataRCJ] _Copy_ is not implemented in Python")

    def __deepcopy__(self, ConstraintDataRCJ, *args, **kwargs):
        raise NotImplementedError("[ConstraintDataRCJ] Deep Copy is not implemented in Python")

    def __init__(self, p_object, *args, **kwargs):
        super().__init__(p_object, *args, **kwargs)

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        raise NotImplementedError("[ConstraintDataRCJ] Reduce is not implemented in Python")
