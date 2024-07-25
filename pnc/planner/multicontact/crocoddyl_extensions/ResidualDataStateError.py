from crocoddyl.libcrocoddyl_pywrap import *


class ResidualDataStateError(ResidualDataAbstract):
    def copy(self, ResidualDataStateError, *args, **kwargs):
        return self.copy(*args, **kwargs)

    def __copy__(self, ResidualDataStateError, *args, **kwargs):
        raise NotImplementedError("[ResidualDataStateError] Copy is not implemented in Python")

    def __deepcopy__(self, ResidualDataStateError, *args, **kwargs):
        raise NotImplementedError("[ResidualDataStateError] Deep copy method is not implemented in Python")

    def __init__(self, p_object, *args, **kwargs):
        """
            :param model: residual model
            :param data: shared data
        """
        super().__init__(p_object, *args, **kwargs)

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        raise NotImplementedError("[ResidualDataStateError] Reduce is not implemented in Python")

    pinocchio = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """pinocchio data"""


