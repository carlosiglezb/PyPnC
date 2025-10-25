import numpy as np

from crocoddyl.libcrocoddyl_pywrap import *

class ActivationDataDistanceQuad(ActivationDataAbstract):
    """
    Data structure for the `ActivationModelDistanceQuad`.
    This class inherits from `ActivationDataAbstract` and contains `a_value`, `Ar`, `Arr`.
    The original C++ `dd` and `one_minus_dd` fields are not needed for the `exp(-x/d)` implementation.
    """
    def __init__(self, nr: int):
        """
        Initializes the data structure.

        Args:
            nr (int): The dimension of the residual vector.
        """
        super().__init__(nr)
        # Specific fields like 'dd' and 'one_minus_dd' from the original C++
        # quadratic-exp implementation are omitted as they are not relevant
        # for the requested `exp(-x/d)` function.

class ActivationModelDistanceQuad(ActivationModelAbstract):
    """
    Quadratic-exponential activation model.

    This activation model implements the function:
    $ \sum_{i} \exp\left(-\frac{r_i - d_1}{d_0}\right) $
    where $r_i$ is a component of the residual vector $\mathbf{r}$ and $d_0$ is a scalar parameter.

    The `calc()` method computes this sum, and `calcDiff()` computes its first and
    second derivatives with respect to the residual vector.
    """

    def __init__(self, nr: int, d0: float = 0.1, d1: float = 0.0):
        """
        Initializes the quadratic-exponential activation model.

        Args:
            nr (int): Dimension of the residual vector.
            d0 (float): The 'width' parameter for the exponential function.
                        Must be a strictly positive value (default: 0.1).
                        This corresponds to 'alpha' in the original C++ comments.

        Raises:
            ValueError: If `d0` is not strictly positive.
        """
        super().__init__(nr)
        if d0 <= 0.0:
            raise ValueError("Invalid argument: d0 should be a strictly positive value.")
        self._d0: float = d0
        self._d0inv: float = 1.0 / d0  # Pre-calculate inverse for efficiency
        self._d1: float = d1

    @property
    def d0(self) -> float:
        """
        Gets the current value of the `d0` parameter.
        """
        return self._d0

    @d0.setter
    def d0(self, value: float):
        """
        Sets the `d0` parameter and automatically updates its inverse.

        Args:
            value (float): The new value for `d0`.

        Raises:
            ValueError: If `value` is not strictly positive.
        """
        if value <= 0.0:
            raise ValueError("Invalid argument: d0 should be a strictly positive value.")
        self._d0 = value
        self._d0inv = 1.0 / value

    def calc(self, data: ActivationDataAbstract, r: np.ndarray):
        """
        Computes the activation value: $ \sum_{i} \exp\left(-\frac{r_i - d_1}{d_0}\right) $.

        Args:
            data (ActivationDataAbstract): The data object to store the computed activation value.
            r (np.ndarray): The residual vector $\mathbf{r} \in \mathbb{R}^{nr}$.

        Raises:
            ValueError: If the dimension of `r` does not match `nr`.
        """
        # r = np.linalg.norm(r)

        if r.size != self.nr:
            raise ValueError(
                f"Invalid argument: Residual vector `r` has wrong dimension "
                f"(it should be {self.nr}, but got {r.size})."
            )

        # Calculate exp(-r_i - d_1 / d0) for each element and sum them up.
        # This implements the user's specific request: exp(-x/d) in a summation.
        data.a_value = np.sum(np.exp(-(r - self._d1) / self._d0))

    def calcDiff(self, data: ActivationDataAbstract, r: np.ndarray):
        """
        Computes the first and second derivatives of the activation function.

        For $ A = \sum_{i} \exp\left(-\frac{r_i - d_1}{d_0}\right) $:
        - Gradient ($ \mathbf{A_r} $): $ \frac{\partial A}{\partial r_i} = -\frac{1}{d_0} \exp\left(-\frac{r_i - d_1}{d_0}\right) $
        - Hessian ($ \mathbf{A_{rr}} $, diagonal elements): $ \frac{\partial^2 A}{\partial r_i^2} = \left(\frac{1}{d_0}\right)^2 \exp\left(-\frac{r_i}{d_0}\right) $

        Args:
            data (ActivationDataAbstract): The data object to store the computed derivatives.
            r (np.ndarray): The residual vector $\mathbf{r} \in \mathbb{R}^{nr}$.

        Raises:
            ValueError: If the dimension of `r` does not match `nr`.
        """
        # r = np.linalg.norm(r)
        if r.size != self.nr:
            raise ValueError(
                f"Invalid argument: Residual vector `r` has wrong dimension "
                f"(it should be {self.nr}, but got {r.size})."
            )

        exp_term = np.exp(-(r - self._d1) / self._d0)

        # Compute the gradient vector (Ar)
        data.Ar = -self._d0inv * exp_term

        # Compute the Hessian matrix (Arr), assuming it's diagonal as per C++ `Arr.diagonal()`
        data.Arr = np.diag((self._d0inv ** 2) * exp_term)

    # def createData(self) -> ActivationDataDistanceQuad:
    #     """
    #     Creates an instance of `ActivationDataDistanceQuad` for this model.
    #
    #     Returns:
    #         ActivationDataDistanceQuad: A new data object specifically for this activation model.
    #     """
    #     return ActivationDataDistanceQuad(self)

    def __str__(self) -> str:
        """
        Returns a descriptive string representation of the model.
        """
        return f"ActivationModelDistanceQuad {{nr={self.nr}, d0={self._d0}}}"