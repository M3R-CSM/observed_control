# core/anticipated_condition.py
import abc
from typing import Callable

import autograd.numpy as np
from autograd import hessian, jacobian

class AnticipatedConditionBase(abc.ABC):
    """Abstract base class for anticipated conditions (cost functions).

    This class defines the interface for a condition that the MPC controller
    will anticipate and optimize for. Subclasses must implement the `value`
    method, which calculates the cost.
    """

    def __init__(self, n_x: int, n_u: int):
        """Initializes the anticipated condition.

        Args:
            n_x: The number of states in the system.
            n_u: The number of control inputs in the system.
        """
        self.n_x = n_x
        self.n_u = n_u

        # A helper function to combine state and control for autograd
        def _helper_func(params, t):
            x = params[:self.n_x]
            u = params[self.n_x:]
            return self.value(t, x, u)

        self._sensitivity_func = jacobian(_helper_func, 0)
        self._hessian_func = hessian(_helper_func, 0)

    @abc.abstractmethod
    def value(self, t: float, x: np.ndarray, u: np.ndarray) -> float:
        """The value (cost) of the condition at a given time, state, and control.

        Args:
            t: The current time.
            x: The state vector.
            u: The control vector.

        Returns:
            The scalar cost value.
        """
        raise NotImplementedError

    def sensitivity(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes the sensitivity (gradient) of the condition's value.

        Args:
            t: The current time.
            x: The state vector.
            u: The control vector.

        Returns:
            The gradient of the value with respect to the state and control.
        """
        params = np.concatenate((x, u))
        return self._sensitivity_func(params, t)

    def hessian(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes the Hessian of the condition's value.

        Args:
            t: The current time.
            x: The state vector.
            u: The control vector.

        Returns:
            The Hessian matrix of the value with respect to the state and control.
        """
        params = np.concatenate((x, u))
        return self._hessian_func(params, t)

class AnticipatedCondition(AnticipatedConditionBase):
    """A concrete implementation of an anticipated condition.

    This class can be used in two ways:
    1.  By passing a `value_func` to the constructor.
    2.  By subclassing and overriding the `value` method.
    """

    def __init__(self, n_x: int, n_u: int, value_func: Callable[[float, np.ndarray, np.ndarray], float] = None):
        """Initializes the condition.

        Args:
            n_x: The number of states.
            n_u: The number of controls.
            value_func: A function with the signature `f(t, x, u)` that
                computes the cost. If None, this class must be
                subclassed and the `value` method must be overridden.
        """
        super().__init__(n_x, n_u)
        if value_func:
            self._value_func = value_func
        elif not hasattr(self, '_value_func'):
             if type(self) == AnticipatedCondition:
                 raise ValueError("`value_func` must be provided if not subclassing.")

    def value(self, t: float, x: np.ndarray, u: np.ndarray) -> float:
        """The condition's value. Delegates to the provided `value_func`."""
        if hasattr(self, '_value_func'):
            return self._value_func(t, x, u)
        raise NotImplementedError("`value` method must be overridden in a subclass if `value_func` is not provided.")
