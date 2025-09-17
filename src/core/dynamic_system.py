# core/dynamic_system.py
import abc
from typing import Callable, Tuple

import autograd.numpy as np
from autograd import jacobian
from scipy.integrate import odeint

class DynamicSystemBase(abc.ABC):
    """Abstract base class for dynamic systems.

    This class defines the interface for a dynamic system. Subclasses must
    implement the `ode` method, which describes the system's dynamics.
    """

    def __init__(self, n_x: int, n_u: int):
        """Initializes the dynamic system.

        Args:
            n_x: The number of states in the system.
            n_u: The number of control inputs in the system.
        """
        self.n_x = n_x
        self.n_u = n_u

        # Autograd jacobians for the ODE function
        self._jac_x = jacobian(self.ode, 1)
        self._jac_u = jacobian(self.ode, 2)

    @abc.abstractmethod
    def ode(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """The ordinary differential equation defining the system dynamics.

        Args:
            t: The current time.
            x: The state vector.
            u: The control vector.

        Returns:
            The derivative of the state vector.
        """
        raise NotImplementedError

    def _augmented_ode(self, aug_state: np.ndarray, t: float, u: np.ndarray) -> np.ndarray:
        """The augmented ODE including state and sensitivity dynamics.

        This private method is used by the `solve` method to integrate the
        state and its sensitivities (Jacobians) simultaneously.

        Args:
            aug_state: The augmented state vector, containing the system state,
                       the state sensitivity matrix (Phi_x), and the control
                       sensitivity matrix (Phi_u).
            t: The current time.
            u: The control vector.

        Returns:
            The derivative of the augmented state vector.
        """
        # Unpack the augmented state
        x = aug_state[:self.n_x]
        phi_x = aug_state[self.n_x:self.n_x + self.n_x * self.n_x].reshape((self.n_x, self.n_x))
        phi_u = aug_state[self.n_x + self.n_x * self.n_x:].reshape((self.n_x, self.n_u))

        # Compute the ODE and its Jacobians
        ode_val = self.ode(t, x, u)
        ode_jac_x = self._jac_x(t, x, u)
        ode_jac_u = self._jac_u(t, x, u)

        # Sensitivity dynamics
        d_phi_x_dt = ode_jac_x @ phi_x
        d_phi_u_dt = ode_jac_x @ phi_u + ode_jac_u

        return np.concatenate([
            ode_val,
            d_phi_x_dt.flatten(),
            d_phi_u_dt.flatten(),
        ])

    def solve(self, t_init: float, t_final: float, x_init: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solves the ODE and computes sensitivities over a time interval.

        Args:
            t_init: The initial time.
            t_final: The final time.
            x_init: The initial state vector.
            u: The control vector, assumed constant over the interval.

        Returns:
            A tuple containing:
                - The final state vector.
                - The state sensitivity matrix (Phi_x).
                - The control sensitivity matrix (Phi_u).
        """
        # Initial augmented state
        phi_x_init = np.eye(self.n_x)
        phi_u_init = np.zeros((self.n_x, self.n_u))
        aug_x_init = np.concatenate([
            x_init,
            phi_x_init.flatten(),
            phi_u_init.flatten(),
        ])

        # Solve the augmented ODE
        sol = odeint(self._augmented_ode, aug_x_init, [t_init, t_final], args=(u,))
        final_aug_state = sol[-1]

        # Unpack the final solution
        x_final = final_aug_state[:self.n_x]
        phi_x_final = final_aug_state[self.n_x:self.n_x + self.n_x * self.n_x].reshape((self.n_x, self.n_x))
        phi_u_final = final_aug_state[self.n_x + self.n_x * self.n_x:].reshape((self.n_x, self.n_u))

        return x_final, phi_x_final, phi_u_final

class DynamicSystem(DynamicSystemBase):
    """A concrete implementation of a dynamic system.

    This class can be used in two ways:
    1.  By passing an `ode_func` to the constructor.
    2.  By subclassing and overriding the `ode` method.
    """

    def __init__(self, n_x: int, n_u: int, ode_func: Callable[[float, np.ndarray, np.ndarray], np.ndarray] = None):
        """Initializes the dynamic system.

        Args:
            n_x: The number of states.
            n_u: The number of controls.
            ode_func: A function with the signature `f(t, x, u)` that
                computes the state derivative. If None, this class must be
                subclassed and the `ode` method must be overridden.
        """
        super().__init__(n_x, n_u)
        if ode_func:
            self._ode_func = ode_func
        elif not hasattr(self, '_ode_func'):
            # This check is for when a subclass is overriding the `ode` method
            # but did not provide an `ode_func` to the constructor.
            if type(self) == DynamicSystem:
                 raise ValueError("`ode_func` must be provided if not subclassing.")

    def ode(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """The system's ODE. Delegates to the provided `ode_func`."""
        if hasattr(self, '_ode_func'):
            return self._ode_func(t, x, u)
        raise NotImplementedError("`ode` method must be overridden in a subclass if `ode_func` is not provided.")
