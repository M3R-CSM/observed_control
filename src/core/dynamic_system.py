# core/dynamic_system.py
import abc
from typing import Callable, Tuple

import autograd.numpy as np
from autograd import jacobian
from scipy.integrate import solve_ivp

class DynamicSystemBase(abc.ABC):
    """Abstract base class for dynamic systems.

    This class defines the interface for a dynamic system. Subclasses must
    implement the `ode` method, which describes the system's dynamics.
    """

    def __init__(self, n_x: int, n_u: int):
        """Initializes the dynamic system."""
        self.n_x = n_x
        self.n_u = n_u
        # Store autograd functions as a fallback
        self._autograd_jac_x = jacobian(self.ode, 1)
        self._autograd_jac_u = jacobian(self.ode, 2)

    @abc.abstractmethod
    def ode(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """The ordinary differential equation defining the system dynamics."""
        raise NotImplementedError

    def _get_jacobians(self, t: float, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Jacobians of the ODE. This default implementation uses
        autograd. Subclasses can override this for analytical Jacobians.
        """
        return self._autograd_jac_x(t, x, u), self._autograd_jac_u(t, x, u)

    def _augmented_ode(self, t: float, aug_state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """The augmented ODE including state and sensitivity dynamics."""
        # Unpack the augmented state
        x = aug_state[:self.n_x]
        phi_x = aug_state[self.n_x:self.n_x + self.n_x * self.n_x].reshape((self.n_x, self.n_x))
        phi_u = aug_state[self.n_x + self.n_x * self.n_x:].reshape((self.n_x, self.n_u))

        # Compute the ODE and its Jacobians
        ode_val = self.ode(t, x, u)
        ode_jac_x, ode_jac_u = self._get_jacobians(t, x, u)

        # Sensitivity dynamics
        d_phi_x_dt = ode_jac_x @ phi_x
        d_phi_u_dt = ode_jac_x @ phi_u + ode_jac_u

        return np.concatenate([
            ode_val,
            d_phi_x_dt.flatten(),
            d_phi_u_dt.flatten(),
        ])

    def solve(self, t_init: float, t_final: float, x_init: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solves the ODE and computes sensitivities over a time interval."""
        # Initial augmented state
        phi_x_init = np.eye(self.n_x)
        phi_u_init = np.zeros((self.n_x, self.n_u))
        aug_x_init = np.concatenate([
            x_init,
            phi_x_init.flatten(),
            phi_u_init.flatten(),
        ])

        # Solve the augmented ODE
        sol = solve_ivp(self._augmented_ode, (t_init, t_final), aug_x_init, args=(u,),rtol=1e-6,atol=1e-9)
        final_aug_state = sol.y[:,-1]
        # print(sol)
        # print(final_aug_state)
        assert(sol.success)
        # Unpack the final solution
        x_final = final_aug_state[:self.n_x]
        phi_x_final = final_aug_state[self.n_x:self.n_x + self.n_x * self.n_x].reshape((self.n_x, self.n_x))
        phi_u_final = final_aug_state[self.n_x + self.n_x * self.n_x:].reshape((self.n_x, self.n_u))

        return x_final, phi_x_final, phi_u_final

class DynamicSystem(DynamicSystemBase):
    """A concrete implementation of a dynamic system."""
    def __init__(self, n_x: int, n_u: int, ode_func: Callable[[float, np.ndarray, np.ndarray], np.ndarray] = None):
        """Initializes the dynamic system."""
        super().__init__(n_x, n_u)
        if ode_func:
            self._ode_func = ode_func
        elif not hasattr(self, '_ode_func'):
            if type(self) == DynamicSystem:
                 raise ValueError("`ode_func` must be provided if not subclassing.")

    def ode(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """The system's ODE. Delegates to the provided `ode_func`."""
        if hasattr(self, '_ode_func'):
            return self._ode_func(t, x, u)
        raise NotImplementedError("`ode` method must be overridden in a subclass if `value_func` is not provided.")
