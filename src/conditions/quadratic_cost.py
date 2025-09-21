# Copyright (c) 2025 Andrew Petruska
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# conditions/quadratic_cost.py

"""Defines a quadratic cost function for optimal control problems."""

from typing import Callable, Tuple

import numpy as np
from core.anticipated_condition import AnticipatedConditionBase


class QuadraticCost(AnticipatedConditionBase):
    """Represents a quadratic cost of the form:

    c(t, x, u) = 0.5 * dx'Q*dx + 0.5 * du'R*du + dx'M*du

    Where:
        dx = x - x_target(t)
        du = u - u_target(t)

    This class provides closed-form analytical solutions for the cost value,
    its sensitivity (gradient), and its Hessian matrix.
    """

    def __init__(self,
                 target_function: Callable[[float], Tuple[np.ndarray, np.ndarray]],
                 Q: np.ndarray,
                 R: np.ndarray,
                 M: np.ndarray = None):
        """Initializes the quadratic cost function.

        Args:
            target_function: A callable that takes a float `t` (time) and
                returns a tuple of target state (x_t) and target control (u_t)
                numpy arrays.
            Q: The state-weighting matrix (n_x, n_x). Must be square.
            R: The control-weighting matrix (n_u, n_u). Must be square.
            M: The state-control cross-weighting matrix (n_x, n_u). If None,
                it is initialized to a zero matrix.
        """
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be a square matrix, but has shape {Q.shape}.")
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            raise ValueError(f"R must be a square matrix, but has shape {R.shape}.")

        n_x, n_u = Q.shape[0], R.shape[0]

        if M is None:
            M = np.zeros((n_x, n_u))
        elif M.shape != (n_x, n_u):
            raise ValueError(
                f"M must have shape (n_x, n_u) = ({n_x}, {n_u}), "
                f"but has shape {M.shape}."
            )

        super().__init__(n_x=n_x, n_u=n_u)
        self.Q = Q
        self.R = R
        self.M = M
        self._target_function = target_function
        self._hessian = np.block([[self.Q, self.M], [self.M.T, self.R]])

    def value(self, t: float, x: np.ndarray, u: np.ndarray) -> float:
        """Computes the scalar cost value at a given time, state, and control.

        Args:
            t: The time at which to evaluate the cost.
            x: The state vector, shape (n_x,).
            u: The control vector, shape (n_u,).

        Returns:
            The scalar cost value.
        """
        x_t, u_t = self._target_function(t)
        dx = x - x_t
        du = u - u_t

        cost = 0.5 * dx.T @ self.Q @ dx
        cost += 0.5 * du.T @ self.R @ du
        cost += dx.T @ self.M @ du
        return cost

    def sensitivity(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes the sensitivity (gradient) of the cost.

        The gradient is computed with respect to the concatenated vector [x, u].

        Args:
            t: The time at which to evaluate the sensitivity.
            x: The state vector, shape (n_x,).
            u: The control vector, shape (n_u,).

        Returns:
            The gradient vector, shape (n_x + n_u,).
        """
        x_t, u_t = self._target_function(t)
        dx = x - x_t
        du = u - u_t

        grad_x = self.Q @ dx + self.M @ du
        grad_u = self.R @ du + self.M.T @ dx
        return np.concatenate([grad_x, grad_u])

    def hessian(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Computes the Hessian matrix of the cost.

        The Hessian is computed with respect to the concatenated vector [x, u].
        For a quadratic function, the Hessian is constant.

        Args:
            t: The time (unused, for API consistency).
            x: The state vector (unused, for API consistency).
            u: The control vector (unused, for API consistency).

        Returns:
            The Hessian matrix, shape (n_x + n_u, n_x + n_u).
        """
        return self._hessian
