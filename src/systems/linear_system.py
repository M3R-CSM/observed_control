# Copyright (c) 2025 Andrew Petruska
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# core/linear_system.py
from typing import Tuple

import autograd.numpy as np
from scipy.linalg import expm

from core.dynamic_system import DynamicSystemBase

class LinearSystem(DynamicSystemBase):
    """Represents a linear time-invariant (LTI) dynamic system.

    This class specializes the DynamicSystemBase for LTI systems of the form:
    x_dot = A * x + B * u.

    It overrides the `solve` method to use the closed-form analytical solution
    based on the matrix exponential, which is more accurate and computationally
    efficient than general-purpose numerical integration for LTI systems.

    This implementation includes caching for the discretized matrices. If `solve`
    is called repeatedly with the same time step `dt`, the expensive matrix
    exponential is not recomputed.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray):
        """Initializes the linear system.

        Args:
            A: The state matrix.
            B: The control matrix.
        """
        n_x, n_u = A.shape[0], B.shape[1]
        if A.shape[0] != A.shape[1] or A.shape[0] != B.shape[0]:
            raise ValueError("Incompatible dimensions for A and B matrices.")

        super().__init__(n_x=n_x, n_u=n_u)
        self.A = A
        self.B = B

        # Caching attributes
        self._last_dt = None
        self._cached_phi_x = None
        self._cached_phi_u = None

    def ode(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """The ordinary differential equation for the LTI system.

        Args:
            t: The current time (unused for time-invariant systems).
            x: The state vector.
            u: The control vector.

        Returns:
            The derivative of the state vector (A*x + B*u).
        """
        return self.A @ x + self.B @ u

    def _get_jacobians(self, t: float, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.A, self.B


    def solve(self, t_init: float, t_final: float, x_init: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solves the LTI system dynamics using the matrix exponential.

        This method provides an exact analytical solution for the final state
        and the state and control sensitivities (Phi_x, Phi_u) over the
        time interval. It caches the discretized matrices to avoid re-computation
        if the time step `dt` remains constant between calls.

        Args:
            t_init: The initial time.
            t_final: The final time.
            x_init: The initial state vector.
            u: The control vector, assumed constant over the interval.

        Returns:
            A tuple containing:
                - The final state vector (x_final).
                - The state sensitivity matrix (Phi_x, also known as Ad).
                - The control sensitivity matrix (Phi_u, also known as Bd).
        """
        dt = t_final - t_init

        # Check if we can use the cached matrices using a tolerance-based comparison.
        if self._last_dt is not None and np.isclose(dt, self._last_dt):
            phi_x = self._cached_phi_x
            phi_u = self._cached_phi_u
        else:
            # Create the augmented matrix [A, B; 0, 0] for discretization
            augmented_matrix = np.block([
                [self.A, self.B],
                [np.zeros((self.n_u, self.n_x)), np.zeros((self.n_u, self.n_u))]
            ])

            # Compute the matrix exponential to get the discretized system matrices
            discretized_matrix = expm(augmented_matrix * dt)

            # Extract and cache the sensitivity matrices
            phi_x = discretized_matrix[:self.n_x, :self.n_x]
            phi_u = discretized_matrix[:self.n_x, self.n_x:]

            self._last_dt = dt
            self._cached_phi_x = phi_x
            self._cached_phi_u = phi_u

        # Calculate the final state using the discretized system
        x_final = phi_x @ x_init + phi_u @ u

        return x_final, phi_x, phi_u
