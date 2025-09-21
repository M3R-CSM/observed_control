# src/systems/cart_pole_system.py
import numpy as np
from numba import jit
from core.dynamic_system import DynamicSystemBase
from typing import Tuple


# ----------------------------------------------------------------------------
# JIT-compiled Standalone Functions
# ----------------------------------------------------------------------------

@jit(nopython=True)
def _cart_pole_ode_jit(t, x, u, mc, mp, l, g, dv, dw):
    """JIT-compiled ODE function for the cart-pole system with damping."""
    _, v, theta, omega = x
    Fx = u[0]

    s_theta = np.sin(theta)
    c_theta = np.cos(theta)

    # Denominator term
    den = l * (mp * s_theta ** 2 + mc)

    # Equation for cart acceleration (x_ddot)
    x_ddot_num = (l * Fx - dv * l * v - mp * l ** 2 * omega ** 2 * s_theta -
                  (dw * omega - g * mp * l * s_theta) * c_theta)
    x_ddot = x_ddot_num / den

    # Equation for pole angular acceleration (theta_ddot)
    theta_ddot_num = ((mp * l * Fx - dv * mp * l * v) * c_theta -
                      dw * (mc + mp) * omega +
                      (-mp ** 2 * l ** 2 * omega ** 2 * c_theta +
                       g * mp ** 2 * l + g * mc * mp * l) * s_theta)
    theta_ddot = theta_ddot_num / (mp * l * den)

    return np.array([v, x_ddot, omega, theta_ddot])


@jit(nopython=True)
def _cart_pole_jacobians_jit(t, x, u, mc, mp, l, g, dv, dw):
    """
    JIT-compiled function for the analytical Jacobians (A and B),
    updated to match the verified C++ implementation.
    """
    _, v, theta, omega = x
    Fx = u[0]

    # Pre-compute trigonometric and other common terms
    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    s_2theta = np.sin(2 * theta)
    c_2theta = np.cos(2 * theta)
    s_theta_sq = s_theta ** 2
    c_theta_sq = c_theta ** 2
    omega_sq = omega ** 2
    l_sq = l ** 2
    mp_sq = mp ** 2

    # Equivalent mass term from C++ implementation
    m_eq = mp * s_theta_sq + mc
    m_eq_sq = m_eq ** 2

    # --- State Jacobian A = df/dx ---
    A = np.zeros((4, 4))

    # d(x_dot)/d(v)
    A[0, 1] = 1.0
    A[2, 3] = 1.0

    # d(v_dot)/d(v)
    A[1, 1] = -dv / m_eq
    # d(omega_dot)/d(v)
    A[3, 1] = -(dv * c_theta) / (l * m_eq)

    # d(v_dot)/d(theta)
    term1_v_theta = (dw * s_theta * omega - g * mp * l - mp * l_sq * c_theta * omega_sq +
                     2 * g * mp * l * c_theta_sq) / (l * m_eq)
    term2_v_theta = (2 * mp * c_theta * s_theta *
                     (dv * l * v - l * Fx + dw * c_theta * omega +
                      mp * l_sq * s_theta * omega_sq - g * mp * l * c_theta * s_theta)) / (l * m_eq_sq)
    A[1, 2] = term1_v_theta + term2_v_theta

    # d(omega_dot)/d(theta)
    # Note: The C++ denominator `(mass_cart + mass_pole / 2 - (mass_pole * c_2th) / 2)` simplifies to `m_eq`
    term1_omega_theta = (
                                dv * s_theta * v - Fx * s_theta + g * mc * c_theta + g * mp * c_theta - mp * l * c_2theta * omega_sq) / (
                                l * m_eq)
    term2_omega_theta_num = (4 * s_2theta *
                             (dw * mc * omega + dw * mp * omega + (mp_sq * l_sq * s_2theta * omega_sq) / 2 -
                              g * mp_sq * l * s_theta - mp * l * Fx * c_theta - g * mc * mp * l * s_theta +
                              dv * mp * l * c_theta * v))
    term2_omega_theta_den = (l_sq * (2 * mc + mp - mp * c_2theta) ** 2)
    A[3, 2] = term1_omega_theta + term2_omega_theta_num / term2_omega_theta_den

    # d(v_dot)/d(omega)
    A[1, 3] = -(2 * mp * s_theta * omega * l_sq + dw * c_theta) / (l * m_eq)
    # d(omega_dot)/d(omega)
    A[3, 3] = -(s_2theta * omega * mp_sq * l_sq + dw * (mp + mc)) / (mp * l_sq * m_eq)

    # --- Control Jacobian B = df/du ---
    B = np.zeros((4, 1))

    # d(v_dot)/d(Fx)
    B[1, 0] = 1 / m_eq
    # d(omega_dot)/d(Fx)
    B[3, 0] = c_theta / (l * m_eq)

    return A, B


# ----------------------------------------------------------------------------
# Class Definition
# ----------------------------------------------------------------------------


class CartPoleSystem(DynamicSystemBase):
    """
    Represents the nonlinear cart-pole dynamic system with viscous damping.
    """

    def __init__(self, mc=0.25, mp=0.20, l=0.45, g=9.81, dv=0.05, dw=0.015):
        """
        Initializes the cart-pole system.

        Args:
            mc (float): Mass of the cart.
            mp (float): Mass of the pole.
            l (float): Half-length of the pole.
            g (float): Acceleration due to gravity.
            dv (float): Damping coefficient for cart velocity.
            dw (float): Damping coefficient for pole angular velocity.
        """
        super().__init__(n_x=4, n_u=1)
        self.mc = mc
        self.mp = mp
        self.l = l
        self.g = g
        self.dv = dv
        self.dw = dw

    def ode(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Wrapper for the JIT-compiled ODE function."""
        return _cart_pole_ode_jit(t, x, u, self.mc, self.mp, self.l,
                                  self.g, self.dv, self.dw)

    def _get_jacobians(self, t: float, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Wrapper for the JIT-compiled Jacobian function."""
        return _cart_pole_jacobians_jit(t, x, u, self.mc, self.mp, self.l,
                                        self.g, self.dv, self.dw)

    def origin(self):
        return np.array([0.0, 0.0, np.pi, 0.0])

    def eigenvalues(self):
        A, _ = _cart_pole_jacobians_jit(0, self.origin(), np.zeros(1), self.mc, self.mp, self.l, self.g, self.dv,
                                        self.dw)
        e, l = np.linalg.eig(A)
        return e
