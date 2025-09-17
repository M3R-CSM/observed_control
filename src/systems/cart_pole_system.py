# src/systems/cart_pole_system.py
import autograd.numpy as np
from core.dynamic_system import DynamicSystemBase

class CartPoleSystem(DynamicSystemBase):
    """Represents the classic nonlinear cart-pole dynamic system.

    The state of the system is defined as:
    x = [x_cart, v_cart, theta_pole, omega_pole]
    where:
        - x_cart: Position of the cart on the track.
        - v_cart: Velocity of the cart.
        - theta_pole: Angle of the pole from the vertical (0 is upright).
        - omega_pole: Angular velocity of the pole.

    The control input is a single force applied to the cart.
    """

    def __init__(self, mc=1.0, mp=0.2, l=0.5, g=9.81):
        """Initializes the cart-pole system with physical parameters.

        Args:
            mc (float): Mass of the cart.
            mp (float): Mass of the pole.
            l (float): Half-length of the pole.
            g (float): Acceleration due to gravity.
        """
        super().__init__(n_x=4, n_u=1)
        self.mc = mc
        self.mp = mp
        self.l = l
        self.g = g
        self.total_mass = self.mc + self.mp

    def ode(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """The ordinary differential equation for the cart-pole system.

        Args:
            t: The current time (unused for this time-invariant system).
            x: The state vector [x_cart, v_cart, theta_pole, omega_pole].
            u: The control vector [force].

        Returns:
            The derivative of the state vector.
        """
        _, _, theta, omega = x
        force = u[0]

        # Pre-calculate trigonometric values for efficiency
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Equations of motion (derived from Lagrangian mechanics)
        temp = (force + self.mp * self.l * omega**2 * sin_theta) / self.total_mass

        alpha = (self.g * sin_theta - cos_theta * temp) / \
                (self.l * (4.0/3.0 - self.mp * cos_theta**2 / self.total_mass))

        a = temp - self.mp * self.l * alpha * cos_theta / self.total_mass

        # State derivatives
        d_x_cart_dt = x[1]
        d_v_cart_dt = a
        d_theta_pole_dt = omega
        d_omega_pole_dt = alpha

        return np.array([d_x_cart_dt, d_v_cart_dt, d_theta_pole_dt, d_omega_pole_dt])
