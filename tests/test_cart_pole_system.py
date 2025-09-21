# tests/test_cart_pole_system.py
import pytest
import numpy as np
from systems.cart_pole_system import CartPoleSystem

@pytest.fixture
def cart_pole():
    """Provides a default CartPoleSystem instance with damping."""
    # Using zero damping for equilibrium tests to ensure perfect zero derivatives.
    return CartPoleSystem(dv=0.0, dw=0.0)

@pytest.fixture
def cart_pole_with_damping():
    """Provides a CartPoleSystem instance with non-zero damping."""
    return CartPoleSystem(dv=0.1, dw=0.05)

def test_equilibrium_upright(cart_pole):
    """Tests the unstable equilibrium (pole upright) at rest."""
    x_upright = np.array([0.0, 0.0, 0.0, 0.0])
    u_zero = np.array([0.0])
    derivatives = cart_pole.ode(0, x_upright, u_zero)
    assert np.allclose(derivatives, 0.0), \
        "Derivatives should be zero at the upright equilibrium."

def test_equilibrium_hanging_down(cart_pole):
    """Tests the stable equilibrium (pole hanging) at rest."""
    x_hanging = np.array([0.0, 0.0, np.pi, 0.0])
    u_zero = np.array([0.0])
    derivatives = cart_pole.ode(0, x_hanging, u_zero)
    assert np.allclose(derivatives, 0.0), \
        "Derivatives should be zero at the hanging equilibrium."

def test_force_effect(cart_pole):
    """Tests that a force causes acceleration from rest."""
    x_rest = np.array([0.0, 0.0, 0.0, 0.0])
    u_force = np.array([10.0])
    derivatives = cart_pole.ode(0, x_rest, u_force)
    assert derivatives[1] > 0, "Positive force should cause positive cart acceleration."

def test_analytical_jacobians(cart_pole_with_damping):
    """
    Compares the analytical Jacobians against a finite difference approximation.
    This is the most critical test for this module.
    """
    system = cart_pole_with_damping
    t = 0.0
    x = np.array([0.1, 0.2, 0.3, 0.4]) # A non-equilibrium state
    u = np.array([5.0])
    epsilon = 1e-7

    # Get analytical Jacobians from the system
    A_analytical, B_analytical = system._get_jacobians(t, x, u)

    # Approximate State Jacobian A = df/dx
    A_approx = np.zeros_like(A_analytical)
    for i in range(system.n_x):
        x_plus = x.copy(); x_plus[i] += epsilon
        x_minus = x.copy(); x_minus[i] -= epsilon
        f_plus = system.ode(t, x_plus, u)
        f_minus = system.ode(t, x_minus, u)
        A_approx[:, i] = (f_plus - f_minus) / (2 * epsilon)

    # Approximate Control Jacobian B = df/du
    B_approx = np.zeros_like(B_analytical)
    for i in range(system.n_u):
        u_plus = u.copy(); u_plus[i] += epsilon
        u_minus = u.copy(); u_minus[i] -= epsilon
        f_plus = system.ode(t, x, u_plus)
        f_minus = system.ode(t, x, u_minus)
        B_approx[:, i] = (f_plus - f_minus) / (2 * epsilon)

    assert np.allclose(A_analytical, A_approx, atol=1e-6), \
        "Analytical state Jacobian (A) does not match finite differences."
    assert np.allclose(B_analytical, B_approx, atol=1e-6), \
        "Analytical control Jacobian (B) does not match finite differences."
