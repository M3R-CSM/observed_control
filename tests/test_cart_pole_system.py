# tests/test_cart_pole_system.py
import pytest
import autograd.numpy as np
from systems.cart_pole_system import CartPoleSystem

@pytest.fixture
def cart_pole():
    """Provides a default CartPoleSystem instance for testing."""
    return CartPoleSystem()

def test_stable_equilibrium(cart_pole):
    """
    Tests the system at its stable equilibrium (pole hanging down).
    With zero velocity and zero force, all derivatives should be zero.
    """
    # State: cart at origin, zero velocity, pole hanging down (pi rad), zero angular velocity
    x_stable = np.array([0.0, 0.0, np.pi, 0.0])
    u_zero = np.array([0.0])

    derivatives = cart_pole.ode(0, x_stable, u_zero)

    assert np.allclose(derivatives, np.zeros(4)), \
        "Derivatives should be zero at the stable equilibrium."

def test_unstable_equilibrium(cart_pole):
    """
    Tests the system at its unstable equilibrium (pole balanced upright).
    With zero velocity and zero force, all derivatives should be zero.
    """
    # State: cart at origin, zero velocity, pole upright (0 rad), zero angular velocity
    x_unstable = np.array([0.0, 0.0, 0.0, 0.0])
    u_zero = np.array([0.0])

    derivatives = cart_pole.ode(0, x_unstable, u_zero)

    assert np.allclose(derivatives, np.zeros(4)), \
        "Derivatives should be zero at the unstable equilibrium."

def test_force_effect(cart_pole):
    """
    Tests that applying a force to the cart from rest causes acceleration.
    """
    x_rest = np.array([0.0, 0.0, 0.0, 0.0])
    u_force = np.array([10.0]) # Apply a force of 10N

    derivatives = cart_pole.ode(0, x_rest, u_force)

    # The cart velocity derivative (acceleration) should be positive.
    cart_acceleration = derivatives[1]
    assert cart_acceleration > 0, \
        "Applying a positive force should cause positive cart acceleration."
