# tests/test_dynamic_system.py
import pytest
import autograd.numpy as np
from core.dynamic_system import DynamicSystem

@pytest.fixture
def linear_system():
    """Fixture for a simple linear system."""
    def linear_ode(t, x, u):
        A = np.array([[-0.1, 1.0], [0.0, -0.2]])
        B = np.array([[0.0], [0.5]])
        return A @ x + B @ u
    return DynamicSystem(n_x=2, n_u=1, ode_func=linear_ode)

@pytest.fixture
def nonlinear_system():
    """Fixture for a nonlinear inverted pendulum system."""
    class InvertedPendulum(DynamicSystem):
        def __init__(self):
            super().__init__(n_x=2, n_u=1)
            self.g = 9.81
            self.l = 1.0
            self.m = 1.0

        def ode(self, t, x, u):
            theta, omega = x
            torque = u[0]
            dtheta_dt = omega
            domega_dt = (self.g / self.l) * np.sin(theta) + (torque / (self.m * self.l**2))
            return np.array([dtheta_dt, domega_dt])
    return InvertedPendulum()

def test_linear_system_integration(linear_system):
    """Tests the integration of the linear system."""
    x_init = np.array([1.0, 1.0])
    u = np.array([0.5])
    x_final, _, _ = linear_system.solve(0.0, 0.1, x_init, u)
    assert x_final.shape == (2,)
    # A simple check to see if the state has changed
    assert not np.allclose(x_final, x_init)

def test_nonlinear_system_integration(nonlinear_system):
    """Tests the integration of the nonlinear system."""
    x_init = np.array([np.pi / 4, 0.0])
    u = np.array([0.0])
    x_final, _, _ = nonlinear_system.solve(0.0, 0.1, x_init, u)
    assert x_final.shape == (2,)
    # For a pendulum starting from rest, velocity should increase
    assert x_final[1] > x_init[1]

def test_sensitivities_finite_difference(linear_system):
    """Tests the computed sensitivities against a finite difference approximation."""
    t_init, t_final = 0.0, 0.1
    x_init = np.array([1.0, 0.5])
    u = np.array([0.1])
    epsilon = 1e-6

    _, phi_x, phi_u = linear_system.solve(t_init, t_final, x_init, u)

    # Test Phi_x
    phi_x_approx = np.zeros_like(phi_x)
    for i in range(linear_system.n_x):
        x_perturbed = x_init.copy()
        x_perturbed[i] += epsilon
        x_final_perturbed, _, _ = linear_system.solve(t_init, t_final, x_perturbed, u)
        x_final_base, _, _ = linear_system.solve(t_init, t_final, x_init, u)
        phi_x_approx[:, i] = (x_final_perturbed - x_final_base) / epsilon
    assert np.allclose(phi_x, phi_x_approx, atol=1e-5)

    # Test Phi_u
    phi_u_approx = np.zeros_like(phi_u)
    for i in range(linear_system.n_u):
        u_perturbed = u.copy()
        u_perturbed[i] += epsilon
        x_final_perturbed, _, _ = linear_system.solve(t_init, t_final, x_init, u_perturbed)
        x_final_base, _, _ = linear_system.solve(t_init, t_final, x_init, u)
        phi_u_approx[:, i] = (x_final_perturbed - x_final_base) / epsilon
    assert np.allclose(phi_u, phi_u_approx, atol=1e-5)
