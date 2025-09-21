# Copyright (c) 2025 Andrew Petruska
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# tests/test_linear_system.py
import pytest
import autograd.numpy as np
from core.dynamic_system import DynamicSystem
from systems.linear_system import LinearSystem
import systems.linear_system as linear_system_module

@pytest.fixture
def msd_matrices():
    """Provides the A and B matrices for a mass-spring-damper system."""
    m, k, c = 1.0, 2.0, 0.5
    A = np.array([[0, 1.0], [-k / m, -c / m]])
    B = np.array([[0], [1.0 / m]])
    return A, B

def test_linear_solver_against_numerical_solver(msd_matrices):
    """
    Verifies the analytical solver is consistent with the numerical solver.
    """
    A, B = msd_matrices
    linear_system = LinearSystem(A=A, B=B)
    numerical_system = DynamicSystem(n_x=2, n_u=1, ode_func=lambda t, x, u: A @ x + B @ u)

    t_init, t_final = 0.0, 0.5
    x_init = np.array([1.0, 0.5])
    u = np.array([0.5])

    x_final_analytical, _, _ = linear_system.solve(t_init, t_final, x_init, u)
    x_final_numerical, _, _ = numerical_system.solve(t_init, t_final, x_init, u)

    assert np.allclose(x_final_analytical, x_final_numerical, atol=1e-9)

def test_linear_system_caching_logic(msd_matrices, mocker):
    """
    Verifies the caching mechanism by mocking `scipy.linalg.expm`.
    """
    A, B = msd_matrices
    linear_system = LinearSystem(A=A, B=B)

    # Mock `expm` where it is imported and used: in the linear_system module
    spy_expm = mocker.spy(linear_system_module, 'expm')

    x_init = np.array([1.0, 0.0])
    u = np.array([0.1])

    # First call with dt = 0.5. `expm` should be called.
    linear_system.solve(t_init=0.0, t_final=0.5, x_init=x_init, u=u)
    assert spy_expm.call_count == 1

    # Second call with the same dt = 0.5. Cache should be used.
    linear_system.solve(t_init=1.0, t_final=1.5, x_init=x_init, u=u)
    assert spy_expm.call_count == 1

    # Third call with a new dt = 0.1. `expm` must be called.
    linear_system.solve(t_init=0.0, t_final=0.1, x_init=x_init, u=u)
    assert spy_expm.call_count == 2
