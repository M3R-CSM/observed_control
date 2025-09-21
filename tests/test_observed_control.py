# Copyright (c) 2025 Andrew Petruska
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# tests/test_observed_control.py
import pytest
import autograd.numpy as np
from core.dynamic_system import DynamicSystem
from core.anticipated_condition import AnticipatedCondition
from core.observed_control import ObservedControl

def test_lqr_convergence():
    """
    Tests that for a linear system and quadratic cost, the control converges
    towards the LQR solution. In this simple case, we expect the control to
    drive the state towards the origin.
    """
    # A simple damped integrator system: dx/dt = -0.1*x + u
    def linear_ode(t, x, u):
        return -0.1 * x + u

    # A simple quadratic cost: x^2 + u^2
    def quadratic_cost(t, x, u):
        return x[0]**2 + u[0]**2

    dynamic_system = DynamicSystem(n_x=1, n_u=1, ode_func=linear_ode)
    anticipated_condition = AnticipatedCondition(n_x=1, n_u=1, value_func=quadratic_cost)

    oc = ObservedControl(
        dynamic_system=dynamic_system,
        anticipated_conditions=[(1.0, anticipated_condition)],
        expected_update_period=0.1,
        min_horizon=5,
        max_horizon=20,
        adaptive_tolerances_trace_p=1e-3,
        adaptive_tolerances_gamma=1e-3,
        delta_control_penalty=100 * np.eye(1),
    )

    initial_state = np.array([10.0])
    initial_control = np.array([0.0])
    start_time = 0.0 # Define a start time for the controller

    # The optimal control for this system should be negative to drive the
    # state from 10.0 towards 0.0.
    optimal_control, _ = oc.control_law(start_time, initial_state, initial_control)

    # Assertions
    assert optimal_control.shape == (1,)
    assert optimal_control[0] < 0, "Control should be negative to drive state to the origin."

    # Simulate one step to see if the state is reduced
    next_state, _, _ = dynamic_system.solve(0.0, 0.1, initial_state, optimal_control)
    assert next_state[0] < initial_state[0]
