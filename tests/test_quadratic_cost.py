# tests/test_quadratic_cost.py
"""Tests for the QuadraticCost class."""

import numpy as standard_np  # For testing functions
import autograd.numpy as np  # For autograd compatibility
import pytest
from autograd import jacobian, hessian

from conditions.quadratic_cost import QuadraticCost


@pytest.fixture
def cost_components():
    """Provides standard components for creating a QuadraticCost instance."""
    n_x, n_u = 3, 2
    Q = np.diag([1.0, 2.0, 3.0])
    R = np.diag([0.5, 0.5])
    M = np.array([[0.1, 0.0],
                  [0.0, 0.2],
                  [0.0, 0.0]])
    target_function = lambda t: (np.zeros(n_x), np.zeros(n_u))
    return {
        "n_x": n_x, "n_u": n_u, "Q": Q, "R": R, "M": M,
        "target_function": target_function,
    }


class TestQuadraticCost:
    """Test suite for the QuadraticCost class."""

    def test_initialization(self, cost_components):
        """Tests successful initialization and default M matrix."""
        # Test with explicit M
        cost = QuadraticCost(
            target_function=cost_components["target_function"],
            Q=cost_components["Q"],
            R=cost_components["R"],
            M=cost_components["M"],
        )
        assert cost.n_x == cost_components["n_x"]
        assert cost.n_u == cost_components["n_u"]
        standard_np.testing.assert_array_equal(cost.M, cost_components["M"])

        # Test with default M (should be zeros)
        cost_default_m = QuadraticCost(
            target_function=cost_components["target_function"],
            Q=cost_components["Q"],
            R=cost_components["R"],
        )
        expected_M = np.zeros((cost_components["n_x"], cost_components["n_u"]))
        standard_np.testing.assert_array_equal(cost_default_m.M, expected_M)

    def test_initialization_invalid_dims(self):
        """Tests that initialization fails with incompatible matrix dimensions."""
        Q_good = np.eye(3)
        R_good = np.eye(2)
        target_func = lambda t: (np.zeros(3), np.zeros(2))

        # Non-square Q
        with pytest.raises(ValueError, match="Q must be a square matrix"):
            QuadraticCost(target_func, Q=np.random.rand(3, 2), R=R_good)

        # Non-square R
        with pytest.raises(ValueError, match="R must be a square matrix"):
            QuadraticCost(target_func, Q=Q_good, R=np.random.rand(2, 3))

        # Mismatched M
        with pytest.raises(ValueError, match="M must have shape"):
            QuadraticCost(target_func, Q=Q_good, R=R_good, M=np.random.rand(3, 1))

    def test_value(self, cost_components):
        """Tests the cost value computation."""
        cost = QuadraticCost(**{k: v for k, v in cost_components.items() if k not in ['n_x', 'n_u']})

        t = 0.0
        x = np.array([1.0, 1.0, 1.0])
        u = np.array([1.0, 2.0])

        expected_value = 4.75
        actual_value = cost.value(t, x, u)
        assert standard_np.isclose(actual_value, expected_value)

    def test_sensitivity_against_autograd(self, cost_components):
        """Tests the analytical sensitivity against autograd's jacobian."""
        cost = QuadraticCost(**{k: v for k, v in cost_components.items() if k not in ['n_x', 'n_u']})

        t = 1.0
        x = np.random.rand(cost.n_x)
        u = np.random.rand(cost.n_u)

        def value_wrapper(z):
            x_w, u_w = z[:cost.n_x], z[cost.n_x:]
            return cost.value(t, x_w, u_w)

        z = np.concatenate([x, u])
        numerical_sensitivity = jacobian(value_wrapper)(z)
        analytical_sensitivity = cost.sensitivity(t, x, u)

        standard_np.testing.assert_allclose(
            analytical_sensitivity,
            numerical_sensitivity,
            rtol=1e-7,
            err_msg="Analytical sensitivity does not match numerical gradient.",
        )

    def test_hessian_against_autograd(self, cost_components):
        """Tests the analytical Hessian against autograd's hessian."""
        cost = QuadraticCost(**{k: v for k, v in cost_components.items() if k not in ['n_x', 'n_u']})

        t = 1.0
        x = np.random.rand(cost.n_x)
        u = np.random.rand(cost.n_u)

        def value_wrapper(z):
            x_w, u_w = z[:cost.n_x], z[cost.n_x:]
            return cost.value(t, x_w, u_w)

        z = np.concatenate([x, u])
        numerical_hessian = hessian(value_wrapper)(z)
        analytical_hessian = cost.hessian(t, x, u)

        standard_np.testing.assert_allclose(
            analytical_hessian,
            numerical_hessian,
            rtol=1e-7,
            err_msg="Analytical Hessian does not match numerical Hessian.",
        )
