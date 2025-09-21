# Copyright (c) 2025 Andrew Petruska
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# tests/test_anticipated_condition.py
import pytest
import autograd.numpy as np
from core.anticipated_condition import AnticipatedCondition

def quadratic_cost(t, x, u):
    """A simple quadratic cost function for testing."""
    return np.sum(x**2) + np.sum(u**2)

class PolymorphicCondition(AnticipatedCondition):
    """A subclass that overrides the value method."""
    def __init__(self, n_x, n_u):
        super().__init__(n_x=n_x, n_u=n_u)

    def value(self, t, x, u):
        return np.sum(x**3) + np.sum(u)

@pytest.fixture
def quadratic_condition():
    return AnticipatedCondition(n_x=2, n_u=1, value_func=quadratic_cost)

@pytest.fixture
def polymorphic_condition():
    return PolymorphicCondition(n_x=2, n_u=1)

def test_value_calculation(quadratic_condition):
    """Tests that the value is computed correctly."""
    x = np.array([1.0, 2.0])
    u = np.array([3.0])
    expected_value = (1**2 + 2**2) + 3**2
    assert np.isclose(quadratic_condition.value(0.0, x, u), expected_value)

def test_sensitivity_calculation(quadratic_condition):
    """Tests the sensitivity (gradient) calculation."""
    x = np.array([1.0, 2.0])
    u = np.array([3.0])
    sensitivity = quadratic_condition.sensitivity(0.0, x, u)
    expected_sensitivity = np.array([2*x[0], 2*x[1], 2*u[0]])
    assert np.allclose(sensitivity, expected_sensitivity)

def test_hessian_calculation(quadratic_condition):
    """Tests the Hessian calculation."""
    x = np.array([1.0, 2.0])
    u = np.array([3.0])
    hessian = quadratic_condition.hessian(0.0, x, u)
    expected_hessian = np.diag([2.0, 2.0, 2.0])
    assert np.allclose(hessian, expected_hessian)

def test_polymorphic_condition(polymorphic_condition):
    """Tests a polymorphic condition where the value method is overridden."""
    x = np.array([1.0, 2.0])
    u = np.array([3.0])

    # Test value
    expected_value = (1**3 + 2**3) + 3
    assert np.isclose(polymorphic_condition.value(0.0, x, u), expected_value)

    # Test sensitivity
    expected_sensitivity = np.array([3*x[0]**2, 3*x[1]**2, 1.0])
    assert np.allclose(polymorphic_condition.sensitivity(0.0, x, u), expected_sensitivity)

    # Test Hessian
    expected_hessian_xx = np.diag([6*x[0], 6*x[1]])
    expected_hessian = np.block([
        [expected_hessian_xx, np.zeros((2, 1))],
        [np.zeros((1, 2)), np.zeros((1, 1))]
    ])
    assert np.allclose(polymorphic_condition.hessian(0.0, x, u), expected_hessian)
