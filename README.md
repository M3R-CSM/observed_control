# Observed Control for NMPC

This project is a Python implementation of the **Observed Control** algorithm for Nonlinear Model Predictive Control (NMPC), as described in the paper *"Observed Control Linearly Scalable Nonlinear Model Predictive Control with Adaptive Horizons"* (Hamzezadeh & Petruska, arXiv:2508.13339v1).

The core of the project is a flexible and extensible framework for defining dynamic systems and anticipated future conditions (costs), using `autograd` for automatic differentiation of all necessary Jacobians and Hessians.

---

## Features ✨

* **Efficient NMPC Solver**: Implements Algorithm 3 ("Efficient Observed Control") from the reference paper, providing a computationally efficient NMPC solution with linear scalability.
* **Adaptive Horizon**: The controller dynamically adjusts the prediction horizon based on convergence criteria.
* **Extensible Framework**: Uses abstract base classes for `DynamicSystem` and `AnticipatedCondition`, allowing you to easily define custom models and cost functions either by subclassing or by passing functions directly.
* **Automatic Differentiation**: Leverages **`autograd`** to automatically compute the complex gradients and Hessians required by the algorithm, making it easy to use with new models.
* **Robust Testing**: Includes a test suite using **`pytest`** to ensure the correctness of the implementation and verify key behaviors, such as convergence to LQR solutions for linear systems.

---

## Installation 🚀

To get started, clone the repository and install the package in an editable mode along with its development dependencies. This project requires Python 3.8+.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/observed_control.git
cd observed_control

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 3. Install the project in editable mode with development tools
pip install -e .[dev]
```


## Usage Example

Here is a complete example of setting up and running the `ObservedControl` solver for a simple damped integrator system.

```python
# main.py
import autograd.numpy as np
from core.dynamic_system import DynamicSystem
from core.anticipated_condition import AnticipatedCondition
from core.observed_control import ObservedControl

# 1. Define the dynamic system
# A simple damped integrator: dx/dt = -0.1*x + u
def linear_ode(t, x, u):
    return -0.1 * x + u

dynamic_system = DynamicSystem(n_x=1, n_u=1, ode_func=linear_ode)

# 2. Define the anticipated condition (cost function)
# A quadratic cost to drive the state and control to zero: x^2 + u^2
def quadratic_cost(t, x, u):
    return x[0]**2 + u[0]**2

anticipated_condition = AnticipatedCondition(n_x=1, n_u=1

```

## Development

### Running Tests 🧪

To run the entire test suite, execute the following command from the project root:

```bash
pytest
```

## Reference Paper

This implementation is based on the following work. Please refer to it for a detailed theoretical background on the algorithm.

> Hamzezadeh, E. T., & Petruska, A. J. (2025). *Observed Control Linearly Scalable Nonlinear Model Predictive Control with Adaptive Horizons*. arXiv preprint arXiv:2508.13339.
