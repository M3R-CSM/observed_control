# Observed Control for NMPC

This project is a Python implementation of the **Observed Control** algorithm for Nonlinear Model Predictive Control (NMPC), as described in the paper *"Observed Control: Linearly Scalable Nonlinear Model Predictive Control with Adaptive Horizons"* (Hamzezadeh & Petruska, arXiv:2508.13339v1).

The core of the project is a flexible and extensible framework for defining dynamic systems and anticipated future conditions (costs), using `autograd` for automatic differentiation of all necessary Jacobians and Hessians.

-----

## Features 

  * **Efficient NMPC Solver**: Implements Algorithm 3 ("Efficient Observed Control") from the reference paper, providing a computationally efficient NMPC solution with linear scalability.
  * **Adaptive Horizon**: The controller dynamically adjusts the prediction horizon based on convergence criteria.
  * **Extensible Framework**: Uses abstract base classes for `DynamicSystem` and `AnticipatedCondition`, allowing you to easily define custom models and cost functions.
  * **Automatic Differentiation**: Leverages **`autograd`** to automatically compute the complex gradients and Hessians required by the algorithm, making it easy to use with new models.
  * **Specialized Solvers**: Includes optimized analytical solvers for linear time-invariant (LTI) systems to improve speed and accuracy where applicable.
  * **Robust Testing**: Includes a test suite using **`pytest`** to ensure the correctness of the implementation.

-----

## Project Structure

The repository is organized into several key directories:

  * `core/`: Contains the central logic of the Observed Control algorithm (`observed_control.py`) and the abstract base classes for dynamic systems (`dynamic_system.py`) and cost functions (`anticipated_condition.py`).
  * `systems/`: Provides concrete implementations of dynamic systems, such as a general `LinearSystem` and a nonlinear `CartPoleSystem`.
  * `conditions/`: Contains concrete implementations of cost functions, such as the common `QuadraticCost`.
  * `examples/`: Includes standalone scripts that demonstrate how to use the controller for different tasks.

-----

## Installation 

To get started, clone the repository and install the necessary dependencies in a virtual environment.

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd observed_control

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 3. Install the project in editable mode with development tools
pip install -e .[dev]
```

-----

## Running the Examples 

The `examples/` directory contains scripts that demonstrate the controller's functionality.

### Mass-Spring-Damper System

This example demonstrates the controller driving a simple linear mass-spring-damper system to a target state of `position=1.0`.

**To run:**

```bash
python examples/msd_control_simulation.py
```

This will generate a plot showing the system's state (position and velocity) and the control input over time as it reaches the target.

### Cart-Pole Swing-Up

This more complex example showcases the NMPC controller performing a swing-up maneuver for an underactuated cart-pole system—a classic nonlinear control problem. The goal is to swing the pole from a downward-hanging position to an upright, balanced position.

**To run:**

```bash
python examples/cart_pole_swing_up.py
```

This script will produce both an **animation** of the cart-pole swing-up and a **plot** of the state and control histories.

-----

## Development

### Running Tests 

To run the entire test suite, execute the following command from the project root:

```bash
pytest
```

-----

## Reference Paper

This implementation is based on the following work. Please refer to it for a detailed theoretical background on the algorithm.

> Hamzezadeh, E. T., & Petruska, A. J. (2025). *Observed Control: Linearly Scalable Nonlinear Model Predictive Control with Adaptive Horizons*. arXiv preprint arXiv:2508.13339.
