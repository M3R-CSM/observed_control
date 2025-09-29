# examples/msd_control_simulation.py
import autograd.numpy as np

import matplotlib
import matplotlib.pyplot as plt

from systems.linear_system import LinearSystem
from core.anticipated_condition import AnticipatedCondition
from core.iterated_observed_control import IteratedObservedControl
from core.observed_control import ObservedControl
from conditions.quadratic_cost import QuadraticCost

matplotlib.use("Qt5Agg")


def main():
    """
    Runs a simulation of Observed Control on a linear mass-spring-damper
    system and plots the results.
    """
    # 1. Define the Dynamic System (Mass-Spring-Damper)
    # ----------------------------------------------------
    # System parameters: mass (m), spring constant (k), damping (c)
    m, k, c = 1.0, 0.0, 0.1
    A = np.array([[0.0, 1.0], [-k / m, -c / m]])
    B = np.array([[0.0], [1.0 / m]])

    # Instantiate the linear system model
    dynamic_system = LinearSystem(A=A, B=B)
    n_x, n_u = dynamic_system.n_x, dynamic_system.n_u

    # 2. Define the Anticipated Condition (Quadratic Cost)
    # ----------------------------------------------------
    # Define the target state [position=1, velocity=0]
    step_time = 6
    x_target = np.array([1.0, 0.0])

    # Define weighting matrices for state error (Q) and control effort (R)
    Q = np.diag([1.0, .001])  # Penalize position error more than velocity
    R = 0.0 * np.diag([1.0])  # Penalize control effort

    def target(t):
        if t > step_time:
            return x_target, np.zeros(1)
        else:
            return np.zeros(2), np.zeros(1)

    anticipated_condition = QuadraticCost(target_function=target, Q=Q, R=R)

    # #### autodifferentiation version
    # def quadratic_cost(t, x, u):
    #     """Calculates the quadratic cost."""
    #     x_t, u_target = target(t)
    #     error = x - x_t
    #
    #     state_cost = (error.T @ Q @ error)
    #     control_cost = (u_target - u).T @ R @ (u_target - u)
    #
    #     return state_cost + control_cost
    #
    # # Instantiate the cost function object
    # anticipated_condition = AnticipatedCondition(
    #     n_x=n_x, n_u=n_u, value_func=quadratic_cost
    # )

    # 3. Configure and Instantiate the Observed Controller
    # ----------------------------------------------------
    # oc_controller = IteratedObservedControl(
    oc_controller = ObservedControl(
        dynamic_system=dynamic_system,
        anticipated_conditions=[(1.0, anticipated_condition)],
        expected_update_period=0.1,
        min_horizon=10,
        max_horizon=150,
        adaptive_tolerances_trace_p=1e-5,
        adaptive_tolerances_gamma=1e-4,
        delta_control_penalty=1 * np.eye(n_u),
    )

    # 4. Set up and Run the Simulation
    # ---------------------------------
    sim_time = step_time + 4  # Total simulation time
    dt = oc_controller.expected_update_period
    num_steps = int(sim_time / dt)

    # Initial state [position=0, velocity=0]
    x_current = np.array([0.0, 0.0])

    # Store history for plotting
    time_history = [0.0]
    state_history = [x_current]
    control_history = []

    current_control = np.zeros(n_u)

    print("Running simulation...")
    sim_time = 0
    for i in range(num_steps):
        # Compute the optimal control action
        # The initial control guess is always 0 for this example
        optimal_control, info = oc_controller.control_law(sim_time, x_current, current_control)
        print(f"sim time: {i * dt:.3f}", f"\tcompute_time: {info['compute_time'] * 1000:.1f}ms",
              f"\thorizon_len: {info['final_horizon']:d}")

        # Apply the control to the system to get the next state
        x_next, _, _ = dynamic_system.solve(
            t_init=sim_time, t_final=sim_time + dt, x_init=x_current, u=optimal_control
        )
        # print("x_next: ", x_next.T)

        # Store results
        control_history.append(optimal_control)
        state_history.append(x_next)
        time_history.append((i + 1) * dt)

        # Update the state for the next iteration
        x_current = x_next
        current_control = optimal_control
        sim_time += dt
    print("Simulation complete.")

    # 5. Plot the Results
    # -------------------
    # Convert history lists to numpy arrays for easier indexing
    state_history = np.array(state_history)
    control_history = np.array(control_history)
    time_history = np.array(time_history)

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Observed Control of a Mass-Spring-Damper System", fontsize=16)

    # Plot state trajectories
    axs[0].plot(time_history, state_history[:, 0], label="Position (x)", lw=2)
    axs[0].plot(time_history, state_history[:, 1], label="Velocity (ẋ)", lw=2)
    y = [target(t)[0][0] for t in time_history]
    axs[0].plot(time_history, y, color='r', linestyle='--', label="Target Position")
    axs[0].set_ylabel("State")
    axs[0].legend()
    axs[0].grid(True)

    # Plot control input
    axs[1].plot(time_history[:-1], control_history, label="Control Force (u)", color='g', lw=2)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Control Input")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
