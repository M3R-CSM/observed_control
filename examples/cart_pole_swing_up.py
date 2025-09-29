# Copyright (c) 2025 Andrew Petruska
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# examples/cart_pole_swing_up.py
import numpy as np

# Adjust the import path to match your project structure
from systems.cart_pole_system import CartPoleSystem
from core.observed_control import ObservedControl
from conditions.quadratic_cost import QuadraticCost
from cart_pole_plotting import *


def main():
    """
    Runs a swing-up and stabilization simulation for the Cart-Pole system
    and creates an animation of the result.
    """
    # 1. Define the Dynamic System
    # ---------------------------
    dynamic_system = CartPoleSystem(mc=0.25, mp=0.20, l=0.45, dv=0.05, dw=0.015)
    n_x, n_u = dynamic_system.n_x, dynamic_system.n_u

    # 2. Define the Anticipated Condition (Cost Function)
    # ---------------------------------------------------
    x_target = np.array([0.0, 0.0, 0.0, 0.0])
    Q = np.diag([1000, 1, 300.0, 25])
    R = np.diag([20])
    M = np.zeros((n_x, n_u))

    def target(t):
        if t >= 2.5:
            return x_target, np.zeros(n_u)
        else:
            return np.array([0.0, 0.0, np.pi, 0.0]), np.zeros(n_u)

    anticipated_condition = QuadraticCost(target, Q, R, M)

    # 3. Configure and Instantiate the Observed Controller
    # ----------------------------------------------------
    # In the original file, there was a delta_control_penalty argument that
    # is not in the ObservedControl __init__ signature. It has been removed.
    oc_controller = ObservedControl(
        dynamic_system=dynamic_system,
        anticipated_conditions=[(1.0, anticipated_condition)],
        expected_update_period=0.05,
        min_horizon=10,
        max_horizon=250,
        adaptive_tolerances_trace_p=1e-3,
        adaptive_tolerances_gamma=5e-3,
        delta_control_penalty=0.1 * np.eye(n_u),
    )

    # 4. Set up and Run the Simulation
    # ---------------------------------
    sim_time = 5
    dt = oc_controller.expected_update_period
    num_steps = int(sim_time / dt)
    x_current = dynamic_system.origin()

    time_history = [0.0]
    state_history = [x_current]
    current_control = np.zeros(n_u)
    control_history = []

    print("Running cart-pole swing-up simulation...")
    for i in range(num_steps):
        optimal_control, info = oc_controller.control_law(
            current_time=i * dt,
            initial_state=x_current,
            initial_control=current_control
        )
        print(f"sim time: {i * dt:.3f}", f"\tcompute_time: {info['compute_time'] * 1000:.1f}ms",
              f"\thorizon_len: {info['final_horizon']:d}")
        x_next, _, _ = dynamic_system.solve(t_init=i * dt, t_final=(i + 1) * dt, x_init=x_current, u=optimal_control)

        state_history.append(x_next)
        control_history.append(optimal_control[0])
        time_history.append((i + 1) * dt)

        x_current = x_next
        current_control = optimal_control

    print("Simulation complete.")

    # 5. Animate and Plot the Results
    # -------------------------------
    state_history = np.array(state_history)
    time_history = np.array(time_history)

    animate_cart_pole(time_history, state_history, dynamic_system.l)

    control_history = np.array(control_history)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Cart-Pole Swing-Up State and Control History", fontsize=16)

    axs[0].plot(time_history, state_history[:, 0], label="Cart Position (m)")
    axs[0].plot(time_history, state_history[:, 1], label="Cart Velocity (m/s)")
    axs[0].set_ylabel("Cart State")
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    pole_angle_deg = np.rad2deg(np.unwrap(state_history[:, 2]))
    axs[1].plot(time_history, pole_angle_deg, label="Pole Angle (deg)")
    axs[1].set_ylabel("Pole Angle")
    axs[1].grid(True)
    axs[1].legend(loc='upper right')

    axs[2].plot(time_history[:-1], control_history, label="Control Force (N)", drawstyle='steps-post')
    axs[2].set_ylabel("Control Input")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True)
    axs[2].legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
