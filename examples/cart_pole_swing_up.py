# examples/cart_pole_swing_up.py
import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.systems.cart_pole_system import CartPoleSystem
from src.core.anticipated_condition import AnticipatedCondition
from src.core.observed_control import ObservedControl

def main():
    """
    Runs a swing-up and stabilization simulation for the Cart-Pole system
    and creates an animation of the result.
    """
    # 1. Define the Dynamic System
    # ---------------------------
    dynamic_system = CartPoleSystem()
    n_x, n_u = dynamic_system.n_x, dynamic_system.n_u

    # 2. Define the Anticipated Condition (Cost Function)
    # ---------------------------------------------------
    x_target = np.array([0.0, 0.0, 0.0, 0.0])
    Q = np.diag([0.1, 0.01, 10.0, 0.05])
    R = np.diag([0.0])

    def swing_up_cost(t, x, u):
        """A cost function designed for the swing-up maneuver."""
        error = x - x_target
        # angle_cost = (1 - np.cos(error[2]))**2
        # other_states_cost = error[0]**2 + error[1]**2 + error[3]**2
        # control_cost = u.T @ R @ u
        # return 10 * angle_cost + other_states_cost + control_cost
        return error.T @ Q @ error + u.T @ R @ u

    anticipated_condition = AnticipatedCondition(
        n_x=n_x, n_u=n_u, value_func=swing_up_cost
    )

    # 3. Configure and Instantiate the Observed Controller
    # ----------------------------------------------------
    oc_controller = ObservedControl(
        dynamic_system=dynamic_system,
        anticipated_conditions=[(1.0, anticipated_condition)],
        expected_update_period=0.1,
        min_horizon=25,
        max_horizon=100,
        adaptive_tolerances_trace_p=1e-5,
        adaptive_tolerances_gamma=1e-3,
        process_noise=1 * np.eye(n_u),
    )

    # 4. Set up and Run the Simulation
    # ---------------------------------
    sim_time = 2.0
    dt = oc_controller.expected_update_period
    num_steps = int(sim_time / dt)
    x_current = np.array([0.0, 0.0, np.pi, 0.0])

    state_history = [x_current]
    current_control = np.zeros(n_u)
    print("Running cart-pole swing-up simulation...")
    for i in range(num_steps):
        optimal_control, info = oc_controller.control_law(
            current_time=i * dt,
            initial_state=x_current,
            initial_control=current_control
        )
        print(info)
        x_next, _, _ = dynamic_system.solve(0, dt, x_current, optimal_control)
        state_history.append(x_next)
        x_current = x_next
        current_control = optimal_control
    print("Simulation complete.")

    # 5. Create the Animation
    # -----------------------
    state_history = np.array(state_history)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(xlim=(-2.5, 2.5), ylim=(-0.5, 1.5))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--')

    # Static elements
    ax.plot([-2.5, 2.5], [0, 0], 'k-', lw=2) # Track

    # Animated elements
    cart_width = 0.4
    cart_height = 0.2
    cart = plt.Rectangle([0, 0], cart_width, cart_height, fc='royalblue', ec='k')

    pole_length = dynamic_system.l * 2 # Full pole length
    pole, = ax.plot([], [], 'o-', lw=3, color='goldenrod', markersize=8)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, va='top')

    def init():
        """Initializes the animation elements."""
        ax.add_patch(cart)
        pole.set_data([], [])
        time_text.set_text('')
        return cart, pole, time_text

    def animate(i):
        """Updates the animation elements for each frame."""
        cart_x = state_history[i, 0]
        theta = state_history[i, 2]

        # Update cart position
        cart.set_xy([cart_x - cart_width / 2, 0])

        # Update pole position
        pole_x = [cart_x, cart_x + pole_length * np.sin(theta)]
        pole_y = [cart_height / 2, cart_height / 2 + pole_length * np.cos(theta)]
        pole.set_data(pole_x, pole_y)

        time_text.set_text(f'Time: {i * dt:.2f}s')
        return cart, pole, time_text

    # Create and run the animation
    # The interval is dt in milliseconds
    anim = FuncAnimation(fig, animate, init_func=init,
                           frames=len(state_history), interval=dt * 1000,
                           blit=True, repeat=False)

    plt.title("Cart-Pole Swing-Up Animation", fontsize=16)
    plt.show()

if __name__ == "__main__":
    main()


# # examples/cart_pole_swing_up.py
# import autograd.numpy as np
# import matplotlib.pyplot as plt
#
# from systems.cart_pole_system import CartPoleSystem
# from core.anticipated_condition import AnticipatedCondition
# from core.observed_control import ObservedControl
#
# def main():
#     """
#     Runs a swing-up and stabilization simulation for the Cart-Pole system
#     using the Observed Control algorithm.
#     """
#     # 1. Define the Dynamic System
#     # ---------------------------
#     dynamic_system = CartPoleSystem()
#     n_x, n_u = dynamic_system.n_x, dynamic_system.n_u
#
#     # 2. Define the Anticipated Condition (Cost Function)
#     # ---------------------------------------------------
#     # The goal is to bring the cart to the origin (x=0) and have the pole
#     # upright (theta=0) with zero velocity.
#     x_target = np.array([0.0, 0.0, 0.0, 0.0])
#
#     # Weighting matrices for state error (Q) and control effort (R)
#     Q = np.diag([0.1, 0.1, 10.0, 0.1])
#     R = np.diag([0.01])
#
#     def swing_up_cost(t, x, u):
#         """
#         A cost function designed for the swing-up maneuver.
#         It heavily penalizes the pole not being upright.
#         """
#         error = x - x_target
#
#         # We need a special term for the angle to handle the wrap-around at pi.
#         # Penalizing (1 - cos(theta)) makes the cost landscape smooth and
#         # drives theta towards 0 (or 2*pi, etc.), which is the upright position.
#         # angle_cost = (1 - np.cos(error[2]))**2
#         #
#         # # Combine with a standard quadratic cost on other states
#         # other_states_cost = error[0]**2 + error[1]**2 + error[3]**2
#         #
#         # control_cost = u.T @ R @ u
#         #
#         # return 10 * angle_cost + other_states_cost + control_cost
#         return error.T @ Q @ error + u.T @ R @ u
#
#     anticipated_condition = AnticipatedCondition(
#         n_x=n_x, n_u=n_u, value_func=swing_up_cost
#     )
#
#     # 3. Configure and Instantiate the Observed Controller
#     # ----------------------------------------------------
#     oc_controller = ObservedControl(
#         dynamic_system=dynamic_system,
#         anticipated_conditions=[(1.0, anticipated_condition)],
#         expected_update_period=0.1, # A smaller timestep for the nonlinear system
#         min_horizon=10,
#         max_horizon=150,
#         adaptive_tolerances_trace_p=1e-5,
#         adaptive_tolerances_gamma=1e-3,
#         process_noise=3 * np.eye(n_u),
#     )
#
#     # 4. Set up and Run the Simulation
#     # ---------------------------------
#     sim_time = 6.0
#     dt = oc_controller.expected_update_period
#     num_steps = 20#int(sim_time / dt)
#
#     # Initial state: cart at origin, pole hanging down
#     x_current = np.array([0.0, 0.0, np.pi-.001, 0.0])
#
#     # Store history for plotting
#     time_history = [0.0]
#     state_history = [x_current]
#     control_history = []
#
#     print("Running cart-pole swing-up simulation...")
#     for i in range(num_steps):
#         # Compute the optimal control action
#         optimal_control, info = oc_controller.control_law(
#             current_time=i * dt,
#             initial_state=x_current,
#             initial_control=np.zeros(n_u)
#         )
#         print(info)
#
#         # Apply control and get next state
#         x_next, _, _ = dynamic_system.solve(0, dt, x_current, optimal_control)
#
#         # Store results
#         control_history.append(optimal_control)
#         state_history.append(x_next)
#         time_history.append((i + 1) * dt)
#         x_current = x_next
#     print("Simulation complete.")
#
#     # 5. Plot the Results
#     # -------------------
#     state_history = np.array(state_history)
#     control_history = np.array(control_history)
#     time_history = np.array(time_history)
#
#     fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
#     fig.suptitle("Cart-Pole Swing-Up with Observed Control", fontsize=16)
#
#     # Plot cart position and velocity
#     axs[0].plot(time_history, state_history[:, 0], label="Cart Position (m)")
#     axs[0].plot(time_history, state_history[:, 1], label="Cart Velocity (m/s)")
#     axs[0].set_ylabel("Cart State")
#     axs[0].legend()
#     axs[0].grid(True)
#
#     # Plot pole angle (convert to degrees for intuitive plotting)
#     pole_angle_deg = np.rad2deg(np.unwrap(state_history[:, 2]))
#     axs[1].plot(time_history, pole_angle_deg, label="Pole Angle (deg)")
#     axs[1].set_ylabel("Pole Angle")
#     axs[1].legend()
#     axs[1].grid(True)
#
#     # Plot control input
#     axs[2].plot(time_history[:-1], control_history, label="Control Force (N)", color='g')
#     axs[2].set_xlabel("Time (s)")
#     axs[2].set_ylabel("Control Input")
#     axs[2].legend()
#     axs[2].grid(True)
#
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.show()
#
# if __name__ == "__main__":
#     main()
