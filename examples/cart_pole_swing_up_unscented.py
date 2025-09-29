# examples/cart_pole_swing_up.py
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Button

# Adjust the import path to match your project structure
from systems.cart_pole_system import CartPoleSystem
from core.unscented_observed_control import UnscentedObservedControl
from conditions.quadratic_cost import QuadraticCost

matplotlib.use("Qt5Agg")


def animate_cart_pole(time_history, state_history, l):
    """Creates and runs an animation of the cart-pole system."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.5, 2.5)
    ax.set_title("Cart-Pole Swing-Up Animation")
    ax.set_xlabel("Position (m)")

    cart_width = 0.125
    cart_height = 0.0625
    pole_radius = 0.02
    pole_length = 2 * l

    dt = time_history[1] - time_history[0]

    # Pre-calculate pole end-points for the entire trajectory
    cart_x_history = state_history[:, 0]
    theta_history = state_history[:, 2]
    pole_x_history = cart_x_history - pole_length * np.sin(theta_history)
    pole_y_history = pole_length * np.cos(theta_history)

    # Artists that will be updated in each frame
    frame_artists = []
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # Ground line
    ax.plot([-2.5, 2.5], [0, 0], 'k-', lw=1, zorder=1)

    def plot_single_cart(frame_index):
        """
        Clears the old artists and draws a single, detailed cart-pole state.
        """
        nonlocal frame_artists
        for artist in frame_artists:
            artist.remove()
        frame_artists = []

        cart_x = cart_x_history[frame_index]
        theta = theta_history[frame_index]
        pole_x = pole_x_history[frame_index]
        pole_y = pole_y_history[frame_index]

        cart_color = 'black'
        outline_color = 'black'
        linewidth = 1.5

        origin = (cart_x - (cart_width / 2.0), -(cart_height / 2.0))
        rectangle = Rectangle(origin, cart_width, cart_height, edgecolor=outline_color, facecolor=cart_color,
                              linewidth=linewidth, zorder=10)
        circle = Circle((pole_x, pole_y), pole_radius,
                        edgecolor=outline_color, facecolor=cart_color, linewidth=linewidth,
                        zorder=10)

        pole_x_points = [cart_x, pole_x + pole_radius * np.sin(theta)]
        pole_y_points = [0, pole_y - pole_radius * np.cos(theta)]

        line1, = ax.plot(pole_x_points, pole_y_points, linewidth=5, color=outline_color, zorder=8)
        line2, = ax.plot(pole_x_points, pole_y_points, linewidth=3, color=cart_color, zorder=9)

        ax.add_patch(rectangle)
        ax.add_patch(circle)
        frame_artists.extend([rectangle, circle, line1, line2])

    def init():
        """Initializes the animation by drawing the first frame."""
        plot_single_cart(0)
        time_text.set_text('')
        return frame_artists + [time_text]

    def animate(i):
        """Update artists for each animation frame."""
        plot_single_cart(i)
        time_text.set_text(f'Time: {time_history[i]:.2f}s')

        if i == len(time_history) - 1:  # stop at end
            anim.event_source.stop()

        return frame_artists + [time_text]

    anim = FuncAnimation(fig, animate, frames=len(time_history),
                         init_func=init, blit=False, interval=dt * 1000,
                         repeat=True)

    def reset_animation(event):
        """Callback to reset and restart the animation."""
        anim.frame_seq = anim.new_frame_seq()  # Reset frame counter
        anim.event_source.start()  # Restart the animation timer

    ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(ax_reset, 'Replay', hovercolor='0.975')
    button.on_clicked(reset_animation)

    plt.show()


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
    oc_controller = UnscentedObservedControl(
        dynamic_system=dynamic_system,
        anticipated_conditions=[(1.0, anticipated_condition)],
        expected_update_period=0.02,
        min_horizon=35,
        max_horizon=100,
        adaptive_tolerances_trace_p=1e-3,
        adaptive_tolerances_gamma=1e-3,
        delta_control_penalty=0.001 * np.eye(n_u),
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
