# Copyright (c) 2025 Andrew Petruska
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Button

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
