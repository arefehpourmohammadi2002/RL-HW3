import numpy as np
import matplotlib.pyplot as plt

DEALER_LABELS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
PLAYER_SUMS = np.arange(12, 22)


def plot_value_surface(V, ax, ace, title):

    # ensure correct slicing
    V_2d = np.array(V[:, :, ace])

    # fix orientation if needed
    if V_2d.shape != (len(PLAYER_SUMS), 10):
        V_2d = V_2d.T

    X, Y = np.meshgrid(np.arange(1, 11), PLAYER_SUMS)

    ax.plot_surface(
        X, Y, V_2d,
        cmap='viridis',
        edgecolor='k',
        linewidth=0.2,
        antialiased=True
    )

    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_zlabel("Value")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(DEALER_LABELS)
    ax.set_zlim(-1, 1)
    ax.set_title(title)
    ax.set_zticks([-1, 1])
    ax.set_box_aspect((1, 1, 0.2))
    ax.view_init(elev=25, azim=-60)


def plot_blackjack_results(V_bad, V_good, save_path=None):

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_value_surface(V_bad, ax1, ace=1, title="$v_*$ -- usable ace (10000)")

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_value_surface(V_good, ax2, ace=1, title="$v_*$ -- usable ace (500000)")

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    plot_value_surface(V_bad, ax3, ace=0, title="$v_*$ -- no usable ace")

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_value_surface(V_good, ax4, ace=0, title="$v_*$ -- no usable ace")

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")

    return fig