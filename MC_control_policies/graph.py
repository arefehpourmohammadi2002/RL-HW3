import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  


DEALER_LABELS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
PLAYER_SUMS = np.arange(12, 22)


def stick_boundary(policy_pi, ace):

    sticks = np.zeros((10, 10), dtype=bool)

    for s in range(10):
        for d in range(10): 
            sticks[s, d] = policy_pi[s, d, ace, 1] > policy_pi[s, d, ace, 0]  

    boundaries = np.full(10, 22)
    for d in range(10):
        boundary = 22
        for s in range(9, -1, -1): 
            if sticks[s, d]:
                boundary = s + 12 
            else:
                break
        boundaries[d] = boundary
    return boundaries



def plot_policy(policy_pi, ax, ace, title):
    boundaries = stick_boundary(policy_pi, ace)   # 10 values, A..10
    x = np.arange(1, 11)                          # 1..10 -> A,2,...,10
    y = np.array(boundaries) - 0.5

    ax.fill_between(x, y, 22, step='mid', color='0.85')
    ax.step(x, y, where='mid', linewidth=1.8)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(11, 21.5)

    ax.set_xticks(np.arange(0.5, 11.5, 1))

    ax.set_xticks(np.arange(1, 11), minor=True)
    ax.set_xticklabels(DEALER_LABELS, minor=True)

    ax.set_xticklabels([])

    ax.set_yticks(np.arange(10.5, 21.5, 1))

    ax.set_yticks(np.arange(11, 22), minor=True)
    ax.set_yticklabels(range(11, 22), minor=True)

    ax.set_yticklabels([])

    ax.set_xlabel("Dealer showing")
    ax.set_ylabel("Player sum")
    ax.set_title(title)

    ax.text(5.5, 20.3, "STICK",
            ha="center", va="center", fontsize=11)
    ax.text(5.5, 13.5, "HIT",
            ha="center", va="center", fontsize=11)
    ax.set_box_aspect(0.5)
    ax.tick_params(axis='both', which='major', length=4)
    ax.tick_params(axis='both', which='minor', length=0)


def plot_value_surface(action_value_fun, ax, ace, title):
    V = np.max(action_value_fun[:, :, ace, :], axis=-1)  
    X, Y = np.meshgrid(np.arange(1, 11), PLAYER_SUMS)     

    ax.plot_surface(X, Y, V, cmap='viridis', edgecolor='k', linewidth=0.2, antialiased=True)
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


def plot_blackjack_results(policy_pi, action_value_fun, save_path=None):

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    plot_policy(policy_pi, ax1, ace=1, title="$\\pi_*$ -- usable ace")

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_value_surface(action_value_fun, ax2, ace=1, title="$v_*$ -- usable ace")

    ax3 = fig.add_subplot(2, 2, 3)
    plot_policy(policy_pi, ax3, ace=0, title="$\\pi_*$ -- no usable ace")

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    plot_value_surface(action_value_fun, ax4, ace=0, title="$v_*$ -- no usable ace")

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")

    return fig
