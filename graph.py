import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

def create_blackjack_policy_chart(policy):
    policy_wo_useable_ace = policy[:, :, 0, :]   # No loop needed, NumPy slicing is cleaner!
    policy_with_useable_ace = policy[:, :, 1, :]

    # --- 2. Dynamically extract the boundary ---
    # We will create step-plot arrays for Matplotlib

    x_line = []
    y_line = []

    # Loop through each dealer card column (j = 0 is 'A', j = 1 is '2', ..., j = 9 is '10')
    for j in range(10):
        # Find the indices where the policy says HIT (action == 1)
        hit_indices = np.where(policy_wo_useable_ace[:, j, 1] == 1)[0]
        
        if len(hit_indices) > 0:
            # The highest index where we hit. 
            # Map index to actual player score: index 0 is 11, index 1 is 12, etc.
            max_hit_idx = np.max(hit_indices)
            highest_hit_score = max_hit_idx + 11 
            
            # The boundary line sits exactly between the highest HIT score and the lowest STICK score
            boundary_y = highest_hit_score + 0.5
        else:
            # If the policy never hits for this column, boundary sits below the minimum score (11)
            boundary_y = 10.5 
            
        # To make a proper step-like look in Matplotlib, we define the line 
        # spanning from the left edge to the right edge of the current column.
        x_start = j + 0.5
        x_end = j + 1.5
        
        # Add horizontal segment for this column
        x_line.extend([x_start, x_end])
        y_line.extend([boundary_y, boundary_y])

    # --- 3. Plotting ---
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x_line, y_line, color='black', linewidth=1.5)

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_right()

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)

    x_ticks = list(range(1, 11))
    x_labels = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=12)

    y_ticks = list(range(11, 22))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontsize=12)

    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(10.5, 21.5)
    ax.tick_params(direction='out', length=5, width=1)

    ax.text(5.5, 20.0, 'STICK', fontsize=18, ha='center', va='center')
    ax.text(5.5, 14.5, 'HIT', fontsize=18, ha='center', va='center')

    # --- 5. Save the Plot ---
    plt.tight_layout()
    plt.savefig('blackjack_strategy.png', dpi=300, bbox_inches='tight')
    print("Plot successfully saved as blackjack_strategy.png")