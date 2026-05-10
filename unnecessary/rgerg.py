import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

def plot_gradient_grid():
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- Configuration ---
    grid_size = 11       # 11x11 grid to match the image
    pad_width = 0.6
    gap = 0.2
    base_margin = 0.8

    # Calculate dimensions
    total_grid_span = (grid_size * pad_width) + ((grid_size - 1) * gap)
    base_width = total_grid_span + (2 * base_margin)

    # Heights
    base_thickness = 0.2
    pad_thickness = 0.05

    # --- 1. Draw the Grey Base ---
    # Centering the base at (0,0) for easier math
    offset = base_width / 2
    ax.bar3d(-offset, -offset, 0, base_width, base_width, base_thickness,
             color='#B0B5C0', edgecolor='#808080', shade=False)

    # --- 2. Draw the Gradient Pads ---
    center_idx = grid_size // 2
    max_dist = np.sqrt(center_idx**2 + center_idx**2) # Max possible distance from center

    # Colormap: YlOrBr (Yellow-Orange-Brown) matches your image well
    # We want center = dark (high value) and edge = light (low value)
    colormap = cm.get_cmap('YlOrBr')

    for row in range(grid_size):
        for col in range(grid_size):
            # Calculate distance from the center (row, col)
            dist = np.sqrt((row - center_idx)**2 + (col - center_idx)**2)

            # Normalize distance (0.0 to 1.0)
            # We invert it so closer to center = higher value (darker color)
            norm_dist = 1.0 - (dist / (max_dist * 1.2)) # 1.2 buffer to keep corners from being white

            # Get color from colormap
            # We add a minimum floor so the edges aren't pure white
            color_val = colormap(norm_dist * 0.8 + 0.1)

            # Calculate physical position
            x = (col * (pad_width + gap)) - (total_grid_span / 2) + (pad_width/2)
            y = (row * (pad_width + gap)) - (total_grid_span / 2) + (pad_width/2)

            # Draw the pad
            ax.bar3d(x, y, base_thickness, pad_width, pad_width, pad_thickness,
                     color=color_val, edgecolor=None, shade=False)

            # Add a thin dark edge to simulate the border seen in image
            ax.plot([x, x+pad_width, x+pad_width, x, x],
                    [y, y, y+pad_width, y+pad_width, y],
                    [base_thickness+pad_thickness]*5, color='grey', linewidth=0.5)

    # --- 3. View Settings ---
    ax.set_axis_off()

    # Top-down view (90 degrees elevation) to match the 2D look
    ax.view_init(elev=90, azim=-90)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_gradient_grid()
