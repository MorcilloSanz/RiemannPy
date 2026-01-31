import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold
from pydiffold.field import ScalarField


ALPHA = 0.2
DELTA_T = 1.0
ITERATIONS = 10


def solve_equation(phi: ScalarField) -> None:
    phi.values = phi.values + DELTA_T * ALPHA * phi.laplacian


if __name__ == "__main__":
    
    test_path = str(Path(__file__).resolve().parent)
    
    # Load manifold
    points = np.loadtxt(test_path + '/assets/bunny.txt')
    manifold = Manifold(points)
    
    # Scalar field at t=0
    values = np.sin(manifold.points[:, 0] * 1.5) + np.cos(manifold.points[:, 1] * 1.5)
    phi = ScalarField(manifold, values)
        
    # List to store the state at each step for plotting
    history = [phi.values.copy()]

    # Solve the equation
    for t in range(ITERATIONS):
        solve_equation(phi)
        history.append(phi.values.copy())
        
    # --- Plotting Section ---
    n_plots = len(history)
    cols = 4
    rows = (n_plots + cols - 1) // cols
    
    fig = plt.figure(figsize=(18, 4 * rows))
    
    # Set global color limits so the scale remains constant across all plots
    v_min = min(h.min() for h in history)
    v_max = max(h.max() for h in history)

    for i, values in enumerate(history):
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        
        # Scatter plot of the bunny points
        scatter = ax.scatter(
            manifold.points[:, 0], 
            manifold.points[:, 1], 
            manifold.points[:, 2], 
            c=values, 
            cmap='magma', 
            s=1.5, # Point size
            vmin=v_min, 
            vmax=v_max
        )
        
        ax.set_title(f"Iteration {i}")
        ax.set_axis_off() # Hide axes for a cleaner look
        ax.view_init(elev=15, azim=45) # Set a nice viewing angle

    # Add a global colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    fig.colorbar(scatter, cax=cbar_ax, label='Temperature ($\phi$)')
    
    plt.suptitle("Heat Equation Evolution on Bunny Manifold", fontsize=18)
    plt.subplots_adjust(right=0.9, wspace=0.05, hspace=0.2)
    
    plt.show()
    