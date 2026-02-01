import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(str(Path(__file__).resolve().parent.parent))
from riemannpy.manifold import Manifold
from riemannpy.field import ScalarField

ALPHA = 0.25
DELTA_T = 1.0
TOTAL_STEPS = 100
PLOTS_TO_SHOW = 5

def solve_equation(phi: ScalarField) -> None:
    phi.values = phi.values - DELTA_T * ALPHA * phi.laplacian

if __name__ == "__main__":
    
    # Load Standford Bunny
    test_path = str(Path(__file__).resolve().parent)
    points_file = Path(test_path) / "assets" / "bunny.txt"
    
    # Create Manifold
    points = np.loadtxt(str(points_file))
    transform = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    points = points @ transform.T
    manifold = Manifold(points)
    
    # Temperature scalar field at t=0
    init_values = np.sin(manifold.points[:, 0] * 1.5) + np.cos(manifold.points[:, 1] * 1.5)
    phi = ScalarField(manifold, init_values)
    
    # Solve equation
    history = []
    interval = TOTAL_STEPS // (PLOTS_TO_SHOW - 1)
    
    for t in range(TOTAL_STEPS + 1):
        if t % interval == 0:
            history.append((t, phi.values.copy()))
        solve_equation(phi)
        
    # --- Plotting ---
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig = plt.figure(figsize=(22, 6))
    
    v_min, v_max = history[0][1].min(), history[0][1].max()
    x_range = [points[:, 0].min(), points[:, 0].max()]
    y_range = [points[:, 1].min(), points[:, 1].max()]
    z_range = [points[:, 2].min(), points[:, 2].max()]

    for i, (step, state_values) in enumerate(history):
        
        ax = fig.add_subplot(1, 5, i + 1, projection='3d')
        
        scatter = ax.scatter(
            manifold.points[:, 0], manifold.points[:, 1], manifold.points[:, 2], 
            c=state_values, cmap='gist_heat', s=6.0, 
            vmin=v_min, vmax=v_max, antialiased=True
        )
        
        # Zoom
        ax.set_xlim(x_range); ax.set_ylim(y_range); ax.set_zlim(z_range)
        ax.set_box_aspect((x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0]))
        ax.dist = 7 
        
        ax.set_axis_off()
        ax.set_title(f"Step {step}\n$t = {step * DELTA_T}s$", pad=-15, fontsize=12)
        ax.view_init(elev=15, azim=90)

    #cbar_ax = fig.add_axes([0.93, 0.25, 0.01, 0.5])
    #fig.colorbar(scatter, cax=cbar_ax).set_label('Temperature ($\phi$)', rotation=270, labelpad=15)
    
    plt.suptitle("Heat Equation on the Stanford Bunny Manifold", fontsize=16, y=0.95)
    plt.subplots_adjust(left=0.01, right=0.91, wspace=0.0, hspace=0.0)
    
    plt.show()