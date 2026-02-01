import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from riemannpy.manifold import Manifold


if __name__ == "__main__":
    
    # Load points
    test_path: str = str(Path(__file__).resolve().parent)
    points: np.array = np.loadtxt(test_path + '/assets/bunny.txt')

    # Transform coords
    transform: np.array = np.array([
        [1, 0, 0],
        [0, 0, 1], 
        [0, 1, 0]
    ])
    
    points = points @ transform.T
    points = points[:1000] # subsample

    # Compute manifold
    manifold: Manifold = Manifold(points)

    # Point coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Normal component
    u = manifold.normal_vectors[:, 0]
    v = manifold.normal_vectors[:, 1]
    w = manifold.normal_vectors[:, 2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='blue', s=0.25)
    ax.quiver(x, y, z, u, v, w, normalize=False, color='red', linewidth=0.3)
    
    ax.set_box_aspect([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ])

    ax.set_axis_off()
    ax.grid(False)

    plt.show()