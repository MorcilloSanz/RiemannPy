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
    points = points[:500]  # subsample

    # Compute manifold
    manifold: Manifold = Manifold(points)

    # Point coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Tangent bundle
    t1 = manifold.tangent_vectors[:, 0, :]  # First tangent vectors
    t2 = manifold.tangent_vectors[:, 1, :]  # Second tangent vectors

    # First tangent vector components
    u1 = t1[:, 0]
    v1 = t1[:, 1]
    w1 = t1[:, 2]

    # Second tangent vector components
    u2 = t2[:, 0]
    v2 = t2[:, 1]
    w2 = t2[:, 2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='blue', s=0.25)

    ax.quiver(x, y, z, u1, v1, w1, normalize=False, color='green', linewidth=0.3)
    ax.quiver(x, y, z, u2, v2, w2, normalize=False, color='orange', linewidth=0.3)
    
    ax.set_box_aspect([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ])

    ax.set_axis_off()
    ax.grid(False)

    plt.show()