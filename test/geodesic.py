import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold


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

    # Compute manifold
    manifold: Manifold = Manifold(points)
    
    start_point_index = 140
    end_point_index = 600
    
    geodesic, arc_length = manifold.geodesic(start_point_index, end_point_index)
    geodesic_coords: np.array = manifold.points[geodesic]
    
    print(f'\033[1;32mGeodesic of arc length {arc_length}\033[0m γ: {geodesic}')
    print(f'\033[1;32mGeodesic vertex coordinates\033[0m γ: {geodesic_coords}')

    # Point coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    gx = geodesic_coords[:, 0]
    gy = geodesic_coords[:, 1]
    gz = geodesic_coords[:, 2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title(
        f"Geodesic arc length: {arc_length:.2f}",
        fontsize=14,
        fontweight='bold',
        pad=12
    )

    ax.scatter(x, y, z, c='black', s=0.1)
    
    ax.plot(
        gx, gy, gz,
        c='red',
        linewidth=3.5)
    
    px, py, pz = points[start_point_index]
    ax.scatter(px, py, pz,
            c='red',
            s=40,
            marker='o',
            linewidths=1.5)
    
    px, py, pz = points[end_point_index]
    ax.scatter(px, py, pz,
            c='red',
            s=40,
            marker='o',
            linewidths=1.5)
    
    ax.set_box_aspect([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ])

    ax.set_axis_off()
    ax.grid(False)

    plt.show()