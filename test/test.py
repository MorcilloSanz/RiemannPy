import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from pydiffold.manifold import Manifold


def sample_sphere(n_points=1000, radius=1.0) -> np.array:
    """Generate uniformly distributed random points on the surface of a sphere.

    Args:
        n_points (int, optional): Number of points to sample. Defaults to 1000.
        radius (float, optional): Radius of the sphere. Defaults to 1.0.

    Returns:
        numpy.ndarray: Array of shape (n_points, 3) containing the sampled 3D points
        on the sphere's surface.
    """
    # Random angles
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    cos_theta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(cos_theta)

    # Convert spherical to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * cos_theta

    return np.vstack((x, y, z)).T


if __name__ == "__main__":

    points = sample_sphere(n_points=5000, radius=100)
    manifold = Manifold(points)
    
    print(f'\033[1;32mMetric tensor\033[0m g_μν(p):\n {manifold.metric_tensor}\n')
    print(f'\033[1;32mMetric tensor inverse\033[0m g^μν(p):\n {manifold.metric_tensor}\n')
    print(f'\033[1;32mMetric tensor derivatives\033[0m ∂_α g_μν(p):\n {manifold.metric_tensor_derivatives}\n')