import numpy as np


def k_neighbors(points, p0, k=50) -> np.array:
    """
    Returns the k nearest neighbors of a given point from a set of points.

    Args:
        points (np.array): Array of shape (n_points, n_dimensions) representing the dataset of points.
        p0 (np.array): Array of shape (n_dimensions,) representing the query point.
        k (int, optional): Number of nearest neighbors to return. Defaults to 50.

    Returns:
        np.array: Array of shape (k, n_dimensions) containing the k nearest points to `p0`.
    """
    d = np.linalg.norm(points - p0, axis=1)
    idx = np.argsort(d)[:k]
    return points[idx]