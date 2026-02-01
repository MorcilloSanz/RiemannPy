import numpy as np


def fit_plane(points: np.array) -> tuple[np.array, np.array]:
    """
    Plane fitting using SVD minimizing orthogonal distance.

    Args:
        points: array (N,3)

    Returns: 
        unit normal (3,), point on the plane p0 (3,)
        Plane equation: normal . (X - p0) = 0
    """
    points = np.asarray(points)
    assert points.shape[1] == 3, "points must have shape (N,3)"

    # Centroid
    p0 = points.mean(axis=0)
    
    # Center points
    X = points - p0

    # SVD
    u, s, vh = np.linalg.svd(X, full_matrices=False)

    # The normal is the last row of V^T (last singular vector)
    normal = vh[-1, :]
    
    # Normalize
    normal = normal / np.linalg.norm(normal)

    return normal, p0


def fit_paraboloid(points: np.array) -> np.array:
    """
    Fit a paraboloid of the form:
        z = a x^2 + b y^2 + c xy + d x + e y + f

    Args:
        points: array (N,3)

    Returns:
        params: np.array with [a, b, c, d, e, f]

    Resulting equation:
        z = a x^2 + b y^2 + c xy + d x + e y + f
    """
    points = np.asarray(points)
    assert points.shape[1] == 3, "points must have shape (N,3)"

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Build the design matrix
    # Each row: [x^2, y^2, x*y, x, y, 1]
    A = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])

    # Solve using SVD (least squares)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    params = Vt.T @ (np.linalg.pinv(np.diag(S)) @ (U.T @ z))

    return params
