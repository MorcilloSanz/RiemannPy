import numpy as np


def local_frame_from_normal(n: np.array) -> tuple[np.array, np.array, np.array]:
    """
    Construct an orthonormal local reference frame (u, v, w) from a normal vector.

    Given a normal vector `n`, this function generates two tangent vectors `u`
    and `v` such that {u, v, n} forms a right-handed orthonormal basis.

    The generated frame satisfies:
        - w = normalized(n)
        - u ⟂ w
        - v = w × u
        - {u, v, w} is orthonormal

    This frame is commonly used to define a local coordinate system on a surface
    where w is the surface normal, and u, v span the tangent plane.

    Args:
        n : array-like of shape (3,)
        Normal vector at the surface point.

    Returns:
        u : ndarray of shape (3,)
            First tangent unit vector.
        v : ndarray of shape (3,)
            Second tangent unit vector, perpendicular to u and w.
        w : ndarray of shape (3,)
            Normalized normal vector, used as local z-axis.
    """
    n = n / np.linalg.norm(n)

    # Pick a non-parallel vector to construct u
    if abs(n[0]) < abs(n[2]):
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 0.0, 1.0])

    u = np.cross(n, tmp)
    u /= np.linalg.norm(u)

    v = np.cross(n, u)

    return u, v, n


def rotation_from_frame(u: np.array, v: np.array, w: np.array) -> np.array:
    """
    Build a rotation matrix from an orthonormal frame.

    The matrix uses the vectors (u, v, w) as the rows of the rotation matrix.
    Applying this matrix to a point brings it into the local coordinate system
    whose axes are defined by {u, v, w}.

    Formally:
        R = [u^T
             v^T
             w^T]

    So that:
        X_local = R @ (X - p0)

    Args:
        u: array-like of shape (3,)
        w: array-like of shape (3,)
        v: array-like of shape (3,)
        Orthonormal basis vectors forming the local reference frame.

    Returns:
        R : ndarray of shape (3,3)
            Rotation matrix that maps world coordinates into local coordinates.
    """
    return np.vstack([u, v, w])


def transform_to_local_coordinates(points: np.array, p0: np.array, normal: np.array):
    """
    Transform 3D points into a local coordinate system aligned with the surface normal.

    Steps:
        1. Compute an orthonormal local frame (u, v, w), where w is the normal.
        2. Build the rotation matrix R from (u, v, w).
        3. Translate all points so that p0 becomes the origin.
        4. Rotate the translated points into the local coordinate system.

    The resulting coordinates have:
        - z_local ≈ 0 for points lying on the local tangent plane
        - x_local, y_local spanning the tangent plane
        - z_local measuring deviation along the normal direction

    Args:
        points : ndarray of shape (N,3)
            Input 3D points in global coordinates.
        p0 : array-like of shape (3,)
            Surface point around which the local system is defined.
        normal : array-like of shape (3,)
            Normal vector at p0.

    Returns:
        X_local : ndarray of shape (N,3)
            Points expressed in the local coordinate system.
        R : ndarray of shape (3,3)
            Rotation matrix used for the transformation.
        (u, v, w) : tuple of ndarrays
            The orthonormal frame defining the local coordinate system.
    """
    # Build local frame
    u, v, w = local_frame_from_normal(normal)

    # Rotation matrix
    R = rotation_from_frame(u, v, w)

    # Center points around p0
    X_centered = points - p0

    # Rotate into local coordinates
    X_local = (R @ X_centered.T).T

    return X_local, R, (u, v, w)